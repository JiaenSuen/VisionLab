"""
mamba.py
========

Pure PyTorch Mamba building blocks designed to run on native Windows 11.

Dependencies:
    pip install torch

This implementation intentionally avoids:
    - mamba-ssm
    - causal-conv1d
    - Triton
    - custom CUDA/C++ extensions

It implements the mathematical Mamba recurrence in ordinary PyTorch.  It is
therefore suitable for study, verification, CPU execution, and native Windows,
but it is considerably slower than the official fused selective-scan kernel.

Tensor convention throughout this file:
    sequence tensors: [batch, length, channels]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class MambaConfig:
    """Configuration for one unidirectional Mamba mixer."""

    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: float = 2.0
    dt_rank: Optional[int] = None
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False

    @property
    def d_inner(self) -> int:
        return max(1, int(self.d_model * self.expand))

    @property
    def resolved_dt_rank(self) -> int:
        # Same common rule used by Mamba-1 implementations.
        return self.dt_rank or math.ceil(self.d_model / 16)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= drop_prob < 1.0:
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x * random_tensor / keep_prob


class RMSNorm(nn.Module):
    """Root-mean-square normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        source_dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float.to(source_dtype) * self.weight)


def inverse_softplus(x: Tensor) -> Tensor:
    """Numerically stable inverse of softplus for positive x."""
    return x + torch.log(-torch.expm1(-x))


def selective_scan_ref(
    u: Tensor,
    delta: Tensor,
    A: Tensor,
    B: Tensor,
    C: Tensor,
    D: Optional[Tensor] = None,
    delta_bias: Optional[Tensor] = None,
    delta_softplus: bool = True,
    return_last_state: bool = False,
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Pure PyTorch reference selective scan.

    Recurrence:
        h_t = exp(delta_t * A) * h_(t-1)
              + delta_t * B_t * u_t
        y_t = sum_n(C_t * h_t) + D * u_t

    Shapes:
        u:     [B, L, D]
        delta: [B, L, D]
        A:     [D, N], normally negative
        B:     [B, L, N]
        C:     [B, L, N]
        D:     [D]
        output:[B, L, D]

    The recurrence is evaluated internally in float32 for stability and cast
    back to the input dtype at the end.
    """
    if u.ndim != 3:
        raise ValueError(f"u must have shape [B, L, D], got {tuple(u.shape)}")
    if delta.shape != u.shape:
        raise ValueError("delta must have the same shape as u.")

    batch, length, d_inner = u.shape
    if A.ndim != 2 or A.shape[0] != d_inner:
        raise ValueError("A must have shape [D, N].")
    d_state = A.shape[1]
    expected_bc = (batch, length, d_state)
    if B.shape != expected_bc or C.shape != expected_bc:
        raise ValueError(
            f"B and C must have shape {expected_bc}; "
            f"got B={tuple(B.shape)}, C={tuple(C.shape)}"
        )

    output_dtype = u.dtype
    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().view(1, 1, -1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    state = torch.zeros(
        batch, d_inner, d_state, device=u.device, dtype=torch.float32
    )
    outputs = []

    # Explicit recurrence: mathematically clear and fully autograd-compatible.
    for index in range(length):
        dt = delta_f[:, index, :]                    # [B, D]
        u_t = u_f[:, index, :]                       # [B, D]
        b_t = B_f[:, index, :]                       # [B, N]
        c_t = C_f[:, index, :]                       # [B, N]

        transition = torch.exp(dt.unsqueeze(-1) * A_f.unsqueeze(0))
        input_update = (
            dt.unsqueeze(-1)
            * b_t.unsqueeze(1)
            * u_t.unsqueeze(-1)
        )
        state = transition * state + input_update

        y_t = torch.sum(state * c_t.unsqueeze(1), dim=-1)
        if D is not None:
            y_t = y_t + D.float().view(1, -1) * u_t
        outputs.append(y_t)

    y = torch.stack(outputs, dim=1).to(output_dtype)
    if return_last_state:
        return y, state
    return y


class MambaMixer(nn.Module):
    """
    Mamba-1 style unidirectional selective state-space mixer.

    Pipeline:
        in projection -> x/z split
        -> causal depthwise Conv1d + SiLU
        -> input-dependent dt, B, C
        -> selective scan
        -> SiLU gate
        -> output projection
    """

    def __init__(self, config: MambaConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_inner = config.d_inner
        self.d_state = config.d_state
        self.dt_rank = config.resolved_dt_rank
        self.d_conv = config.d_conv

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=config.bias
        )

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=config.conv_bias,
        )

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A = -exp(A_log), giving stable negative continuous-time dynamics.
        a = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        a = a.unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=config.bias
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Initialize dt so softplus(dt_bias) lies approximately in [dt_min, dt_max].
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(self.d_inner)
            * (math.log(self.config.dt_max) - math.log(self.config.dt_min))
            + math.log(self.config.dt_min)
        ).clamp(min=self.config.dt_init_floor)

        with torch.no_grad():
            self.dt_proj.bias.copy_(inverse_softplus(dt))

    def _causal_depthwise_conv(self, x: Tensor) -> Tensor:
        # x: [B, L, D] -> Conv1d expects [B, D, L].
        length = x.shape[1]
        x = self.conv1d(x.transpose(1, 2))
        x = x[..., :length]
        return x.transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"Expected [B, L, {self.d_model}], got {tuple(x.shape)}"
            )

        x_branch, gate = self.in_proj(x).chunk(2, dim=-1)
        x_branch = F.silu(self._causal_depthwise_conv(x_branch))

        projected = self.x_proj(x_branch)
        dt_low_rank, B, C = torch.split(
            projected,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        delta = F.linear(dt_low_rank, self.dt_proj.weight, bias=None)

        A = -torch.exp(self.A_log.float())
        y = selective_scan_ref(
            u=x_branch,
            delta=delta,
            A=A,
            B=B,
            C=C,
            D=self.D,
            delta_bias=self.dt_proj.bias,
            delta_softplus=True,
        )

        y = y * F.silu(gate)
        return self.out_proj(y)


class BidirectionalMamba(nn.Module):
    """
    Bidirectional Mamba mixer for visual sequences.

    Two directional Mamba mixers are used:
      - forward mixer processes the original token order
      - backward mixer processes the reversed order

    The backward output is flipped back and fused with the forward output.
    Independent directional parameters are used, matching the intent of Vim's
    bidirectional selective state-space modeling more closely than sharing one
    unidirectional mixer.
    """

    def __init__(
        self,
        config: MambaConfig,
        merge_mode: str = "mean",
    ) -> None:
        super().__init__()
        if merge_mode not in {"mean", "sum"}:
            raise ValueError("merge_mode must be 'mean' or 'sum'.")
        self.forward_mixer = MambaMixer(config)
        self.backward_mixer = MambaMixer(config)
        self.merge_mode = merge_mode

    def forward(self, x: Tensor) -> Tensor:
        y_forward = self.forward_mixer(x)
        y_backward = self.backward_mixer(torch.flip(x, dims=(1,)))
        y_backward = torch.flip(y_backward, dims=(1,))

        y = y_forward + y_backward
        return y * 0.5 if self.merge_mode == "mean" else y


class MambaBlock(nn.Module):
    """Pre-normalized residual Mamba block."""

    def __init__(
        self,
        config: MambaConfig,
        bidirectional: bool = False,
        drop_path: float = 0.0,
        norm_eps: float = 1e-5,
        merge_mode: str = "mean",
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.d_model, eps=norm_eps)
        self.mixer = (
            BidirectionalMamba(config, merge_mode=merge_mode)
            if bidirectional
            else MambaMixer(config)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.drop_path(self.mixer(self.norm(x)))


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of model parameters."""
    parameters = module.parameters()
    if trainable_only:
        return sum(p.numel() for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


__all__ = [
    "MambaConfig",
    "DropPath",
    "RMSNorm",
    "selective_scan_ref",
    "MambaMixer",
    "BidirectionalMamba",
    "MambaBlock",
    "count_parameters",
]
