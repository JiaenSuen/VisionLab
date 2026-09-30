import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from econvnext import build_EConvNeXt_Mini, build_EConvNeXt_Small, build_EConvNeXt_Tiny


class VisualTokenProjector(nn.Module):
    """Convert E-ConvNeXt C4/C5 maps into multi-scale visual tokens."""

    def __init__(self, c4_dim: int, c5_dim: int, d_model: int = 256, pool_size: int = 7, dropout: float = 0.10):
        super().__init__()
        self.pool_size = int(pool_size)
        self.proj_c4 = nn.Conv2d(c4_dim, d_model, kernel_size=1, bias=False)
        self.proj_c5 = nn.Conv2d(c5_dim, d_model, kernel_size=1, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pos = nn.Parameter(torch.zeros(1, 2 * self.pool_size * self.pool_size, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, c4: torch.Tensor, c5: torch.Tensor) -> torch.Tensor:
        p = self.pool_size
        c4 = F.adaptive_avg_pool2d(self.proj_c4(c4), (p, p))
        c5 = F.adaptive_avg_pool2d(self.proj_c5(c5), (p, p))
        t4 = c4.flatten(2).transpose(1, 2).contiguous()
        t5 = c5.flatten(2).transpose(1, 2).contiguous()
        tokens = torch.cat([t4, t5], dim=1)
        tokens = self.norm(tokens + self.pos[:, : tokens.size(1)])
        return self.dropout(tokens)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResamplerBlock(nn.Module):
    """Perceiver/Q-Former-style visual resampling block.

    Learned visual queries cross-attend to CNN visual tokens, then self-attend among themselves.
    This keeps decoder cross-attention cheap because the language decoder attends to ~32 tokens.
    """

    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.10):
        super().__init__()
        self.q_norm = nn.LayerNorm(d_model)
        self.kv_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout)

    def forward(self, queries: torch.Tensor, visual_tokens: torch.Tensor) -> torch.Tensor:
        q = self.q_norm(queries)
        kv = self.kv_norm(visual_tokens)
        x = queries + self.cross_attn(q, kv, kv, need_weights=False)[0]
        s = self.self_norm(x)
        x = x + self.self_attn(s, s, s, need_weights=False)[0]
        x = x + self.ffn(self.ffn_norm(x))
        return x


class VisualResampler(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_queries: int = 32,
        num_layers: int = 2,
        nhead: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(1, num_queries, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.layers = nn.ModuleList([ResamplerBlock(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        b = visual_tokens.size(0)
        x = self.queries.expand(b, -1, -1)
        for layer in self.layers:
            x = layer(x, visual_tokens)
        return self.norm(x)


class EfficientCaptionDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        ffn_dim: int = 768,
        dropout: float = 0.15,
        max_len: int = 40,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.pad_idx = int(pad_idx)
        self.max_len = int(max_len)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

        layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_embed.weight

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, t = input_ids.shape
        if t > self.max_len:
            input_ids = input_ids[:, : self.max_len]
            t = self.max_len
        x = self.token_embed(input_ids) * math.sqrt(self.token_embed.embedding_dim)
        x = x + self.pos[:, :t]
        tgt_mask = self.causal_mask(t, input_ids.device)
        tgt_key_padding_mask = input_ids.eq(self.pad_idx)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        x = self.norm(x)
        return self.head(x)


class EConvNeXtCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        bos_idx: int,
        eos_idx: int,
        encoder_variant: str = "mini",
        d_model: int = 256,
        visual_pool_size: int = 7,
        visual_queries: int = 32,
        resampler_layers: int = 2,
        decoder_layers: int = 3,
        nhead: int = 4,
        ffn_dim: int = 768,
        dropout: float = 0.15,
        max_len: int = 40,
    ):
        super().__init__()
        encoder_variant = encoder_variant.lower().strip()
        builders = {
            "mini": build_EConvNeXt_Mini,
            "tiny": build_EConvNeXt_Tiny,
            "small": build_EConvNeXt_Small,
        }
        if encoder_variant not in builders:
            raise ValueError(f"Unknown encoder_variant: {encoder_variant}. Use mini, tiny, or small.")
        self.encoder = builders[encoder_variant](num_classes=1, img_channels=3)
        dims = list(self.encoder.out_channels)
        self.visual_projector = VisualTokenProjector(dims[-2], dims[-1], d_model=d_model, pool_size=visual_pool_size, dropout=dropout)
        self.resampler = VisualResampler(
            d_model=d_model,
            num_queries=visual_queries,
            num_layers=resampler_layers,
            nhead=nhead,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.decoder = EfficientCaptionDecoder(
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            d_model=d_model,
            nhead=nhead,
            num_layers=decoder_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_len=max_len,
            tie_weights=True,
        )
        self.pad_idx = int(pad_idx)
        self.bos_idx = int(bos_idx)
        self.eos_idx = int(eos_idx)
        self.max_len = int(max_len)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        features = self.encoder.forward_feature_maps(images)
        c4, c5 = features[-2], features[-1]
        visual_tokens = self.visual_projector(c4, c5)
        return self.resampler(visual_tokens)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        memory = self.encode_image(images)
        return self.decoder(input_ids, memory)

    @torch.no_grad()
    def generate_greedy(self, images: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        self.eval()
        max_len = int(max_len or self.max_len)
        memory = self.encode_image(images)
        b = images.size(0)
        ys = torch.full((b, 1), self.bos_idx, dtype=torch.long, device=images.device)
        finished = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            logits = self.decoder(ys, memory)[:, -1, :]
            logits[:, self.pad_idx] = -1e9
            logits[:, self.bos_idx] = -1e9
            next_token = logits.argmax(dim=-1)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_idx), next_token)
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished |= next_token.eq(self.eos_idx)
            if bool(finished.all()):
                break
        return ys

    @torch.no_grad()
    def generate_beam(self, images: torch.Tensor, max_len: Optional[int] = None, beam_size: int = 3, length_penalty: float = 0.70) -> torch.Tensor:
        self.eval()
        max_len = int(max_len or self.max_len)
        beam_size = int(max(1, beam_size))
        if beam_size == 1:
            return self.generate_greedy(images, max_len=max_len)

        memory_batch = self.encode_image(images)
        outputs = []
        for i in range(images.size(0)):
            memory = memory_batch[i : i + 1]
            beams = [(torch.tensor([self.bos_idx], dtype=torch.long, device=images.device), 0.0, False)]
            for _ in range(max_len - 1):
                candidates = []
                active = [b for b in beams if not b[2]]
                completed = [b for b in beams if b[2]]
                if not active:
                    break
                seqs = torch.nn.utils.rnn.pad_sequence([b[0] for b in active], batch_first=True, padding_value=self.pad_idx)
                mem = memory.expand(seqs.size(0), -1, -1).contiguous()
                logits = self.decoder(seqs, mem)[:, -1, :]
                logits[:, self.pad_idx] = -1e9
                logits[:, self.bos_idx] = -1e9
                log_probs = F.log_softmax(logits, dim=-1)
                top_scores, top_ids = log_probs.topk(beam_size, dim=-1)
                for row, (seq, score, _) in enumerate(active):
                    for k in range(beam_size):
                        token = top_ids[row, k]
                        new_seq = torch.cat([seq, token.view(1)], dim=0)
                        new_score = score + float(top_scores[row, k].item())
                        done = bool(token.item() == self.eos_idx)
                        candidates.append((new_seq, new_score, done))
                candidates.extend(completed)

                def normalized_score(item):
                    seq, score, _ = item
                    lp = ((5.0 + len(seq)) / 6.0) ** length_penalty
                    return score / lp

                beams = sorted(candidates, key=normalized_score, reverse=True)[:beam_size]
                if all(done for _, _, done in beams):
                    break

            best = max(beams, key=lambda item: item[1] / (((5.0 + len(item[0])) / 6.0) ** length_penalty))[0]
            outputs.append(best)

        return torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True, padding_value=self.pad_idx)


def build_caption_model(config: Dict, vocab_size: int, pad_idx: int, bos_idx: int, eos_idx: int) -> EConvNeXtCaptioner:
    return EConvNeXtCaptioner(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        encoder_variant=config.get("encoder_variant", "mini"),
        d_model=int(config.get("d_model", 256)),
        visual_pool_size=int(config.get("visual_pool_size", 7)),
        visual_queries=int(config.get("visual_queries", 32)),
        resampler_layers=int(config.get("resampler_layers", 2)),
        decoder_layers=int(config.get("decoder_layers", 3)),
        nhead=int(config.get("nhead", 4)),
        ffn_dim=int(config.get("ffn_dim", 768)),
        dropout=float(config.get("dropout", 0.15)),
        max_len=int(config.get("max_len", 40)),
    )
