"""
loss.py

Dual-branch YOLOv10-style anchor-free, objectness-free loss for E-ConvNeXt VOC.

It keeps the stable parts of the previous experiment:
  * raw VOC boxes: [class, x, y, w, h] normalized
  * no anchors, no objectness target
  * quality-aware classification target from detached IoU

Additions for YOLOv10-style training:
  * one-to-many branch: dense center-prior assignment for recall and stable learning
  * one-to-one branch: one nearest positive per GT on its best level, used for inference alignment
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorList = List[torch.Tensor]


def xywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(dim=-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def bbox_iou_xywh(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = xywh_to_xyxy(pred)
    t = xywh_to_xyxy(target)
    px1, py1, px2, py2 = p.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = t.unbind(dim=-1)
    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    ta = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    return inter / (pa + ta - inter + eps)


def bbox_ciou_xywh(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    p = xywh_to_xyxy(pred)
    t = xywh_to_xyxy(target)
    px1, py1, px2, py2 = p.unbind(dim=-1)
    tx1, ty1, tx2, ty2 = t.unbind(dim=-1)

    ix1 = torch.maximum(px1, tx1)
    iy1 = torch.maximum(py1, ty1)
    ix2 = torch.minimum(px2, tx2)
    iy2 = torch.minimum(py2, ty2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    pa = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    ta = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    iou = inter / (pa + ta - inter + eps)

    px, py, pw, ph = pred.unbind(dim=-1)
    tx, ty, tw, th = target.unbind(dim=-1)
    rho2 = (px - tx).pow(2) + (py - ty).pow(2)
    cx1 = torch.minimum(px1, tx1)
    cy1 = torch.minimum(py1, ty1)
    cx2 = torch.maximum(px2, tx2)
    cy2 = torch.maximum(py2, ty2)
    c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps
    v = (4.0 / (torch.pi ** 2)) * (torch.atan(tw / (th + eps)) - torch.atan(pw / (ph + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (1.0 - iou + v + eps)
    ciou = iou - rho2 / c2 - alpha * v
    return ciou.clamp(min=-1.0, max=1.0)


def quality_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    neg_weight: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float()
    targets = targets.float()
    prob = torch.sigmoid(logits).clamp(min=eps, max=1.0 - eps)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos = targets > 0
    weight = torch.where(pos, targets.clamp(min=eps), neg_weight * prob.pow(gamma))
    return (bce * weight).sum()


class YOLOv10EConvNeXtLoss(nn.Module):
    """Loss for outputs from YOLOv10EConvNeXt.

    Args:
        center_radius: per-level center prior radius in grid units for [large, medium, small].
        one2one_gain: multiplier for the one-to-one branch.
    """

    def __init__(
        self,
        num_classes: int = 20,
        box_gain: float = 5.0,
        cls_gain: float = 1.0,
        one2one_gain: float = 0.5,
        center_radius: Sequence[float] = (1.0, 1.35, 1.75),
        wh_logit_clip: float = 4.0,
        gamma: float = 2.0,
        neg_weight: float = 0.75,
        min_pos_iou_quality: float = 0.10,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.box_gain = float(box_gain)
        self.cls_gain = float(cls_gain)
        self.one2one_gain = float(one2one_gain)
        self.center_radius = tuple(float(x) for x in center_radius)
        self.wh_logit_clip = float(wh_logit_clip)
        self.gamma = float(gamma)
        self.neg_weight = float(neg_weight)
        self.min_pos_iou_quality = float(min_pos_iou_quality)

    @staticmethod
    def _best_level(w: float, h: float) -> int:
        """Level index in [large=0, medium=1, small=2]."""
        max_side = max(float(w), float(h))
        if max_side < 0.16:
            return 2
        if max_side < 0.36:
            return 1
        return 0

    def _levels_for_gt(self, w: float, h: float, assign_mode: str) -> List[int]:
        best = self._best_level(w, h)
        if assign_mode == "one2one":
            return [best]
        if best == 2:
            return [2, 1]
        if best == 1:
            return [1, 2, 0]
        return [0, 1]

    def _decode_grid_boxes(self, pred: torch.Tensor) -> torch.Tensor:
        _, _, h, w = pred.shape
        device = pred.device
        dtype = pred.dtype
        box_logits = pred[:, 0:4].float()
        yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
        xx = xx.view(1, 1, h, w).float()
        yy = yy.view(1, 1, h, w).float()
        xy = torch.sigmoid(box_logits[:, 0:2])
        cx = xy[:, 0:1] + xx
        cy = xy[:, 1:2] + yy
        bw = torch.exp(box_logits[:, 2:3].clamp(min=-self.wh_logit_clip, max=self.wh_logit_clip))
        bh = torch.exp(box_logits[:, 3:4].clamp(min=-self.wh_logit_clip, max=self.wh_logit_clip))
        return torch.cat([cx, cy, bw, bh], dim=1).to(dtype=dtype)

    def _build_targets_for_level(
        self,
        outputs: TensorList,
        targets: List[torch.Tensor],
        level_idx: int,
        assign_mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred = outputs[level_idx]
        bsz, _, gh, gw = pred.shape
        device = pred.device
        dtype = pred.dtype

        pos_mask = torch.zeros((bsz, gh, gw), dtype=torch.bool, device=device)
        cls_target = torch.zeros((bsz, self.num_classes, gh, gw), dtype=dtype, device=device)
        box_target = torch.zeros((bsz, 4, gh, gw), dtype=dtype, device=device)
        area_map = torch.full((bsz, gh, gw), float("inf"), dtype=dtype, device=device)

        radius = self.center_radius[level_idx]
        r_int = int(max(1, round(radius)))

        for b, boxes in enumerate(targets):
            if boxes is None or boxes.numel() == 0:
                continue
            boxes = boxes.to(device=device, dtype=dtype)
            for gt in boxes:
                cls = int(gt[0].item())
                if cls < 0 or cls >= self.num_classes:
                    continue
                x = float(gt[1].item())
                y = float(gt[2].item())
                bw = float(gt[3].item())
                bh = float(gt[4].item())
                if bw <= 0 or bh <= 0:
                    continue
                if level_idx not in self._levels_for_gt(bw, bh, assign_mode=assign_mode):
                    continue

                gx = x * gw
                gy = y * gh
                cx = int(min(max(gx, 0), gw - 1))
                cy = int(min(max(gy, 0), gh - 1))
                area = max(bw * bh, 1e-8)

                if assign_mode == "one2one":
                    candidate_cells = [(cy, cx)]
                else:
                    x0 = max(0, cx - r_int)
                    x1 = min(gw - 1, cx + r_int)
                    y0 = max(0, cy - r_int)
                    y1 = min(gh - 1, cy + r_int)
                    candidate_cells = []
                    for iy in range(y0, y1 + 1):
                        for ix in range(x0, x1 + 1):
                            dist = ((ix + 0.5 - gx) ** 2 + (iy + 0.5 - gy) ** 2) ** 0.5
                            if dist <= radius:
                                candidate_cells.append((iy, ix))

                for iy, ix in candidate_cells:
                    if area >= float(area_map[b, iy, ix].item()):
                        continue
                    area_map[b, iy, ix] = area
                    pos_mask[b, iy, ix] = True
                    cls_target[b, :, iy, ix] = 0
                    cls_target[b, cls, iy, ix] = 1.0  # replaced by IoU quality after decoding
                    box_target[b, :, iy, ix] = torch.tensor([gx, gy, bw * gw, bh * gh], device=device, dtype=dtype)

        return pos_mask, cls_target, box_target

    def _loss_single_branch(
        self,
        outputs: TensorList,
        targets: List[torch.Tensor],
        assign_mode: str,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_box = outputs[0].sum() * 0.0
        total_cls = outputs[0].sum() * 0.0
        total_pos = 0

        for level_idx, pred in enumerate(outputs):
            bsz, ch, gh, gw = pred.shape
            expected = 4 + self.num_classes
            if ch != expected:
                raise ValueError(f"Expected {expected} output channels, got {ch}")
            pos_mask, cls_target, box_target = self._build_targets_for_level(outputs, targets, level_idx, assign_mode)
            cls_logits = pred[:, 4:].float()
            pred_boxes = self._decode_grid_boxes(pred).float()
            num_pos = int(pos_mask.sum().item())

            if num_pos > 0:
                pred_pos = pred_boxes.permute(0, 2, 3, 1)[pos_mask]
                target_pos = box_target.permute(0, 2, 3, 1)[pos_mask].float()
                ciou = bbox_ciou_xywh(pred_pos, target_pos)
                iou_q = ciou.detach().clamp(min=self.min_pos_iou_quality, max=1.0)
                total_box = total_box + (1.0 - ciou).sum()

                # Insert IoU quality into the positive class target.
                pos_indices = pos_mask.nonzero(as_tuple=False)
                for k, (b, iy, ix) in enumerate(pos_indices):
                    positive_classes = torch.where(cls_target[b, :, iy, ix] > 0)[0]
                    if positive_classes.numel() > 0:
                        cls_target[b, positive_classes[0], iy, ix] = iou_q[k].to(dtype=cls_target.dtype)

            total_cls = total_cls + quality_focal_bce(
                cls_logits,
                cls_target,
                gamma=self.gamma,
                neg_weight=self.neg_weight,
            )
            total_pos += max(num_pos, 1)

        normalizer = max(total_pos, 1)
        loss_box = total_box / normalizer
        loss_cls = total_cls / normalizer
        loss_total = self.box_gain * loss_box + self.cls_gain * loss_cls
        return loss_total, {
            f"{assign_mode}_box": float(loss_box.detach().cpu()),
            f"{assign_mode}_cls": float(loss_cls.detach().cpu()),
            f"{assign_mode}_pos": float(total_pos),
        }

    def forward(
        self,
        outputs: Union[Dict[str, TensorList], TensorList],
        targets: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if isinstance(outputs, dict):
            many_loss, many_stats = self._loss_single_branch(outputs["one2many"], targets, assign_mode="one2many")
            one_loss, one_stats = self._loss_single_branch(outputs["one2one"], targets, assign_mode="one2one")
            total = many_loss + self.one2one_gain * one_loss
            stats = {**many_stats, **one_stats}
            stats["loss_many"] = float(many_loss.detach().cpu())
            stats["loss_one"] = float(one_loss.detach().cpu())
            stats["loss"] = float(total.detach().cpu())
            stats["box"] = many_stats.get("one2many_box", 0.0) + self.one2one_gain * one_stats.get("one2one_box", 0.0)
            stats["cls"] = many_stats.get("one2many_cls", 0.0) + self.one2one_gain * one_stats.get("one2one_cls", 0.0)
            stats["pos"] = many_stats.get("one2many_pos", 0.0) + one_stats.get("one2one_pos", 0.0)
            return total, stats

        # Backward-compatible single-branch mode.
        total, stats = self._loss_single_branch(outputs, targets, assign_mode="one2many")
        stats["loss"] = float(total.detach().cpu())
        stats["box"] = stats.get("one2many_box", 0.0)
        stats["cls"] = stats.get("one2many_cls", 0.0)
        stats["pos"] = stats.get("one2many_pos", 0.0)
        return total, stats
