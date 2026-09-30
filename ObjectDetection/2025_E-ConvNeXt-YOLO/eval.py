"""
eval.py

Fast VOC-style evaluation for E-ConvNeXt + YOLOv10 detector.
Prediction list format:
  [image_idx, class, score, x, y, w, h]
All boxes are normalized midpoint coordinates.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm import tqdm

try:
    from torchvision.ops import batched_nms
except Exception:  # pragma: no cover
    batched_nms = None


VOC_CLASSES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
    10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
    15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


def midpoint_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(dim=-1)
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def intersection_over_union(boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, box_format: str = "midpoint") -> torch.Tensor:
    if box_format == "midpoint":
        b1 = midpoint_to_corners(boxes_preds)
        b2 = midpoint_to_corners(boxes_labels)
    else:
        b1, b2 = boxes_preds, boxes_labels
    b1x1, b1y1, b1x2, b1y2 = b1[..., 0:1], b1[..., 1:2], b1[..., 2:3], b1[..., 3:4]
    b2x1, b2y1, b2x2, b2y2 = b2[..., 0:1], b2[..., 1:2], b2[..., 2:3], b2[..., 3:4]
    x1 = torch.maximum(b1x1, b2x1)
    y1 = torch.maximum(b1y1, b2y1)
    x2 = torch.minimum(b1x2, b2x2)
    y2 = torch.minimum(b1y2, b2y2)
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    a1 = (b1x2 - b1x1).abs() * (b1y2 - b1y1).abs()
    a2 = (b2x2 - b2x1).abs() * (b2y2 - b2y1).abs()
    return inter / (a1 + a2 - inter + 1e-6)


def tensor_nms(boxes_cls_score_xywh: torch.Tensor, iou_threshold: float = 0.45, max_detections: int = 300) -> torch.Tensor:
    if boxes_cls_score_xywh.numel() == 0:
        return boxes_cls_score_xywh.new_zeros((0, 6))
    labels = boxes_cls_score_xywh[:, 0]
    scores = boxes_cls_score_xywh[:, 1]
    boxes_xyxy = midpoint_to_corners(boxes_cls_score_xywh[:, 2:6]).clamp(0, 1)
    if batched_nms is not None and iou_threshold < 0.999:
        keep = batched_nms(boxes_xyxy, scores, labels, iou_threshold)
    else:
        keep_list = []
        order = scores.argsort(descending=True)
        while order.numel() > 0:
            cur = order[0]
            keep_list.append(cur)
            if order.numel() == 1:
                break
            rest = order[1:]
            if iou_threshold >= 0.999:
                order = rest
                continue
            same = labels[rest] == labels[cur]
            ious = intersection_over_union(
                boxes_cls_score_xywh[cur, 2:6].unsqueeze(0),
                boxes_cls_score_xywh[rest, 2:6],
            ).squeeze(-1)
            order = rest[(~same) | (ious < iou_threshold)]
        keep = torch.stack(keep_list) if keep_list else torch.empty(0, dtype=torch.long, device=boxes_cls_score_xywh.device)
    return boxes_cls_score_xywh[keep[:max_detections]]


def _as_output_list(outputs, branch: str = "one2one") -> List[torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs[branch]
    return outputs


def decode_outputs(outputs, score_threshold: float = 1e-4, topk: Optional[int] = 2000, branch: str = "one2one") -> List[torch.Tensor]:
    """Return per-image list of [class, score, x, y, w, h] tensors."""
    outputs = _as_output_list(outputs, branch=branch)
    device = outputs[0].device
    batch_size = outputs[0].shape[0]
    per_image: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

    for out in outputs:
        bsz, c4, gh, gw = out.shape
        num_classes = c4 - 4
        box_logits = out[:, 0:4].float()
        cls_logits = out[:, 4:].float()
        yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        xx = xx.reshape(1, 1, gh, gw).float()
        yy = yy.reshape(1, 1, gh, gw).float()
        xy = torch.sigmoid(box_logits[:, 0:2])
        x = (xy[:, 0:1] + xx) / gw
        y = (xy[:, 1:2] + yy) / gh
        w = torch.exp(box_logits[:, 2:3].clamp(min=-4.0, max=4.0)) / gw
        h = torch.exp(box_logits[:, 3:4].clamp(min=-4.0, max=4.0)) / gh
        boxes = torch.cat([x, y, w, h], dim=1).permute(0, 2, 3, 1).reshape(bsz, gh * gw, 4)
        scores = torch.sigmoid(cls_logits).permute(0, 2, 3, 1).reshape(bsz, gh * gw, num_classes)

        for b in range(bsz):
            s = scores[b]
            flat = s.reshape(-1)
            if topk is not None and flat.numel() > topk:
                vals, idx = torch.topk(flat, k=topk, largest=True)
            else:
                vals = flat
                idx = torch.arange(flat.numel(), device=device)
            mask = vals > score_threshold
            if not mask.any():
                continue
            vals = vals[mask]
            idx = idx[mask]
            point_idx = torch.div(idx, num_classes, rounding_mode="floor")
            cls_idx = (idx % num_classes).float()
            selected_boxes = boxes[b, point_idx].clamp(min=0.0, max=1.0)
            decoded = torch.cat([cls_idx.unsqueeze(1), vals.unsqueeze(1), selected_boxes], dim=1)
            decoded = decoded[torch.isfinite(decoded).all(dim=1)]
            if decoded.numel() > 0:
                per_image[b].append(decoded)

    merged = []
    for boxes in per_image:
        merged.append(torch.cat(boxes, dim=0) if boxes else torch.zeros((0, 6), device=device))
    return merged


@torch.inference_mode()
def get_bboxes(
    loader,
    model,
    iou_threshold: float = 0.45,
    threshold: float = 1e-4,
    device: str = "cuda",
    desc: str = "Evaluating",
    max_eval_batches: Optional[int] = None,
    max_detections: int = 300,
    use_amp: bool = True,
    pre_nms_topk: int = 2000,
    branch: str = "one2one",
):
    all_pred, all_true = [], []
    model.eval()
    image_idx = 0
    use_amp_now = bool(use_amp and str(device).startswith("cuda"))
    try:
        autocast_ctx = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda", "enabled": use_amp_now}
    except Exception:  # pragma: no cover
        autocast_ctx = torch.cuda.amp.autocast
        autocast_kwargs = {"enabled": use_amp_now}

    for batch_idx, (x, targets) in enumerate(tqdm(loader, leave=True, desc=desc)):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break
        x = x.to(device, non_blocking=True)
        with autocast_ctx(**autocast_kwargs):
            outputs = model(x, branch=branch)
        decoded = decode_outputs(outputs, score_threshold=threshold, topk=pre_nms_topk, branch=branch)
        for b, det in enumerate(decoded):
            if det.numel() > 0:
                keep = tensor_nms(det.float(), iou_threshold=iou_threshold, max_detections=max_detections)
                for box in keep.detach().cpu().tolist():
                    all_pred.append([image_idx] + box)
            gt = targets[b]
            if gt is not None and gt.numel() > 0:
                for row in gt.detach().cpu().tolist():
                    cls, x0, y0, w, h = row
                    all_true.append([image_idx, cls, 1.0, x0, y0, w, h])
            image_idx += 1
    model.train()
    return all_pred, all_true


def _voc_ap_from_pr(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    if recalls.numel() == 0:
        return 0.0
    mrec = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])
    for i in range(mpre.numel() - 1, 0, -1):
        mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_average_precision(
    pred_boxes: List[List[float]],
    true_boxes: List[List[float]],
    iou_threshold: float = 0.5,
    num_classes: int = 20,
    return_per_class: bool = False,
):
    gt_by_class_img: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
    used_by_class_img: Dict[int, Dict[int, torch.Tensor]] = defaultdict(dict)
    det_by_class: Dict[int, List[List[float]]] = defaultdict(list)
    gt_count = defaultdict(int)

    for gt in true_boxes:
        img_id, cls = int(gt[0]), int(gt[1])
        gt_by_class_img[cls].setdefault(img_id, []).append(torch.tensor(gt[3:7], dtype=torch.float32))
        gt_count[cls] += 1
    for cls, by_img in list(gt_by_class_img.items()):
        for img_id, boxes in list(by_img.items()):
            stacked = torch.stack(boxes, dim=0)
            gt_by_class_img[cls][img_id] = stacked
            used_by_class_img[cls][img_id] = torch.zeros(stacked.shape[0], dtype=torch.bool)
    for det in pred_boxes:
        det_by_class[int(det[1])].append(det)

    aps = {}
    eps = 1e-6
    for cls in range(num_classes):
        detections = det_by_class.get(cls, [])
        total_gt = int(gt_count.get(cls, 0))
        if total_gt == 0:
            continue
        detections.sort(key=lambda x: x[2], reverse=True)
        tp = torch.zeros(len(detections))
        fp = torch.zeros(len(detections))
        for i, det in enumerate(detections):
            img_id = int(det[0])
            gts = gt_by_class_img.get(cls, {}).get(img_id, None)
            if gts is None or gts.numel() == 0:
                fp[i] = 1.0
                continue
            det_box = torch.tensor(det[3:7], dtype=torch.float32).unsqueeze(0)
            ious = intersection_over_union(det_box, gts).squeeze(-1)
            best_iou, best_gt_idx = torch.max(ious, dim=0)
            if float(best_iou) > iou_threshold and not used_by_class_img[cls][img_id][best_gt_idx]:
                tp[i] = 1.0
                used_by_class_img[cls][img_id][best_gt_idx] = True
            else:
                fp[i] = 1.0
        tp_cum = torch.cumsum(tp, dim=0)
        fp_cum = torch.cumsum(fp, dim=0)
        recalls = tp_cum / (total_gt + eps)
        precisions = tp_cum / (tp_cum + fp_cum + eps)
        aps[cls] = _voc_ap_from_pr(recalls, precisions)

    mAP = float(sum(aps.values()) / max(len(aps), 1))
    if return_per_class:
        return mAP, aps
    return mAP


def save_checkpoint(state, filename: str) -> None:
    torch.save(state, filename)
