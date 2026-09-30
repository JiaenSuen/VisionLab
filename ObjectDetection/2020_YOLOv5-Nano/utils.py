import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
from tqdm import tqdm

try:
    from torchvision.ops import batched_nms
except Exception:
    batched_nms = None


VOC_CLASSES = {
    0: "aeroplane", 1: "bicycle", 2: "bird", 3: "boat", 4: "bottle",
    5: "bus", 6: "car", 7: "cat", 8: "chair", 9: "cow",
    10: "diningtable", 11: "dog", 12: "horse", 13: "motorbike", 14: "person",
    15: "pottedplant", 16: "sheep", 17: "sofa", 18: "train", 19: "tvmonitor",
}


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculate Intersection over Union.

    Args:
        boxes_preds: Tensor of boxes.
        boxes_labels: Tensor of boxes.
        box_format: "midpoint" for [x, y, w, h], or "corners" for [x1, y1, x2, y2].

    Returns:
        Tensor containing IoU values.
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]

        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    else:
        raise ValueError("box_format must be either 'midpoint' or 'corners'.")

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def midpoint_to_corners(boxes):
    """
    Convert boxes from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        boxes: Tensor with shape (..., 4).

    Returns:
        Tensor with shape (..., 4).
    """

    x, y, w, h = boxes.unbind(dim=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return torch.stack((x1, y1, x2, y2), dim=-1)


def non_max_suppression(
    bboxes,
    iou_threshold,
    threshold,
    box_format="midpoint",
    max_detections=300,
):
    """
    Apply class-wise Non-Maximum Suppression.

    Args:
        bboxes: List of boxes in format [class, confidence, x, y, w, h].
        iou_threshold: IoU threshold used by NMS.
        threshold: Confidence threshold.
        box_format: Box format of the input boxes.
        max_detections: Maximum number of boxes to keep after NMS.

    Returns:
        List of boxes after NMS.
    """

    if len(bboxes) == 0:
        return []

    boxes = torch.tensor(bboxes, dtype=torch.float32)
    boxes = boxes[boxes[:, 1] > threshold]

    if boxes.numel() == 0:
        return []

    labels = boxes[:, 0]
    scores = boxes[:, 1]

    if box_format == "midpoint":
        boxes_xyxy = midpoint_to_corners(boxes[:, 2:6])
    elif box_format == "corners":
        boxes_xyxy = boxes[:, 2:6]
    else:
        raise ValueError("box_format must be either 'midpoint' or 'corners'.")

    # Clamp boxes to the normalized image range.
    boxes_xyxy = boxes_xyxy.clamp(0, 1)

    if batched_nms is not None:
        keep = batched_nms(boxes_xyxy, scores, labels, iou_threshold)
    else:
        # Fallback Python NMS if torchvision.ops is not available.
        keep = []
        order = scores.argsort(descending=True)
        while order.numel() > 0:
            current = order[0]
            keep.append(current)
            if order.numel() == 1:
                break
            rest = order[1:]
            same_class = labels[rest] == labels[current]
            ious = intersection_over_union(
                boxes[current, 2:6].unsqueeze(0),
                boxes[rest, 2:6],
                box_format=box_format,
            ).squeeze(-1)
            order = rest[(~same_class) | (ious < iou_threshold)]
        keep = torch.stack(keep) if len(keep) else torch.empty(0, dtype=torch.long)

    if max_detections is not None:
        keep = keep[:max_detections]

    return boxes[keep].cpu().tolist()


def mean_average_precision(
    pred_boxes,
    true_boxes,
    iou_threshold=0.5,
    box_format="midpoint",
    num_classes=20,
):
    """
    Calculate mean Average Precision.

    Args:
        pred_boxes: List of predicted boxes.
                    Format: [image_idx, class, confidence, x, y, w, h].
        true_boxes: List of ground-truth boxes.
                    Format: [image_idx, class, confidence, x, y, w, h].
        iou_threshold: IoU threshold for a true positive.
        box_format: Box format used for IoU calculation.
        num_classes: Number of classes.

    Returns:
        mAP value.
    """

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [d for d in pred_boxes if int(d[1]) == c]
        ground_truths = [gt for gt in true_boxes if int(gt[1]) == c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]

            best_iou = 0.0
            best_gt_idx = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:], dtype=torch.float32),
                    torch.tensor(gt[3:], dtype=torch.float32),
                    box_format=box_format,
                )

                if iou.item() > best_iou:
                    best_iou = iou.item()
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)

        precisions = torch.cat((torch.tensor([1.0]), precisions))
        recalls = torch.cat((torch.tensor([0.0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls))

    if len(average_precisions) == 0:
        return 0.0

    return float(sum(average_precisions) / len(average_precisions))


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Convert multi-scale YOLO predictions or targets to bounding boxes.

    Args:
        predictions:
            If is_preds=True:
                Tensor with shape (batch_size, 3, S, S, 5 + num_classes).
                Format: [objectness_logit, x_logit, y_logit, w_logit, h_logit, class_logits...]

            If is_preds=False:
                Tensor with shape (batch_size, 3, S, S, 6).
                Format: [objectness, x_cell, y_cell, width_cell, height_cell, class_label]

        anchors:
            Tensor with shape (3, 2). Anchors must be scaled to the current grid size.

        S:
            Grid size.

        is_preds:
            Whether the input is model prediction or ground-truth target.

    Returns:
        Tensor with shape (batch_size, 3*S*S, 6).
        Box format: [class, confidence, x, y, w, h].
    """

    batch_size = predictions.shape[0]
    num_anchors = anchors.shape[0]
    device = predictions.device

    anchors = anchors.reshape(1, num_anchors, 1, 1, 2).to(device)

    if is_preds:
        objectness = torch.sigmoid(predictions[..., 0:1])
        xy = torch.sigmoid(predictions[..., 1:3])
        wh = torch.exp(predictions[..., 3:5]) * anchors

        # CrossEntropyLoss is used during training, so softmax is used for class probabilities.
        class_probs = torch.softmax(predictions[..., 5:], dim=-1)
        class_conf, best_class = torch.max(class_probs, dim=-1, keepdim=True)

        scores = objectness * class_conf
        best_class = best_class.float()
        box_predictions = torch.cat((xy, wh), dim=-1)

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
        box_predictions = predictions[..., 1:5]

    cell_indices = (
        torch.arange(S, device=device)
        .repeat(batch_size, num_anchors, S, 1)
        .unsqueeze(-1)
    )

    x = (box_predictions[..., 0:1] + cell_indices) / S
    y = (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4)) / S
    w_h = box_predictions[..., 2:4] / S

    converted_bboxes = torch.cat(
        (best_class, scores, x, y, w_h),
        dim=-1,
    ).reshape(batch_size, num_anchors * S * S, 6)

    return converted_bboxes


def get_bboxes(
    loader,
    model,
    iou_threshold,
    threshold,
    anchors,
    box_format="midpoint",
    device="cuda",
    desc="Evaluating bboxes",
    max_eval_batches=None,
    max_detections=300,
):
    """
    Get predicted and ground-truth boxes from a three-scale YOLO detector.

    Args:
        loader: DataLoader.
        model: Detection model.
        iou_threshold: IoU threshold for NMS.
        threshold: Confidence threshold.
        anchors: Anchors normalized by image size.
        box_format: Box format for NMS and mAP.
        device: Device used for inference.
        desc: tqdm description.
        max_eval_batches: Optional number of batches to evaluate.
        max_detections: Maximum detections per image after NMS.

    Returns:
        all_pred_boxes: [image_idx, class, confidence, x, y, w, h].
        all_true_boxes: [image_idx, class, confidence, x, y, w, h].
    """

    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    loop = tqdm(loader, leave=True, desc=desc)

    for batch_idx, (x, labels) in enumerate(loop):
        if max_eval_batches is not None and batch_idx >= max_eval_batches:
            break

        x = x.to(device, non_blocking=True)

        if isinstance(labels, (list, tuple)):
            labels = [label.to(device, non_blocking=True) for label in labels]
        else:
            raise TypeError("Labels should be a list or tuple with 3 tensors.")

        with torch.inference_mode():
            predictions = model(x)

        if not isinstance(predictions, (list, tuple)):
            raise TypeError("Model output should be a list or tuple with 3 tensors.")

        batch_size = x.shape[0]

        pred_bboxes = [[] for _ in range(batch_size)]
        true_bboxes = [[] for _ in range(batch_size)]

        for scale_idx in range(len(predictions)):
            S = predictions[scale_idx].shape[2]
            scaled_anchor = torch.tensor(anchors[scale_idx], device=device) * S

            boxes_scale = cells_to_bboxes(
                predictions=predictions[scale_idx],
                anchors=scaled_anchor,
                S=S,
                is_preds=True,
            )

            true_boxes_scale = cells_to_bboxes(
                predictions=labels[scale_idx],
                anchors=scaled_anchor,
                S=S,
                is_preds=False,
            )

            for idx in range(batch_size):
                pred_bboxes[idx].append(boxes_scale[idx])
                true_bboxes[idx].append(true_boxes_scale[idx])

        for idx in range(batch_size):
            image_pred_boxes = torch.cat(pred_bboxes[idx], dim=0)

            # Filter by confidence before NMS.
            image_pred_boxes = image_pred_boxes[image_pred_boxes[:, 1] > threshold]

            if image_pred_boxes.numel() > 0:
                nms_boxes = non_max_suppression(
                    bboxes=image_pred_boxes.detach().cpu().tolist(),
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                    max_detections=max_detections,
                )
            else:
                nms_boxes = []

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            image_true_boxes = torch.cat(true_bboxes[idx], dim=0)
            image_true_boxes = image_true_boxes[image_true_boxes[:, 1] > 0.5]

            for box in image_true_boxes.detach().cpu().tolist():
                all_true_boxes.append([train_idx] + box)

            train_idx += 1

        loop.set_postfix(
            pred_boxes=len(all_pred_boxes),
            true_boxes=len(all_true_boxes),
        )

    model.train()

    return all_pred_boxes, all_true_boxes


def plot_image(image, boxes):
    """
    Plot bounding boxes on an image.

    Args:
        image: Image tensor or numpy array with shape (H, W, C).
        boxes: List of boxes in format [class, confidence, x, y, w, h].
    """

    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        class_id = int(box[0])
        confidence = float(box[1])

        x_mid = float(box[2])
        y_mid = float(box[3])
        box_width = float(box[4])
        box_height = float(box[5])

        upper_left_x = x_mid - box_width / 2
        upper_left_y = y_mid - box_height / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box_width * width,
            box_height * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )

        ax.add_patch(rect)

        class_name = VOC_CLASSES.get(class_id, "Unknown")

        ax.text(
            upper_left_x * width,
            upper_left_y * height,
            f"{class_name}: {confidence:.2f}",
            color="red",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.6),
        )

    plt.axis("off")
    plt.show()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Save a model checkpoint."""

    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Load a model checkpoint."""

    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
