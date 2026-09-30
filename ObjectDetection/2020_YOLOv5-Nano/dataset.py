import os
import pandas as pd
import torch
from PIL import Image


def iou_width_height(boxes1, boxes2):
    """
    Calculate IoU for width-height pairs only.

    Args:
        boxes1: Tensor with shape (..., 2), where the last dimension is [width, height].
        boxes2: Tensor with shape (..., 2), where the last dimension is [width, height].

    Returns:
        Tensor containing IoU values for width-height pairs.
    """

    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1]
        + boxes2[..., 0] * boxes2[..., 1]
        - intersection
    )
    return intersection / (union + 1e-6)


class VOCDataset(torch.utils.data.Dataset):
    """
    Pascal VOC-style dataset for the three-scale YOLO detector.

    The label file must contain one bounding box per line:
        class_label x_center y_center width height

    All box coordinates must be normalized between 0 and 1.

    Returns:
        image: Tensor image.
        targets: Tuple of 3 tensors, one tensor for each detection scale.

    Each target tensor shape:
        (3, S, S, 6)

    Target format:
        [objectness, x_cell, y_cell, width_cell, height_cell, class_label]
    """

    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        anchors,
        image_size=448,
        S=(14, 28, 56),
        C=20,
        transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_size = image_size
        self.S = S
        self.C = C

        # Flatten anchors from shape (3, 3, 2) to shape (9, 2).
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2], dtype=torch.float32)

        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image_filename = self.annotations.iloc[index, 0]
        label_filename = self.annotations.iloc[index, 1]

        img_path = os.path.join(self.img_dir, image_filename)
        label_path = os.path.join(self.label_dir, label_filename)

        image = Image.open(img_path).convert("RGB")
        bboxes = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue

                    values = line.split()
                    if len(values) != 5:
                        raise ValueError(
                            f"Invalid label format in {label_path}: '{line}'. "
                            "Expected: class_label x_center y_center width height"
                        )

                    class_label = int(float(values[0]))
                    x = float(values[1])
                    y = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                    bboxes.append([class_label, x, y, width, height])

        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        targets = [
            torch.zeros((self.num_anchors_per_scale, S, S, 6), dtype=torch.float32)
            for S in self.S
        ]

        for box in bboxes:
            class_label, x, y, width, height = box

            x = min(max(float(x), 0.0), 0.999999)
            y = min(max(float(y), 0.0), 0.999999)
            width = min(max(float(width), 1e-6), 1.0)
            height = min(max(float(height), 1e-6), 1.0)

            iou_anchors = iou_width_height(
                torch.tensor([width, height], dtype=torch.float32),
                self.anchors,
            )

            anchor_indices = iou_anchors.argsort(descending=True)
            has_anchor = [False, False, False]

            for anchor_idx_tensor in anchor_indices:
                anchor_idx = int(anchor_idx_tensor.item())

                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale

                S = self.S[scale_idx]
                i = int(S * y)
                j = int(S * x)

                i = min(max(i, 0), S - 1)
                j = min(max(j, 0), S - 1)

                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if anchor_taken == 0 and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1

                    x_cell = S * x - j
                    y_cell = S * y - i
                    width_cell = width * S
                    height_cell = height * S

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell],
                        dtype=torch.float32,
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 5] = class_label

                    has_anchor[scale_idx] = True

                elif anchor_taken == 0 and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    # Ignore this prediction in the no-object loss.
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return image, tuple(targets)
