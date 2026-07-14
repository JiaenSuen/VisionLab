from torch import nn


class SegmentationCrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, ignore_index: int = 255) -> None:
        super().__init__(ignore_index=ignore_index)

