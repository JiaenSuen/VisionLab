from torch import nn


class FastSCNN(nn.Module):
    """Reserved interface for the second project version."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        raise NotImplementedError(
            "Fast-SCNN is intentionally reserved for version 2. "
            "Use configs/lraspp.yaml in this version."
        )

