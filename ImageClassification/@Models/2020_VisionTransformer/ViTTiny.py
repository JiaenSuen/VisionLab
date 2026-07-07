import time

import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 192,
    ):
        super().__init__()

        if image_size % patch_size != 0:
            raise ValueError(
                "Image size must be divisible by patch size."
            )

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # Input shape:
        # (batch_size, in_channels, image_height, image_width)

        x = self.projection(x)

        # Shape:
        # (batch_size, embed_dim, grid_height, grid_width)

        x = x.flatten(start_dim=2)

        # Shape:
        # (batch_size, embed_dim, num_patches)

        x = x.transpose(1, 2)

        # Shape:
        # (batch_size, num_patches, embed_dim)

        return x


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention layer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
    ):
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_projection = nn.Linear(
            embed_dim,
            embed_dim * 3,
        )

        self.attention_dropout = nn.Dropout(
            attention_dropout
        )

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
        )

        self.output_dropout = nn.Dropout(
            projection_dropout
        )

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape

        qkv = self.qkv_projection(x)

        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )

        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0)

        attention_scores = (
            query @ key.transpose(-2, -1)
        ) * self.scale

        attention_weights = attention_scores.softmax(
            dim=-1
        )

        attention_weights = self.attention_dropout(
            attention_weights
        )

        output = attention_weights @ value

        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            embed_dim,
        )

        output = self.output_projection(output)
        output = self.output_dropout(output)

        return output


class MLPBlock(nn.Module):
    """Feed-forward network used in a Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerEncoderBlock(nn.Module):
    """Pre-normalization Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


class TransformerEncoder(nn.Module):
    """Stack multiple Transformer encoder blocks."""

    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.norm(x)


class ViTTiny(nn.Module):
    """A small Vision Transformer for CIFAR-10 and CIFAR-100."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        embedding_dropout: float = 0.1,
    ):
        super().__init__()

        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                embed_dim,
            )
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                num_patches + 1,
                embed_dim,
            )
        )

        self.embedding_dropout = nn.Dropout(
            embedding_dropout
        )

        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        self.classifier = nn.Linear(
            embed_dim,
            num_classes,
        )

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model parameters."""

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

        self.apply(
            self.initialize_module
        )

    @staticmethod
    def initialize_module(module):
        """Initialize an individual module."""

        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(
                module.weight,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(
                module.weight,
                mode="fan_out",
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(
                module.weight
            )

            nn.init.zeros_(
                module.bias
            )

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(
                "Input tensor must have four dimensions."
            )

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, "
                f"but received {x.shape[1]}."
            )

        if x.shape[2:] != (
            self.image_size,
            self.image_size,
        ):
            raise ValueError(
                f"Expected image size "
                f"{self.image_size} x {self.image_size}, "
                f"but received {x.shape[2]} x {x.shape[3]}."
            )

        batch_size = x.shape[0]

        x = self.patch_embedding(x)

        class_token = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            (class_token, x),
            dim=1,
        )

        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.encoder(x)

        class_token_output = x[:, 0]

        logits = self.classifier(
            class_token_output
        )

        return logits


if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    batch_size = 1
    image_size = 32

    # Use 10 for CIFAR-10 or 100 for CIFAR-100.
    num_classes = 10

    model = ViTTiny(
        image_size=image_size,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=6,
        num_heads=3,
        mlp_dim=768,
        dropout=0.1,
        attention_dropout=0.0,
        embedding_dropout=0.1,
    ).to(device)

    random_image = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    model.eval()

    warmup_iterations = 10
    test_iterations = 50

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(random_image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        for _ in range(test_iterations):
            output = model(random_image)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

    average_inference_time_ms = (
        end_time - start_time
    ) / test_iterations * 1000

    expected_output_shape = (
        batch_size,
        num_classes,
    )

    if output.shape != expected_output_shape:
        raise RuntimeError(
            f"Expected output shape {expected_output_shape}, "
            f"but received {tuple(output.shape)}."
        )

    print("=" * 50)
    print("ViT-Tiny Test")
    print("=" * 50)
    print(f"Device:                 {device}")
    print(f"Input shape:            {tuple(random_image.shape)}")
    print(f"Output shape:           {tuple(output.shape)}")
    print(f"Total parameters:       {total_parameters:,}")
    print(
        f"Average inference time: "
        f"{average_inference_time_ms:.3f} ms"
    )
    print("=" * 50)
    print("Model execution test passed.")