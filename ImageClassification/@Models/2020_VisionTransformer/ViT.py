import torch
from torch import nn



# Linear Projection
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.num_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x
    



# MultiHeadSelfAttention
class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.output_projection = nn.Linear(embed_dim, embed_dim)

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, sequence_length, embed_dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)

        query, key, value = qkv.unbind(dim=0) # Split the dimension and remove the dimension.

        attention_scores = query @ key.transpose(-2, -1) # Matrix multiplication operators  ==> torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores * self.scale

        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        output = attention_weights @ value
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            embed_dim,
        )

        output = self.output_projection(output)
        output = self.output_dropout(output)

        return output
    


# Responsible for the feed-forward network in the Transformer block
class MLPBlock(nn.Module):
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
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    






class TransformerEncoder(nn.Module):
    def __init__(
        self,
        depth: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.norm(x)






class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embedding.num_patches

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.encoder = TransformerEncoder(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.patch_embedding(x)

        class_token = self.class_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat([class_token, x], dim=1)
        x = x + self.position_embedding
        x = self.embedding_dropout(x)

        x = self.encoder(x)

        class_token_output = x[:, 0]
        logits = self.classifier(class_token_output)

        return logits
    













import time
if __name__ == "__main__":
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    batch_size = 1
    image_size = 224
    in_channels = 3
    num_classes = 1000

    model = VisionTransformer(
        image_size=image_size,
        patch_size=16,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.0,
        embedding_dropout=0.0,
    ).to(device)

    random_image = torch.randn(
        batch_size,
        in_channels,
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
        # Warm up the model.
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

    average_time_ms = (
        end_time - start_time
    ) / test_iterations * 1000

    print("=" * 50)
    print("Vision Transformer Test")
    print("=" * 50)
    print(f"Device:                 {device}")
    print(f"Input shape:            {tuple(random_image.shape)}")
    print(f"Output shape:           {tuple(output.shape)}")
    print(f"Total parameters:       {total_parameters:,}")
    print(f"Average inference time: {average_time_ms:.3f} ms")
    print("=" * 50)