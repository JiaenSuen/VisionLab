import torch
from yolo import build_yolov10_econvnext, count_parameters
from loss import YOLOv10EConvNeXtLoss


def main():
    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_yolov10_econvnext(num_classes=20, variant="mini").to(device)
    model.train()
    x = torch.randn(2, 3, 128, 128, device=device)
    targets = [
        torch.tensor([[14, 0.50, 0.50, 0.25, 0.40], [6, 0.25, 0.30, 0.20, 0.15]], dtype=torch.float32, device=device),
        torch.tensor([[7, 0.62, 0.55, 0.18, 0.18]], dtype=torch.float32, device=device),
    ]
    out = model(x, branch="both")
    print("params total M", count_parameters(model, trainable_only=False))
    print("one2many shapes", [tuple(t.shape) for t in out["one2many"]])
    print("one2one shapes", [tuple(t.shape) for t in out["one2one"]])
    loss_fn = YOLOv10EConvNeXtLoss(num_classes=20)
    loss, stats = loss_fn(out, targets)
    print("loss", float(loss.detach().cpu()))
    print(stats)
    loss.backward()
    print("backward ok")


if __name__ == "__main__":
    main()
