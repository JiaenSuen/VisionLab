import torch 
import os

device = torch.device('cuda')
def check_accuracy(loader, model): 
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    flops = 0


    try:
        x, _ = next(iter(loader))  # Get one batch
        x = x.to(device=device)
        with torch.profiler.profile(with_flops=True) as prof:
            with torch.no_grad():
                model(x)
        flops = sum(elem.flops for elem in prof.key_averages())
        gflops = flops / 1e9
        print(f"Estimated computational complexity: {gflops:.2f} GFLOPs (batch size {x.size(0)})")
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")


    with torch.no_grad(): 
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
 
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


def save_model(model, path):
    dir_name = os.path.dirname(path)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)

    torch.save(model.state_dict(), path)
    print(f"-Model Save To : {path}")

def load_model(model, path, device='cuda'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"-Can't find model in: {path}")

    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"-Model loaded from: {path}")
    return model