import torch 
import os
import time
device = torch.device('cuda')
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    try:
        x, _ = next(iter(loader))  # Get one batch
        x = x.to(device=device)
        batch_size = x.size(0)


        with torch.profiler.profile(with_flops=True) as prof:
            with torch.no_grad():
                model(x)

        flops = sum(elem.flops for elem in prof.key_averages())
        total_gflops = flops / 1e9
        gflops_per_image = total_gflops / batch_size


        # warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)

        torch.cuda.synchronize()

        runs = 50  # 
        start = time.time()

        with torch.no_grad():
            for _ in range(runs):
                _ = model(x)

        torch.cuda.synchronize()
        end = time.time()

        avg_batch_time = (end - start) / runs
        avg_image_time = avg_batch_time / batch_size

        print(
            f"Estimated computational complexity: "
            f"{gflops_per_image:.4f} GFLOPs per image "
            f"(total batch {batch_size}: {total_gflops:.4f}) | "
            f"Inference time: {avg_image_time*1000:.4f} ms per image"
        )

    except Exception as e:
        print(f"Could not calculate FLOPs or time: {e}")


    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} "
            f"with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
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