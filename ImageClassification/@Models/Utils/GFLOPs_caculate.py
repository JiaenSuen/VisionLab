import torch

def calculate_gflops(model, input_size=(1, 3, 32, 32)):
    flops = 0
    def hook(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            cin = module.in_channels
            cout = module.out_channels
            k_h, k_w = module.kernel_size
            _, _, h, w = output.shape
            flops += 2 * cin * cout * k_h * k_w * h * w
    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    x = torch.randn(input_size)
    model(x)
    for h in hooks: h.remove()
    return flops / 1e9