import torch
import torch.nn as nn
from model import ModelFactory
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import glob

 
MODEL_NAME = "resnet18"     
STRATEGY = "partial"         # freeze / partial / full
device = "cuda" if torch.cuda.is_available() else "cpu"

 
test_path = "cats_and_dogs_small/test"

 
output_base = "VisualCNN"
os.makedirs(output_base, exist_ok=True)

 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
 
print(f"Loading model: {MODEL_NAME} with strategy: {STRATEGY}")
model = ModelFactory(MODEL_NAME, strategy=STRATEGY).build()
best_model_path = f"results/{MODEL_NAME}_{STRATEGY}.pth"

if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
    print("Model loaded successfully.")
else:
    raise FileNotFoundError(f"Not find : {best_model_path}")

model = model.to(device)
model.eval()

 
cat_images = sorted(glob.glob(os.path.join(test_path, "cats", "*.jpg")))
dog_images = sorted(glob.glob(os.path.join(test_path, "dogs", "*.jpg")))

if not cat_images or not dog_images:
    raise FileNotFoundError("No images can be found in the test/cats or test/dogs folders!")

cat_path = cat_images[0]
dog_path = dog_images[0]

print(f"Selected cat image: {cat_path}")
print(f"Selected dog image: {dog_path}")

# Register a forward hook to retrieve all Conv2d layers.
activations = {}
layer_names = []

def get_activation(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook
 
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        layer_names.append(name)
        module.register_forward_hook(get_activation(name))

 
def visualize_image(img_path, class_name):
 
    model_dir = f"{output_base}/{MODEL_NAME}_{STRATEGY}"
    save_dir = f"{model_dir}/{class_name}"
    os.makedirs(save_dir, exist_ok=True)

    
    original_img = Image.open(img_path).convert("RGB")
    input_tensor = transform(original_img).unsqueeze(0).to(device)

 
    activations.clear()

 
    with torch.no_grad():
        _ = model(input_tensor)
 
    original_img.save(f"{save_dir}/original.jpg")

 
    fig_list = []   

    for i, layer_name in enumerate(layer_names):
        if layer_name not in activations:
            continue
        act = activations[layer_name]  # shape: (1, C, H, W)
        act = act.squeeze(0).cpu()     # (C, H, W)

 
        mean_act = torch.mean(act, dim=0).numpy()
 
        mean_act = (mean_act - mean_act.min()) / (mean_act.max() - mean_act.min() + 1e-8)
 
        layer_save_path = f"{save_dir}/layer_{i+1}_{layer_name.replace('.', '_')}.jpg"
        plt.figure(figsize=(8, 8))
        plt.imshow(mean_act, cmap='viridis')
        plt.title(f"Layer {i+1}: {layer_name}\n({act.shape[0]} channels averaged)")
        plt.axis('off')
        plt.savefig(layer_save_path, bbox_inches='tight', dpi=200)
        plt.close()

 
        fig_list.append((mean_act, f"Layer {i+1}\n{layer_name}"))

 
    n_layers = len(fig_list)
    cols = 4
    rows = (n_layers + cols) // cols + 1   

    plt.figure(figsize=(cols*5, rows*5))

 
    plt.subplot(rows, cols, 1)
    plt.imshow(np.array(original_img))
    plt.title(f"Original\n{class_name.capitalize()}")
    plt.axis('off')

 
    for idx, (feat, title) in enumerate(fig_list):
        plt.subplot(rows, cols, idx + 2)
        plt.imshow(feat, cmap='viridis')
        plt.title(title, fontsize=10)
        plt.axis('off')

    combined_path = f"{save_dir}/combined_all_layers.jpg"
    plt.savefig(combined_path, bbox_inches='tight', dpi=150)
    plt.close()

    print(f"{class_name} Visualization complete → {save_dir}")

 
visualize_image(cat_path, "cat")
visualize_image(dog_path, "dog")

print("\nAll feature maps visualized!")
print(f"Output position : {output_base}/{MODEL_NAME}_{STRATEGY}/")
print("Each category folder contains: original.jpg, feature maps for each layer, and combined_all_layers.jpg.")