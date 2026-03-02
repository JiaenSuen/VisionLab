# dataset
import albumentations
import torch
from torch import nn
import numpy as np
from PIL import Image,ImageFile
import os
from pathlib import Path
from utils import StringOperations

ImageFile.LOAD_TRUNCATED_IMAGES = True




class CaptchaDataset:
    def __init__(self, image_paths, targets, orig_labels, resize=None):
        # resize = (height, width)
        self.image_paths = image_paths
        self.targets = targets
        self.orig_labels = orig_labels
        self.resize = resize

        mean = (0.5, 0.5, 0.5) #(0.485, 0.456, 0.406)
        std  = (0.5, 0.5, 0.5) #(0.229, 0.224, 0.225)
        self.aug = albumentations.Compose(
            [
                albumentations.Normalize(
                    mean, std, max_pixel_value=255.0, always_apply=True
                )
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item]).convert("RGB")
        targets = self.targets[item]

        if self.resize is not None:
            image = image.resize(
                (self.resize[1], self.resize[0]), resample=Image.BILINEAR
            )

        image = np.array(image)
        augmented = self.aug(image=image)
        image = augmented["image"]
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": torch.tensor(image, dtype=torch.float),
            "targets": torch.tensor(targets, dtype=torch.long),
            "length": len(targets),  
            "orig_label": self.orig_labels[item]
        }
            

def get_captcha_paths_and_labels(data_dir="dataset"):
    """
    Read all .jpg files from the dataset folder.
    Assuming the filename format is {label}.jpg, for example, abc123.jpg
    
    Returns:
        image_paths: List[str]  
        labels: List[list[int]] 
    """

    VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    image_paths = []
    targets = []
    orig_labels = []

    for filename in os.listdir(data_dir):
 
        name, ext = os.path.splitext(filename)
        if ext.lower() not in VALID_EXTS:
            continue
 
        label_str = name.upper()   

        try:
            target = StringOperations.encode_string(label_str)
            full_path = os.path.join(data_dir, filename)
            
            image_paths.append(full_path)
            targets.append(target)
            orig_labels.append(label_str)
            
        except Exception as e:
            continue

    return image_paths, targets, orig_labels


def collate_fn(batch):
    images = torch.stack([b["images"] for b in batch])

    labels = [b["targets"] for b in batch]
    lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {
        "images": images,
        "targets": labels,
        "lengths": lengths,
        "orig_label": [b["orig_label"] for b in batch]
    }


if __name__ == "__main__":

    image_paths, targets, orig_labels = get_captcha_paths_and_labels("dataset")
    dataset = CaptchaDataset(
        image_paths=image_paths,
        targets=targets,
        resize=(64, 160)  
    )
    print( list(StringOperations.CHARS))
    print("First Image Path : ", dataset.image_paths[0])
    print("First Label Path : ", dataset.targets[0])
    print("Dataset Size : ", len(dataset))


    if targets:
        print("\nExample : ")
        print("File Name -> RawString -> Encoded → Decoded")
        for i in range(min(5, len(targets))):
            orig = os.path.splitext(os.path.basename(image_paths[i]))[0]
            encoded = targets[i]
            decoded = StringOperations.decode_ids(encoded)
            print(f"{orig:12} ->  {orig:8} ->  {encoded} ->  {decoded}")
 
 

    NUM_CLASSES = len(StringOperations.CHARS) + 1  

    # Simulation model output : (seq_len, batch=1, num_classes)
    seq_len = 24   
    fake_logits = torch.randn(seq_len, 1, NUM_CLASSES) * 0.3
    fake_logits[5:8, 0, 14] = 5.0   # Force a segment to favor a certain category
    fake_logits[12:15, 0, 23] = 6.2

 
    print("=== Simulated CTC decoding demonstration ===")
    print("Raw logits shape:", tuple(fake_logits.shape))

    pred = fake_logits[:, 0, :]  # Take the portion where batch=0 (seq_len, num_classes)
    decoded = StringOperations.ctc_decode(pred)
    print("CTC greedy decode Result : ", decoded)

 
    indices = pred.argmax(dim=-1)
    print("Category index after argmax : " , indices.tolist())
    print("Directly decode_ids: ", StringOperations.decode_ids(indices))
    print("ctc_decode process : ", StringOperations.ctc_decode(indices))