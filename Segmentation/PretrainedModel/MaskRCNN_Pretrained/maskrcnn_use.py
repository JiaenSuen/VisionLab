import torch
import torchvision

from PIL import Image
from torchvision import transforms as T
import numpy as np
 
import cv2 
import random

from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)
model = model.cuda()
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
 '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',  'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
 
def get_prediction(img_path, threshold=0.85 ):
    img = Image.open(img_path) 
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    img = img.cuda() 
    pred = model([img]) 
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:pred_t+1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return masks, pred_boxes, pred_class




def random_color_masks(image):

    colors = [ [80, 70, 180], [250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190],[255, 192, 203]]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image==1], g[image==1], b[image==1] = colors[random.randrange(0, len(colors))]
    colored_mask = np.stack([r,g,b], axis=2)
    return colored_mask


def instance_segmentation(img_path, threshold=0.85, rect_th=2,text_size=2, text_th=2 ):
    masks, boxes, pred_cls = get_prediction(img_path, threshold=threshold )
    img_pil = Image.open(img_path).convert("RGB")
    img = np.array(img_pil)
 
    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        pt1 = tuple(int(x) for x in boxes[i][0])
        pt2 = tuple(int(x) for x in boxes[i][1])
        cv2.rectangle(img, pt1, pt2, color=(230, 100, 180), thickness=rect_th)
        cv2.putText(img, pred_cls[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, text_size, (230, 50, 200), thickness=text_th)
    return img, pred_cls, masks[i]


img, pred_classes, masks = instance_segmentation('./data/dog.jpeg')
img_pil = Image.fromarray(img)
img_pil.save("./data/result.png") 