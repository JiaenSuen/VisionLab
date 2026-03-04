import cv2
import matplotlib.pyplot as plt

def show_yolo_annotation(image_path, label_path, class_names=None):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    with open(label_path, 'r') as f:
        labels = f.readlines()

    for line in labels:
        cls, x_center, y_center, bw, bh = map(float, line.strip().split())
        cls = int(cls)

        x_center, y_center, bw, bh = x_center * w, y_center * h, bw * w, bh * h
        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        cv2.rectangle(img, (x1, y1), (x2, y2), (255,136,255), 2)
        label_text = class_names[cls] if class_names else str(cls)
        cv2.putText(img, label_text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,136,255), 2)


    plt.imshow(img)
    plt.imsave("out.jpg",img)
    plt.axis("off")
    plt.show()

with open("input/class_names.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f]


show_yolo_annotation("input/test1.jpg", "input/test1.txt", class_names)
