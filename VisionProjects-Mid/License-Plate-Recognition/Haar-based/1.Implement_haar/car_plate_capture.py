import matplotlib.pyplot as plt
import cv2

def cv2img_via_plt_show(img):
    rgb  = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.title('Image Display via Matplotlib')
    plt.axis('off')  
    plt.show()

def haar_capture( file_name , dir_crop = "car_plate"):
    plate_cascade = cv2.CascadeClassifier("model/haarcascade_russian_plate_number.xml")
    img = cv2.imread(file_name)

    car_plate_set = []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  

    plates = plate_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=8,
        minSize=(90, 30),
        maxSize=(400, 150),
    )

    for (x, y, w, h) in plates:
        aspect_ratio = w / float(h)
        roi = gray[y:y+h, x:x+w]
        edges = cv2.Canny(roi, 100, 200)
        edge_density = edges.sum() / (w * h)



        if 2.0 < aspect_ratio < 6.0 and edge_density > 0.1:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cropped_img = img[y:y+h, x:x+w]
            filename = f'{dir_crop}/{file_name[0:-4]}_{x}_{y}.jpg'
            cv2.imwrite(filename, cropped_img)
            car_plate_set.append(cropped_img)

    return car_plate_set




if __name__ == "__main__":

    target_img = "test_image/Cars0.png"

    car_plate_set   =  haar_capture(  target_img )
    cv2img_via_plt_show(*car_plate_set)
    