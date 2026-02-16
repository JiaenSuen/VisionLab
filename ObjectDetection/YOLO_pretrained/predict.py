from ultralytics import YOLO

model = YOLO("weights/best.pt") 
results = model('data/test03.jpg')
results[0].show()  
results[0].save('out/output03.jpg')