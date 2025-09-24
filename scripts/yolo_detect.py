from ultralytics import YOLO
from logging import log
import os

src = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Mosaic\Stitched-Mosaic\reel_0024_20250322-100021\filtered"
# Load a model 
model = YOLO('pt_models/yolov8m.pt')
# model = YOLO("h:\\YOLO Training\\runs\detect\\buildv8s_may30data_3_may275\\weights\\best.pt")
run_title = 'Abuja_Mosaic_reel_100021'
project = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Mosaic\Yolo\Detect"

results = model.predict(source=src, model=model,
                        device=[0], stream=True,
                        save_crop=True, save_conf=False,
                        save_txt=True, imgsz=1920,
                        conf=0.5, iou=0.3, save=True,
                        name=run_title, project = project,
                        classes=[0,2,3,5,7]
                        )

c = 0
for result in results:
    if c/1==c//100:
        print ('running...',c,end='\r')
    c += 1