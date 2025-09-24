# this code runs two layers of yolo. 1: yolov8l and 2:custom built yolov8 pt for faces and numplates
#and exports annotaiton files

from ultralytics import YOLO
from logging import log
import os,csv, cv2,datetime
from math import ceil
from scripts import list_segmenter

# funcions
def read_ann_file(file_path):
    #this functions reads the annotatio file and returns a list with each line as a tuple
    list = []
    with open(file_path, 'r') as file:
        for line in file:
            x = line.strip().split()  # split the line into a list
            list.append(x)
    return list

def cv2_np_list (box, img_path):
    #crop and return as numpy array
    img = cv2.imread(img_path)
    y0,x0,c = img.shape
    x = int(float(box[1])*x0)
    y = int(float(box[2])*y0)
    w = int(float(box[3])/2*x0)
    h = int(float(box[4])/2*y0)
    crop = img[y-h:y+h,x-w:x+w]
    return crop,y0,x0

def csv_write (path,boxes):
    # write final bboxes as csv file
    csv_f = open (path, 'w',newline='')
    writer = csv.writer(csv_f, delimiter=" ")
    for k,box in enumerate(boxes):
        writer.writerow(['0',box[0],box[1],box[2],box[3]])

start = datetime.datetime.now()
print(start.strftime('%H:%M:%S'))
current_run = str(start.strftime('%H:%M:%S')).replace(":","")

#source paths
src_images = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Mosaic\Stitched-Mosaic\reel_0022_20250322-084221\filtered"
src_labels = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Mosaic\Yolo\Detect\Abuja_Mosaic_reel_084221\labels"

my_model = YOLO(r"D:\PLACE - Zotac\YOLO_jun2024\pt_models\yolov8s_may24_best.pt")

#project = "H:\\YOLO Training\\run"
run_title = 'Abuja_Mosaic_reel_084221'
project = r"D:\PLACE - Zotac\Nigeria\Abuja Ground\Mosaic\Yolo\Squired"

#Model Parameters
confi = 0.01   # Confidence threshold
iou = 0.05      # Intersection Over Union threshold

#feed each image solo
for n,file in enumerate(os.listdir(src_labels)):
    origin_image = os.path.join(src_images,file.replace('.txt','.jpg'))
    out_path = origin_image.replace('.jpg','.txt')

    crops = read_ann_file(os.path.join(src_labels,file))

    if os.path.exists(out_path)==True:
        continue

    crops_np =[]
    for crop in crops:
        if crop[0] in ['0','2','3','5','7']:
            crop_np,imgw,imgh =cv2_np_list(crop,origin_image)
            crops_np.append(crop_np)
    if len(crops_np) == 0:
        continue
    #detect on crops
    results = my_model.predict(source=crops_np,
                            save=True,imgsz=640,iou=iou,conf=confi, stream=True,device=[0],
                            save_crop=False, save_txt=True,
                            project=f'{project}\\{current_run}_crops',
                            name=run_title
                            )
    
    #prep results lists for writing
    xbboxs,classes = [],[]
    for n,result in enumerate(results):
        for c,box in enumerate(result.boxes):
            detect_box = box.xywhn.tolist()[0]
            float_list = [float(i) for i in crops[n][1:5]]
            x1,y1,w1,h1 = float_list[0],float_list[1],float_list[2],float_list[3]
            x2,y2,w2,h2 = detect_box[0],detect_box[1],detect_box[2],detect_box[3]
            xbboxs.append([((x2*w1)+(x1-w1/2)),((y2*h1)+(y1-h1/2)),(w1*w2),(h1*h2)])
            #clas = box.cls.tolist()[0]
            #classes.append(str(box.cls.tolist()[c]))

            #xbboxs.append(list(map(lambda x, y: x * y, detect_box, float_list)))
    # print (xbboxs)
    print (f'{n}/{len(os.listdir(src_labels))}',end='\r')
    csv_write(out_path,xbboxs)

end = datetime.datetime.now()
print (end)
print ('inference duration= ',str(end-start))