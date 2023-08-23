from ultralytics import YOLO
import glob
import os
import numpy as np
import pandas as pd
import torch
torch.cuda.empty_cache()
import time

import gc
gc.collect()

img_sz = 2048

image_folder = '/RawImages/AkronTest_CO' # change this to the path of your images

base_folder = '_'.join(image_folder.split('/')[-2:])

# set inference parameters
name = 'yolov8_model_'+str(img_sz)+'_Images_'+base_folder
project = 'YOLOResults'
save = True # save image results  ######################################################################
save_txt = False # save results to *.txt
save_conf = False # save confidences in --save-txt labels
show_labels = False   # hide labels
show_conf = False # hide confidences
line_width = 2
batch = -1 # batch size
visualize = False # visualize model features
conf_thres = 0.28 # confidence threshold
iou_thres = 0.55 # NMS IoU threshold
imgsz = img_sz # inference size (pixels)
exist_ok = True # if True, it overwrites current 'name' saving folder #####################################
half = True # use FP16 half-precision inference True/False
cache = True # use cache images for faster inference

img_fmt = '.JPEG' # image format


img_paths = glob.glob(os.path.join(image_folder, '**/*'+img_fmt), recursive=True) #get the path of each image in folder and subfolders
# exclude file paths wit "B_"
img_paths = [x for x in img_paths if "B_" not in x] # exclude file paths with "B_" (these are the images that contained barcode/QR data)

# shuffle the list of images
np.random.shuffle(img_paths)

num_imgs = len(img_paths)
print("NUmber of images found = "+str(num_imgs))

# split the list of images into batches of 10 images
nx = 10 # number of images per batch
img_paths2 = [img_paths[i:i + nx] for i in range(0, len(img_paths), nx)]

# load model
model_location = '/kernza_stems/yolov8_model_2048_1/weights/best.pt' # change this to the path of your model
model = YOLO(model_location)

cwd = os.getcwd()

csv_folder = os.path.join(cwd,project,'yolov8_model_'+str(img_sz)+'_csv_'+base_folder)
if not os.path.exists(csv_folder):
  os.mkdir(csv_folder)
  
for i, img_paths in enumerate(img_paths2):
  save_name = os.path.join(csv_folder,('stem_count_'+str(i)+'.csv')) # change name when you change folder

  # do inference and save
  tic = time.time()
  img_results = model.predict(source=img_paths, save=save, save_txt=save_txt, save_conf=save_conf, show_labels=show_labels, show_conf=show_conf,
                              line_width=line_width, visualize=visualize, project=project, name=name, imgsz=imgsz, conf=conf_thres, half=half,
                              iou=iou_thres, exist_ok=exist_ok, batch=batch, cache=cache
  )

  elapsed = 1000*(time.time()-tic)
  time_per_image = elapsed/num_imgs
  print("Infernce time per image = "+str(time_per_image)+ " ms")

  # loop through the results, get stem count, save
  Predictions = pd.DataFrame(columns = ['ImagePath','StemCount'])
  n = 0
  for img_result in img_results:
    # preds = img_result.__getitem__(0)
    preds = img_result
    try:
      preds = preds.cpu().numpy()
    except:
      print('Prediction was on CPU')
    
    n_dets = int(len(preds))
    im_name = img_paths[n]

    Predictions.at[n,'ImagePath'] = im_name
    Predictions.at[n,'StemCount'] = n_dets
    n+=1

  # save results in csv
  Predictions.to_csv(save_name)