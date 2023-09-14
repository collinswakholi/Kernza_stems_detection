from ultralytics import YOLO
import os
import numpy as np
import time
import subprocess

# clear gpu cache
import torch
torch.cuda.empty_cache()

# clear gpu memory
import gc
gc.collect()


img_sz = 2048
data_folder = "Data"

################################################################################################
# Shuffle data, split into training, validation, and test dataset
def run_shuffle_script(data_folder,img_sz):
    script_path = os.path.join("utils", "do_shuffle.py")
    process = subprocess.Popen(["python", script_path, data_folder, str(img_sz)])
    time.sleep(5)
    print('\nShuffle Done\n')
    process.terminate()
    process.wait()
    return


run_shuffle_script(data_folder, img_sz)
time.sleep(5)
################################################################################################

data_folder = os.path.join(data_folder, str(img_sz)) 
save_folder = data_folder+"_shuffled"
yaml_dir = os.path.join(save_folder, "data.yaml")
print("\nYAML Dir = "+yaml_dir+"\n")

# time.sleep(2)

# set training parameters
project = 'kernza_stems'
name = 'yolov8_model_'+str(img_sz)+'_'
data = yaml_dir
imgsz = img_sz
epochs = 500
batch = -1 # 2, 4, 8, 16, 32, 64, 128, 256 # batch size or -1 for auto (largest batch size possible)
optimizer = 'Adam' # 'SGD', 'Adam', 'AdamW', 'RMSprop', 'RAdam', 'Adamax', 'auto'
device = '0' # '0,1,2,3,4,5,6,7' # cuda device, i.e. 0 or 0,1,2,3 or cpu
patience = 100 # early stopping patience
verbose = True # print mAP every epoch
exist_ok = True # change to true if you want to overwrite previous results
name_val = name+"_val" # validation results
single_cls = False # train as single-class dataset
cache = True # use cache images for faster training


# train the model

if __name__ == '__main__':
    
    model = YOLO('yolov8m.yaml') # yolov8n.yaml, yolov8s.yaml, yolov8m.yaml, yolov8l.yaml, yolov8x.yaml
    
    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        optimizer=optimizer,
        verbose=verbose,
        name=name,
        project=project,
        single_cls=single_cls,
        cache = cache,
        exist_ok=exist_ok
    )
    # validate the model
    results = model.val(name=name_val, imgsz=imgsz, device=device, project=project, verbose=verbose, exist_ok=exist_ok, batch=int(np.round(4096/imgsz)))
    
