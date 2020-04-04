import os
from glob import glob
import shutil
import cv2

output = "/home/tupm/HDD/datasets/2d_face/part1/dir_001_filtered"
input_path = "/home/tupm/HDD/datasets/2d_face/part1/dir_001/*"
os.makedirs(output, exist_ok=True)

folders = glob(input_path)
for folder_path in folders:
    os.makedirs(folder_path.replace('dir_001', 'dir_001_filtered'), exist_ok=True)
    images = glob(folder_path + '/*')
    i = 0
    for source in images:
        if i >= 10:
            break
        img = cv2.imread(source)
        if img is not None:
            shutil.copy(source, source.replace('dir_001', 'dir_001_filtered'))
            i+=1