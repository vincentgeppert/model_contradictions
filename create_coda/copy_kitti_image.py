import json
import shutil
import os


json_file="/disk/ml/datasets/CODA/base-val-1500/kitti_indices.json"

src_dir="/disk/ml/datasets/KITTI/object/data/training/image_2/"
dst_dir="/disk/no_backup/ju878/CODA/image/"

with open(json_file, 'r') as file:
    obj = json.load(file)

    for file_name in os.listdir(src_dir):
        #file_name = file_name.split('_')[0]
        source = src_dir + file_name
        
        for file in obj:
            if file.split('_')[1] == file_name:
                print('found')
                destination = dst_dir + file
                
                shutil.copy(source, destination)
                print('copied', file)
