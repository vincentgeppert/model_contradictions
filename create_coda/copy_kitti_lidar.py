import json
import shutil
import os


json_file="/disk/ml/datasets/CODA/base-val-1500/kitti_indices.json"

src_dir="/disk/ml/datasets/KITTI/object/data/training/velodyne/"
dst_dir="/disk/no_backup/ju878/CODA/lidar/"

with open(json_file, 'r') as file:
    obj = json.load(file)

    for file_name in os.listdir(src_dir):
        source = src_dir + file_name
        
        for file in obj:
            if file.split('_')[1].split('.')[0] == file_name.split('.')[0]:
                print('found')
                destination = dst_dir + "kitti_" + file_name
                
                shutil.copy(source, destination)
                print('copied', "kitti_" + file_name)
