import json
import shutil
import os

json_file="/disk/no_backup/ju878/CODA/nuscenes_lidar.json"

src_dir="/disk/ml/datasets/nuScenes/samples/LIDAR_TOP/"
dst_dir="/disk/no_backup/ju878/CODA/lidar/"

with open(json_file, 'r') as file:
    obj = json.load(file)

for file_name in os.listdir(src_dir):
    #file_name = file_name.split('_')[0]
    source = src_dir + file_name

    
    for file in obj:
        if obj[file].split('/')[2] == file_name:
            print('found')
            destination = dst_dir + file.split('.')[0] + '.bin'
            
            shutil.copy(source, destination)
            print('copied', file.split('.')[0] + '.bin')