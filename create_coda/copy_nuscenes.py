#!~/model_contradictions/create_coda/env/bin/python

import json
from nuscenes.nuscenes import NuScenes

datasets_root = "/disk/ml/datasets/"
#Load nuScenes
nusc_trainval = NuScenes(version='v1.0-trainval', dataroot=datasets_root + 'nuScenes', verbose=True)
# For nuScenes, the process is
# 1) Find "file_name": "nuscenes_xxx" in corner_case.json
# 2) Associate this file_name with the nuScenes token via nuscenes_sample_tokens.json
# 3) Get file via nuScenes sample_data.json

# Read JSONs



json_cornercases = datasets_root + 'CODA/base-val-1500/corner_case.json'
json_tokens = datasets_root + 'CODA/base-val-1500/nuscenes_indices.json'
json_nuscenes = datasets_root + 'nuScenes/v1.0-trainval/sample_data.json'
json_nuscenes_image = '/disk/ml/own_datasets/CODA/nuscenes_image.json'
json_nuscenes_lidar = '/disk/ml/own_datasets/CODA/nuscenes_lidar.json'

with open(json_cornercases, 'r') as f:
    data_cornercases = json.load(f)

with open(json_tokens, 'r') as f:
    data_tokens = json.load(f)

with open(json_tokens, 'r') as f:
    nuscenes_image = json.load(f)
    
with open(json_tokens, 'r') as f:
    nuscenes_lidar = json.load(f)
    


#with open(json_nuscenes, 'r') as f:
#    data_nuscenes = json.load(f)

annotations = data_cornercases["annotations"]
images = data_cornercases["images"]
categories = data_cornercases["categories"]

sensor_nuscenes_image = 'CAM_FRONT'
sensor_nuscenes_lidar = 'LIDAR_TOP'


for image in images:
    file_name = image['file_name']
    
    # Check if part of nuScenes  
    if ("nuscenes_" in file_name):

        # Get token ("nuscenes_033402.jpg": "1a41ba0751d5497ebd32df7c86950671")
        token_nuscenes = data_tokens[file_name]

        # Get nuScenes data
        my_sample = nusc_trainval.get('sample', token_nuscenes)
        cam_front_data = nusc_trainval.get('sample_data', my_sample['data'][sensor_nuscenes_image])
        nuscenes_image[file_name] = cam_front_data['filename']
        lidar_top_data = nusc_trainval.get('sample_data', my_sample['data'][sensor_nuscenes_lidar])
        nuscenes_lidar[file_name] = lidar_top_data['filename']

with open(json_nuscenes_image, 'w') as f:
    json.dump(nuscenes_image, f)

with open(json_nuscenes_lidar, 'w') as f:
    json.dump(nuscenes_lidar, f)
