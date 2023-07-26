#!/bin/bash

dataset_root="/disk/ml/datasets/CODA/base-val-1500"
new_CODA_root="/disk/ml/own_datasets/CODA"

#rm -rf "$new_CODA_root"

mkdir "$new_CODA_root"

cp "$dataset_root/README.md" "$new_CODA_root/"
cp "$dataset_root/corner_case.json" "$new_CODA_root/"
cp "$dataset_root/kitti_indices.json" "$new_CODA_root/"
cp "$dataset_root/nuscenes_indices.json" "$new_CODA_root/"
mkdir "$new_CODA_root/image"
mkdir "$new_CODA_root/lidar"

set -e
. ./env/bin/activate
python copy_nuscenes.py
deactivate

python copy_kitti_image.py
python copy_kitti_lidar.py
python copy_nuscenes_image.py
python copy_nuscenes_lidar.py

