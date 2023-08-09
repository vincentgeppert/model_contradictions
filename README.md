# Anomaly Detection with Model Contradictions for Autonomous Driving

This repo is part of the following thesis **[Anomaly Detection with Model Contradictions for Autonomous Driving](https://publikationen.bibliothek.kit.edu/1000161266)**.


## Overview
```bash
└─┬ Create CODA (~ 3h) (create_coda/) 
  │ ├─ Download all datasets (~ 1.65TB)
  │ ├─ create_coda (~ 5min)
  │ ├─ 2d_to_3d_point_wise_clusters (~ 2h)
  │ ├─ (annotation_validation)
  │ ├─ get_gt_translate_to_kittiodometry (~ 35min)
  │ └─ (visualization)
  │
  └ Evaluate detection method (~ 3.5-5.5days)
    └─┬ Setup dependecies (~ 4h)
      │ ├─ get_kitti_raw_mapping (~ instant)
      │ ├─ Download KITTI_Raw(~ 28GB)
      │ ├─ Download pretrained models
      │ ├─ Setup environments (~ 3h)      
      │ └─ set_indices_for_detection_method (~ instant)
      │
      ├ Create CODA in KITTIodometry format (~ 6h) (evaluate_detection_method/create_sequences/) 
      │ ├─ kitti_raw2poses (~ 1min)
      │ ├─ nuscenes2poses (~ 3min)
      │ ├─ once2poses (~ 3min)
      │ └─ create_coda_for_detection_method (~ 6h)
      │
      ├ Run the detection method (~ 3-5days) (supervised_unsupervised_anomaly/)
      │ ├─ sup_semantic
      │ ├─ sup_mos 
      │ ├─ combine semantic and mos
      │ ├─ sup_ground_seg
      │ ├─ self_scene_flow
      │ ├─ self_odometry
      │ ├─ self_motion_labels_2stage
      │ ├─ compare_labels_and_cluster
      │ └─ map_anomalies_on_image
      │
      └ Evaluate the detection method (~ 5.5h) (evaluate_detection_method/)
        ├─ create_evaluation_images (~ 2h)
        ├─ create_evaluation_annotations (~ 2.5h)
        └─ evaluate_detection_method (~ 30sec)                
```

## Requirements
* Clone this repo including the submodules: 
```bash
git clone --recurse-submodules git@github.com:vincentgeppert/model_contradictions.git
```

* Prepare your directories:
    * datasets_root (require ~ 1.65TB): Directory to store all downloaded datasets needed to create CODA. ("/disk/ml/datasets")
    * coda_root (require ~ 18GB): Directory to store the finished CODA dataset. ("/disk/ml/own_datasets/CODA")
    * working_directory (require ~ 300GB): Directory to store all data needed for the detection method. ("/disk/vanishing_data/ju878")
    * inference_root (require ~ 75GB): Directory inside working_directory where the results of the detection method and evaluation are stored. ("/disk/vanishing_data/ju878/inference")


## Create CODA dataset (create_coda/)
* Download and arange all datasets inside the ```datasets_root``` according to ```create_coda/create_coda.ipynb```. This will take a long time since all datasets combined consist of ```~ 1.65TB```. (important: currently (03.08.2023) the google drive link for CODA is not working)
* Check ToDos and run ```create_coda/create_coda.ipynb (runtime: ~ 5min)```. Copies all needed files from the three datasets into ```coda_root```.
* Check ToDos and run ```create_coda/2d_to_3d_point_wise_clusters.ipynb (runtime: ~ 2h)```. This translates the 2D bounding boxes into 3D point-wise label proposals, by utilizing frustums to decrease point cloud size and  DBSCAN and mean shift clustering algorithms to produce pixel accurate object proposals.
* (Optional: You can create your own ```annotation_validation```, by manualy inspecting the clustering result of each annotation, and grading it with a label between 1-9 according to my thesis.)
* Check ToDos and run ```create_coda/get_gt_translate_to_kittiodometry.ipynb (runtime: ~ 35min)```. This creates the 3D point-wise annotated ground truth by labeling the point clouds according to the manual inspection in ```annotation_validation``` and saves the results in ```coda_root``` and transformed into KITTI lidar coordinates in the ```inference_root```.
* (Optional: Run visualizations from the ```create_coda/visualization```, to create the plots used in my thesis. All visualizations are save in ```graphics```)


## Evaluate detection method

### Setup dependencies
* Check ToDos and run ```evaluate_detection_method/get_kitti_raw_mapping.ipynb (runtime: instant)```. This creates the mapping to the KITTI_Raw data ```coda_root/kitti_mapping.json``` and ```coda_root/kitti_needed_raw_files.txt```
* Download all scenes specified in ```coda_root/kitti_needed_raw_files.txt``` from **[KITTI_Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)** and save them in ```datasets_root/KITTI_Raw``` (~ 28GB).
* Download pretrained models from training with KITTI-360 for **[SalsaNext semantic](https://drive.google.com/file/d/1F96PqSejX_kXAoTa88gDH6NpM5Ze_Lv0/view)** and **[SalsaNext motion](https://drive.google.com/file/d/150z3yCYLpwAD6KpdsUbWyOs0GKpLsJVW/view)**, unzip and save them in ```working_directory/pretrained_models```.
* Download pretrained models for self_scene_flow by running ```bash supervised_unsupervised_anomaly/self_scene_flow/flowstep3d/scripts/download_models.sh```
* Create environments used by the detection methods:
    * Create venv for self_scene_flow (~ 9GB):
    * Create conda env for sup_semantic (~ 3.5GB):
    * Create conda env for self_odometry (~ 7.5GB):

self_scene_flow
```bash 
python3 -m venv venv
source venv/bin/activate
cd supervised_unsupervised_anomaly/self_scene_flow/flowstep3d
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt  
cd lib/pointnet2
python3 setup.py install
```

install conda
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
bash Anaconda-latest-Linux-x86_64.sh
#run conda init
#reopen terminal
```

sup_semantic
```bash
cd supervised_unsupervised_anomaly/sup_semantic/SalsaNext
conda env create -f salsanext_environment.yml --name salsanext
conda activate salsanext
#conda deactivate
```

self_odometry
```bash
cd supervised_unsupervised_anomaly/self_odometry/DeLORA
conda env create -f conda/DeLORA-py3.9_environment.yml --name DeLORA-py3.9
conda activate DeLORA-py3.9
pip3 install -e .
#conda deactivate
```

* Check ToDos and run ```evaluate_detection_method/set_indices_for_detection_method.ipynb (runtime: instant)```. This sets the specified sequences and paths in every config file.

**:warning: IMPORTANT:**
**If you use any indices set other than ```indices_all```, the ```create_evaluation``` and ```evaluate_detection_method``` notebooks will not work.**


### Create CODA in KITTIodometry format for detection method (evaluate_detection_method/create_sequences/)
* Prepare ```poses``` in ```working_directory/prepared_poses/```:
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/kitti_raw2poses.ipynb (runtime: ~ 1min)```.
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/nuscenes2poses.ipynb (runtime: ~ 3min)```.
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/once2poses.ipynb (runtime: ~ 3min)```.
* Check ToDos and run ```evaluate_detection_method/create_sequences/create_coda_for_detection_method.ipynb (runtime: ~ 6h; test_indices: ~ 4min)```. This copies and transforms all images of CODA into KITTIodometry format, forming sequences with the 8 preceeding and 8 subsequent images of the original datasets.

### Run the detection method (supervised_unsupervised_anomaly/)

#### Supervised Semantic Segmentation (sup_semantic)
```bash
conda activate salsanext
cd supervised_unsupervised_anomaly/sup_semantic/SalsaNext/
./eval.sh -d /disk/vanishing_data/ju878/CODA_for_detection_method/sequences -p /disk/vanishing_data/ju878/inference -m /disk/vanishing_data/ju878/pretrained_models/pretrained_SalsaNext_semantic/ -c 30 -s valid -e False
#conda deactivate
```
#### Supervised Motion Segmentation (sup_mos)
```bash
# if not already acitvated activate conda env salsanext
conda activate salsanext
cd supervised_unsupervised_anomaly/sup_mos/LiDAR-MOS/
python utils/gen_residual_images_Kitti_odometry.py
cd mos_SalsaNext/
./eval.sh -d /disk/vanishing_data/ju878/CODA_for_detection_method/sequences -p /disk/vanishing_data/ju878/inference -m /disk/vanishing_data/ju878/pretrained_models/pretrained_SalsaNext_mos/ -c 30 -s valid
```

#### Combine sup_semantic and sup_mos
```bash
# if not already acitvated activate conda env salsanext
conda activate salsanext
cd supervised_unsupervised_anomaly/sup_mos/LiDAR-MOS/
python utils/combine_semantics.py
```

#### Supervised Ground Segmentation (sup_ground_seg)
```bash
# deactivate all (base) conda envs
conda deactivate 
cd supervised_unsupervised_anomaly/sup_ground_seg/GndNet/
python evaluate_SemanticKITTI.py --config ./config/config_kittiSem.yaml --resume ./trained_models/checkpoint.pth.tar --data_dir /disk/vanishing_data/ju878/CODA_for_detection_method/sequences --logdir /disk/vanishing_data/ju878/inference
```

#### Self-Supervised Scene Flow (self_scene_flow)
```bash
# deactivate all (base) conda envs
conda deactivate 
source venv/bin/activate
cd supervised_unsupervised_anomaly/self_scene_flow/flowstep3d/
python run.py -c configs/test/flowstep3d_self_KITTI_odometry.yaml
```

#### Self-Supervised Odometry (self_odometry)
```bash
conda activate DeLORA-py3.9
cd supervised_unsupervised_anomaly/self_odometry/DeLORA/
python bin/preprocess_data.py
python bin/run_testing.py --checkpoint /disk/no_backup/ju878/model_contradictions/supervised_unsupervised_anomaly/self_odometry/DeLORA//checkpoints/kitti_example.pth --testing_run_name 2
conda deactivate
```
* Create ```inference_root/self_pose_estimation```and copy all ```transformation_kitti_nnnn.npy``` from ```supervised_unsupervised_anomaly/self_odometry/DeLORA/mlruns/<number>/<hash>/artifacts``` into ```inference_root/self_pose_estimation```.

#### Self-Supervised Motion Labels
```bash
python supervised_unsupervised_anomaly/anomaly_detection/self_motion_labels/self_motion_labels_2stage.py
```

#### Comparison between Supervised and Self-Supervised Models
```bash
python supervised_unsupervised_anomaly/anomaly_detection/compare_and_cluster/compare_labels_and_cluster.py
python supervised_unsupervised_anomaly/anomaly_detection/compare_and_cluster/map_anomalies_on_image.py
```

### Evaluate the detection method (evaluate_detection_method/)
* Check ToDos and run ```evaluate_detection_method/create_evaluation_images.ipynb (runtime: ~ 2h)``` and ```evaluate_detection_method/create_evaluation_annotations.ipynb (runtime: ~ 2.5h)```. 
* Check ToDos and run ```evaluate_detection_method/evaluate_detection_method.ipynb (runtime: ~ 30sec)```. This creates the metrics for the evaluation.
* (Optional: Run visualizations from the ```evaluate_detection_method/visualization```, to create the plots used in my thesis. All visualizations are save in ```graphics```)

### Inference folder after running all the models
```bash
├── anomalies_sup_self                      # compare labels and cluster / map anomalies on image
├── ground_removal                          # ground segmentation labels
├── groundtruth_eval_all                    # create evaluation images
├── groundtruth_eval_all_annotation         # create evaluation images
├── original_labels                         # get gt translate to kittiodometry
├── original_labels_annotation              # get gt translate to kittiodometry
├── SalsaNext_combined_semantics_mos        # semantic motion labels
├── SalsaNext_mos                           # motion labels
├── SalsaNext_semantics                     # semantic labels
├── scene_flow                              # scene flow predictions
├── self_motion_labels                      # self motion labels 2stage
└── self_pose_estimation                    # relative transformations
```
