# Anomaly Detection with Model Contradictions for Autonomous Driving

This repo is part of the following thesis **[Anomaly Detection with Model Contradictions for Autonomous Driving](https://cloud.schiller-lan.party/s/NcfsHT7K5EnAXM3)**.


## Requirements
* Clone this repo including the submodules: 
```bash
git clone --recurse-submodules git@github.com:vincentgeppert/model_contradictions.git
```

* Prepare your directories:
    * datasets_root (~ 1.65TB): Directory to store all downloaded datasets needed to create CODA. ("/disk/ml/datasets")
    * coda_root (~ 18GB): Directory to store the finished CODA dataset. ("/disk/ml/own_datasets/CODA")
    * working_directory (~ 300GB): Directory to store all data needed for the detection method. ("/disk/vanishing_data/ju878")
    * inference_root (~ 75GB): Directory inside working_directory where the results of the detection method and evaluation are stored. ("/disk/vanishing_data/ju878/inference")


## Create CODA dataset
* Download and arange all datasets inside the ```datasets_root``` according to ```create_coda/create_coda.ipynb```. This will take a long time since all datasets combined consist of ```~ 1.65TB```. (important: currently (03.08.2023) the google drive link for CODA is not working)
* Check ToDos and run ```create_coda/create_coda.ipynb (runtime: ~ 5min)```. Copies all needed files from the three datasets into ```coda_root```.
* Check ToDos and run ```create_coda/2d_to_3d_point_wise_clusters.ipynb (runtime: ~ 2h)```. This translates the 2D bounding boxes into 3D point-wise label proposals, by utilizing frustums to decrease point cloud size and  DBSCAN and mean shift clustering algorithms to produce pixel accurate object proposals.
* (Optional: You can create your own ```annotation_validation```, by manualy inspecting the clustering result of each annotation, and grading it with a label between 1-9 according to my thesis.)
* Check ToDos and run ```create_coda/get_gt_translate_to_kittiodometry.ipynb (runtime: ~ 35min)```. This creates the 3D point-wise annotated ground truth by labeling the point clouds according to the manual inspection in ```annotation_validation``` and saves the results in ```coda_root``` and transformed into KITTI lidar coordinates in the ```inference_root```.
* (Optional: Run visualizations from the ```create_coda/visualization```, to create the plots used in my thesis.)


## Evaluate detection method

### Setup dependencies
* Check ToDos and run ```evaluate_detection_method/get_kitti_raw_mapping.ipynb (runtime: instant)```. This creates the mapping to the KITTI_Raw data ```coda_root/kitti_mapping.json``` and ```coda_root/kitti_needed_raw_files.txt```
* Download all scenes specified in ```coda_root/kitti_needed_raw_files.txt``` from **[KITTI_Raw](https://www.cvlibs.net/datasets/kitti/raw_data.php)** and save them in ```datasets_root/KITTI_Raw``` (~ 28GB).
* Download pretrained models
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

**:warning: IMPORTANT :warning:**
**If you use any indices set other than ```indices_all```, the ```create_evaluation``` and ```evaluate_detection_method``` notebooks will not work.**


### Create CODA in KITTIodometry format for detection method
* Prepare ```poses``` in ```working_directory/prepared_poses/```:
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/kitti_raw2poses.ipynb (runtime: ~ 1min)```.
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/nuscenes2poses.ipynb (runtime: ~ 3min)```.
    * Check ToDos and run ```evaluate_detection_method/create_sequences/x2poses/once2poses.ipynb (runtime: ~ 3min)```.
* Check ToDos and run ```evaluate_detection_method/create_sequences/create_coda_for_detection_method.ipynb (runtime: ~ 6h; test_indices: ~ 4min)```. This copies and transforms all images of CODA into KITTIodometry format, forming sequences with the 8 preceeding and 8 subsequent images of the original datasets.

### Run the detection method


