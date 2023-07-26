import json
import os
import shutil
from typing import List, Dict, Tuple, Any

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
import argparse
import math

from tqdm import tqdm

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, Box
from nuscenes.utils.geometry_utils import BoxVisibility, transform_matrix
from nuscenes.utils.kitti import KittiDB

EARTH_RADIUS_METERS = 6.378137e6
REFERENCE_COORDINATES = {
    "boston-seaport": [42.336849169438615, -71.05785369873047],
    "singapore-onenorth": [1.2882100868743724, 103.78475189208984],
    "singapore-hollandvillage": [1.2993652317780957, 103.78217697143555],
    "singapore-queenstown": [1.2782562240223188, 103.76741409301758],
}


class KittiConverter:
    def __init__(self,
                 nusc_kitti_dir: str = '~/nusc_kitti',
                 cam_name: str = 'CAM_FRONT',
                 lidar_name: str = 'LIDAR_TOP',
                 image_count: int = 10,
                 nusc_version: str = 'v1.0-trainval',
                 split: str = 'sweeps'):
        """
        :param nusc_kitti_dir: Where to write the KITTI-style annotations.
        :param cam_name: Name of the camera to export. Note that only one camera is allowed in KITTI.
        :param lidar_name: Name of the lidar sensor.
        :param image_count: Number of images to convert.
        :param nusc_version: nuScenes version to use.
        :param split: Dataset split to use.
        """
        self.nusc_kitti_dir = os.path.expanduser(nusc_kitti_dir)
        self.cam_name = cam_name
        self.lidar_name = lidar_name

        # Create nusc_kitti_dir.
        if not os.path.isdir(self.nusc_kitti_dir):
            os.makedirs(self.nusc_kitti_dir)

        # Select subset of the data to look at.
        #self.nusc = NuScenes(version=nusc_version, dataroot='/disk/ml/datasets/nuScenes', verbose=True)

    def nuscenes_gt_to_kitti(self, sample_tokens, sample_names, dst_dir, nusc) -> None:
        """
        Converts nuScenes GT annotations to KITTI format.
        """
        kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
        kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse
        imsize = (1600, 900)

        token_idx = 0  # Start tokens from 0.

        # Create output folders.
        calib_folder = os.path.join(dst_dir)
        image_folder = os.path.join(dst_dir, 'image_2')
        lidar_folder = os.path.join(dst_dir, 'velodyne')
        for folder in [calib_folder, image_folder, lidar_folder]:
            if not os.path.isdir(folder):
                os.makedirs(folder)

        tokens = []
        times = []
        for index, sample_token in enumerate(sample_tokens):

            # Get sample data.
            sample = nusc.get('sample', sample_token)
            scene_token = sample['scene_token']
            scene = nusc.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            first_sample = nusc.get('sample', first_sample_token)
            first_lid_token = first_sample['data'][self.lidar_name]
            first_lid = nusc.get('sample_data', first_lid_token)
            first_lid_timestamp = first_lid['timestamp']
            sample_annotation_tokens = sample['anns']
            cam_front_token = sample['data'][self.cam_name]
            lidar_token = sample['data'][self.lidar_name]

            # Retrieve sensor records.
            sd_record_cam = nusc.get('sample_data', cam_front_token)
            sd_record_lid = nusc.get('sample_data', lidar_token)
            cs_record_cam = nusc.get('calibrated_sensor', sd_record_cam['calibrated_sensor_token'])
            pose_record_cam = nusc.get('ego_pose', sd_record_cam['ego_pose_token'])
            pose_recordc_lid = nusc.get('ego_pose', sd_record_lid['ego_pose_token'])
            cs_record_lid = nusc.get('calibrated_sensor', sd_record_lid['calibrated_sensor_token'])

            # Combine transformations and convert to KITTI format.
            # Note: cam uses same conventions in KITTI and nuScenes.
            lid_to_ego = transform_matrix(cs_record_lid['translation'], Quaternion(cs_record_lid['rotation']),
                                          inverse=False)
            pose_lid = transform_matrix(pose_recordc_lid['translation'], Quaternion(pose_recordc_lid['rotation']),
                                          inverse=False)
            pose_cam = transform_matrix(pose_record_cam['translation'], Quaternion(pose_record_cam['rotation']),
                                          inverse=True)
            ego_to_cam = transform_matrix(cs_record_cam['translation'], Quaternion(cs_record_cam['rotation']),
                                          inverse=True)
            velo_to_cam = ego_to_cam @ pose_cam @ pose_lid @ lid_to_ego

            # Convert from KITTI to nuScenes LIDAR coordinates, where we apply velo_to_cam.
            velo_to_cam_kitti = np.dot(velo_to_cam, kitti_to_nu_lidar.transformation_matrix)

            # Currently not used.
            imu_to_velo_kitti = np.zeros((3, 4))  # Dummy values.
            r0_rect = Quaternion(axis=[1, 0, 0], angle=0)  # Dummy values.

            # Projection matrix.
            p_left_kitti = np.zeros((3, 4))
            p_left_kitti[:3, :3] = cs_record_cam['camera_intrinsic']  # Cameras are always rectified.

            # Create KITTI style transforms.
            velo_to_cam_rot = velo_to_cam_kitti[:3, :3]
            velo_to_cam_trans = velo_to_cam_kitti[:3, 3]

            # Check that the rotation has the same format as in KITTI.
            assert (velo_to_cam_rot.round(0) == np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])).all()
            assert (velo_to_cam_trans[1:3] < 0).all()

            # Retrieve the token from the lidar.
            # Note that this may be confusing as the filename of the camera will include the timestamp of the lidar,
            # not the camera.
            filename_cam_full = sd_record_cam['filename']
            filename_lid_full = sd_record_lid['filename']
            # token = '%06d' % token_idx # Alternative to use KITTI names.
            token_idx += 1

            # Convert image (jpg to png).
            src_im_path = os.path.join(nusc.dataroot, filename_cam_full)
            dst_im_path = os.path.join(image_folder, sample_names[index] + '.png')
            if not os.path.exists(dst_im_path):
                im = Image.open(src_im_path)
                im.save(dst_im_path, "PNG")

            # Convert lidar.
            # Note that we are only using a single sweep, instead of the commonly used n sweeps.
            src_lid_path = os.path.join(nusc.dataroot, filename_lid_full)
            dst_lid_path = os.path.join(lidar_folder, sample_names[index] + '.bin')
            assert not dst_lid_path.endswith('.pcd.bin')
            #shutil.copy(src_lid_path, dst_lid_path)
            pcl = LidarPointCloud.from_file(src_lid_path)
            pcl.rotate(kitti_to_nu_lidar_inv.rotation_matrix)  # In KITTI lidar frame.
            with open(dst_lid_path, "w") as lid_file:
                pcl.points.T.tofile(lid_file)

            # Add to tokens.
            tokens.append(sample_token)

            # Create calibration file.
            kitti_transforms = dict()
            kitti_transforms['P0'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P1'] = np.zeros((3, 4))  # Dummy values.
            kitti_transforms['P2'] = p_left_kitti  # Left camera transform.
            kitti_transforms['P3'] = np.zeros((3, 4))  # Dummy values.
            #kitti_transforms['R0_rect'] = r0_rect.rotation_matrix  # Cameras are already rectified.
            kitti_transforms['Tr'] = velo_to_cam_kitti[:3, :]
            #kitti_transforms['Tr_imu_to_velo'] = imu_to_velo_kitti
            calib_path = os.path.join(calib_folder, 'calib.txt')
            with open(calib_path, "w") as calib_file:
                for (key, val) in kitti_transforms.items():
                    val = val.flatten()
                    val_str = '%.12e' % val[0]
                    for v in val[1:]:
                        val_str += ' %.12e' % v
                    calib_file.write('%s: %s\n' % (key, val_str))
                
            timestamp = sd_record_lid['timestamp']   
            relativ_timestamp = float(timestamp) - float(first_lid_timestamp) 
            times.append(relativ_timestamp / 1000000)
                    
        #iter_sample = nusc.get('sample', first_sample_token)
        #times = [] 
        #if iter_sample['token'] in sample_tokens:
        #    lidar_token = sample['data'][self.lidar_name]
        #    
        #while iter_sample['next'] != '':
        #    iter_sample_token = iter_sample['next']
        #    iter_sample = nusc.get('sample', iter_sample_token)
        #
        #
        #sd_record_lid['timestamp']
        times.sort()
        with open(f'{calib_folder}times.txt', 'w') as f:
            for time in times:
                f.write('%s\n' % (time))


