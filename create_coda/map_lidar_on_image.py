import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
import yaml
import re
import json

def read_calib_file(filepath, data_dic):    
    """
    Inspired by: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()

            if len(line) == 0: continue
            key, value = line.split(':', 1)

            data_dic[key] = np.array([float(x) for x in value.split()])
    return data_dic 

def project_velo_to_cam2(calib):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """
    P_velo2cam_ref = np.vstack((calib['Tr'].reshape(3, 4), np.array([0., 0., 0., 1.])))  # velo2ref_cam
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ P_velo2cam_ref
    
    return proj_mat, P_velo2cam_ref
    
def project_to_image(points, proj_mat):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """
    num_pts = points.shape[1]

    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    
    return points[:2, :]

def get_labels_and_colors(anomaly_labels, color_dict):
    if color_dict:
        max_sem_key = 0
        for key, data in color_dict.items():
            if key + 1 > max_sem_key:
                max_sem_key = key + 1
        color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for key, value in color_dict.items():
            color_lut[key] = np.array(value, np.float32)
    else:
        quit()

    sem_label = np.zeros((0, 1), dtype=np.int16)
    sem_label_color = np.zeros((0, 3), dtype=np.float32)

    sem_label = anomaly_labels
    sem_label_color = color_lut[sem_label]
    sem_label_color = sem_label_color.reshape((-1, 3))

    #check
    assert(sem_label.shape[0] == sem_label_color.shape[0])

    return sem_label_color


def get_point_camerafov(pts_velo, calib, img, img_width, img_height):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """    
    # projection matrix (project from velo2cam2)
    pts_velo_xyz = pts_velo[:, :3]
    #mask = pts_velo_xyz[:, 0] > 20
    #pts_velo_xyz = pts_velo_xyz[mask]
    
    proj_velo2cam2, P_velo2cam_ref = project_velo_to_cam2(calib)
    # apply projection
    pts_2d = project_to_image(pts_velo_xyz.transpose(), proj_velo2cam2)
    print(pts_2d.shape)
    # Filter lidar points to be within image FOV
    
    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo_xyz[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]   #xy
    print(imgfov_pc_pixel.shape)
    print('---------')
    #Filter semantic color
    #sem_label_color_velo = label_color[inds, :]
    #sem_label_color_velo = np.array(sem_label_color_velo).astype(np.int16)
    
    for i in range(imgfov_pc_pixel.shape[1]):
        #color_label = np.array(sem_label_color_velo[i, :])
        #b=int(color_label[0])
        #g=int(color_label[1])
        #r=int(color_label[2])
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                int(np.round(imgfov_pc_pixel[1, i]))),
                2, color=(255,0,0), thickness=-1)
        
    return img

def save_img_to_file(img, path_to_save, seq , frame_image):
    if not os.path.exists(os.path.join(path_to_save, seq)):
        os.makedirs(os.path.join(path_to_save, seq))
    save_path = os.path.join(path_to_save, seq, frame_image)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def map_anomalies_on_image(path_velo, path_labels, path_img, path_clustering_3, path_clustering_4, path_calibrations, path_save, sequence, frame_img):
    #open point cloud
    pc_velo_camerafov = np.fromfile(path_velo, dtype=np.float32).reshape((-1,4))
    #anomaly_labels = np.fromfile(path_labels, dtype=np.int16).reshape((-1))
    #clusters_3 =  np.fromfile(path_clustering_3, dtype=np.int16).reshape((-1)) #self:dyn and sup:static
    #clusters_4 =  np.fromfile(path_clustering_4, dtype=np.int16).reshape((-1)) #self:static and sup:dyn

    #mask_consistent_static = anomaly_labels == 1
    #mask_consistent_dyn = anomaly_labels == 2
    #mask_inconsistent_3 = anomaly_labels == 3
    #mask_inconsistent_4 = anomaly_labels == 4
#
    #mask_inconsistent_no_clusters_3 = clusters_3 == -1 #-1 indicate that these points are outlier and do not belong to a cluster
    #anomalie_labels_inconsistent_3 = anomaly_labels[mask_inconsistent_3]
    #anomalie_labels_inconsistent_3[mask_inconsistent_no_clusters_3] = -1
    #
    #mask_inconsistent_no_clusters_4 = clusters_4 == -1 #-1 indicate that these points are outlier and do not belong to a cluster
    #anomalie_labels_inconsistent_4 = anomaly_labels[mask_inconsistent_4]
    #anomalie_labels_inconsistent_4[mask_inconsistent_no_clusters_4] = -1
#
    #anomaly_labels[mask_consistent_static] = 1
    #anomaly_labels[mask_consistent_dyn] = 2
    #anomaly_labels[mask_inconsistent_3] = anomalie_labels_inconsistent_3
    #anomaly_labels[mask_inconsistent_4] = anomalie_labels_inconsistent_4
    
    #{-1:[0,0,0], 1:[34,139,34], 2:[65,105,225], 3:[178,34,34], 4:[225,225,0]}
    #color_dict = {-1:[0,0,0], 1:[34,139,34], 2:[65,105,225], 3:[178,34,34], 4:[225,225,0]}
    #sem_label_color = get_labels_and_colors(anomaly_labels, color_dict)
    
    rgb = cv2.cvtColor(cv2.imread(os.path.join(path_img)), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    calib_dic = {}
    calib = read_calib_file(path_calibrations, calib_dic)

    img = get_point_camerafov(pc_velo_camerafov, calib, rgb, img_width, img_height)

    #if anomaly_labels.shape[0] != 0:
    save_img_to_file(img, '/disk/vanishing_data/ju878/CODA_for_Finn_sequences/lidar_to_image/', sequence, frame_img)

def main(seq):
    seq = '{0:04d}'.format(int(seq))
    # load config file
    #config_filename = 'anomaly_detection/config/config_paths.yaml'
    #config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    
    path_dataset = '/disk/vanishing_data/ju878/CODA_for_Finn_sequences/sequences'
    path_inference = '/disk/vanishing_data/ju878/log_finn'
    path_anomalies_sup_self = os.path.join(path_inference, 'anomalies_sup_self')

    frames = os.listdir(os.path.join(path_dataset, seq, 'velodyne'))
    frames = sorted(frames, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    for frame in range(len(frames)):
        frame_label = frames[frame]
        frame_image = frames[frame].split('.')[0] +'.png'
        frame_clusters_incon_static_dyn_3 = frames[frame].split('.')[0] +'_cluster_incon_static_dyn_3.bin'
        frame_clusters_incon_dyn_static_4 = frames[frame].split('.')[0] +'_cluster_incon_dyn_static_4.bin'
        
        path_calib_frame = os.path.join(path_dataset, seq, 'calib.txt')
        path_pc_velo_camerafov_frame = os.path.join(path_dataset, seq, 'velodyne', frame_label)
        path_clusters_frame_3 = os.path.join(path_anomalies_sup_self, seq, 'clusters', frame_clusters_incon_static_dyn_3)
        path_clusters_frame_4 = os.path.join(path_anomalies_sup_self, seq, 'clusters', frame_clusters_incon_dyn_static_4)
        path_anomaly_labels_frame = os.path.join(path_anomalies_sup_self, seq, 'anomaly_labels', frame_label)
        path_image_frame = os.path.join(path_dataset, seq, 'image_2', frame_image)

        map_anomalies_on_image(path_pc_velo_camerafov_frame, path_anomaly_labels_frame, path_image_frame, path_clusters_frame_3, path_clusters_frame_4, path_calib_frame, path_anomalies_sup_self, seq, frame_image)
        
if __name__ == '__main__':
    #[1,2,3,4,5,1396,1490,1492,1058]
    sequences = [1,2,3,4,5]
    with Pool(12) as p:
        p.map(main, sequences)