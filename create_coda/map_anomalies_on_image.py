import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from multiprocessing import Pool
import yaml
import re
from tqdm import tqdm

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


def get_point_camerafov(pts_velo, calib, img, img_width, img_height, label_color, dataset):
    """
    Inspired by: https://github.com/darylclimb/cvml_project/blob/master/projections/lidar_camera_projection/utils.py
    """
    # projection matrix (project from velo2cam2)
    pts_velo_xyz = pts_velo[:, :3]
    proj_velo2cam2, P_velo2cam_ref = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_velo_xyz.transpose(), proj_velo2cam2)
    

    inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                (pts_velo_xyz[:, 0] > 0)
                )[0]
    
    if dataset == 'nuscenes':
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo_xyz[:, 0] > 1)
                    )[0]
    elif dataset == 'once':
        #for i, pt_2d in enumerate(pts_2d[0]):
        #    if pt_2d < img_width / 2:
        #        pts_2d[0, i] = pt_2d - (((img_width / 2 - pt_2d) ** 0.6) ** 0.6)
        #    elif pt_2d > img_width / 2:
        #        pts_2d[0, i] = pt_2d + (((pt_2d - img_width / 2) ** 0.6) ** 0.6)
        #for i , pt_2d in enumerate(pts_2d[1]):
        #    if pt_2d < img_height / 1.3:
        #        pts_2d[1, i] = pt_2d - ((pt_2d ** 0.7) ** 0.7)
        #    elif pt_2d > img_height / 1.3:
        #        pts_2d[1, i] = pt_2d - ((pt_2d ** 0.78) ** 0.78)
                
        inds = np.where((pts_2d[0, :] < img_width) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_height) & (pts_2d[1, :] >= 0) &
                    (pts_velo_xyz[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]   #xy

    #Filter semantic color
    sem_label_color_velo = label_color[inds, :]
    sem_label_color_velo = np.array(sem_label_color_velo).astype(np.int16)
    
    for i in range(imgfov_pc_pixel.shape[1]):
        color_label = np.array(sem_label_color_velo[i, :])
        b=int(color_label[0])
        g=int(color_label[1])
        r=int(color_label[2])
        cv2.circle(img, (int(np.round(imgfov_pc_pixel[0, i])),
                int(np.round(imgfov_pc_pixel[1, i]))),
                2, color=(b,g,r), thickness=-1)
        
    return img

def save_img_to_file(img, path_to_save, seq):
    if not os.path.exists(os.path.join(path_to_save, seq)):
        os.makedirs(os.path.join(path_to_save, seq))
    save_path = os.path.join(path_to_save, seq, seq + '.png')
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def map_anomalies_on_image(path_velo, path_labels, path_img, path_clustering_3, path_clustering_4, path_original_label, path_calibrations, path_save, sequence, frame_img, dataset):
    #open point cloud
    pc_velo_camerafov = np.fromfile(path_velo, dtype=np.float32).reshape((-1,3))
    anomaly_labels = np.fromfile(path_labels, dtype=np.int16).reshape((-1))
    #clusters_3 =  np.fromfile(path_clustering_3, dtype=np.int16).reshape((-1)) #self:dyn and sup:static
    #clusters_4 =  np.fromfile(path_clustering_4, dtype=np.int16).reshape((-1)) #self:static and sup:dyn
    original_labels = np.fromfile(path_original_label, dtype=np.float32).reshape((-1,4))
    zero_column = np.zeros((1, original_labels.shape[0]))    
    original_labels = np.insert(original_labels, 3, zero_column, axis=1)
    
    zero_column = np.zeros((1, pc_velo_camerafov.shape[0])) 
    new_labels = np.insert(pc_velo_camerafov, 3, anomaly_labels, axis=1)   
    new_labels = np.insert(new_labels, 4, zero_column, axis=1)
    
    #points = []
    #labels = []
    #for o_point in original_labels:
    #    for n_point in new_labels:
    #        if o_point[0] + 0.001 >= n_point[0] and o_point[0] - 0.001 <= n_point[0] and o_point[1] + 0.001 >= n_point[1] and o_point[1] - 0.001 <= n_point[1] and o_point[2] + 0.001 >= n_point[2] and o_point[2] - 0.001 <= n_point[2]:
    #            points.append(o_point[:3])
    #            if o_point[4] == -1:
    #                if n_point[3] == 1 or n_point[3] == 2:
    #                    labels.append(1)
    #                elif n_point[3] == 3 or n_point[3] == 4:
    #                    labels.append(3)
    #            elif o_point[4] == 1:
    #                if n_point[3] == 1 or n_point[3] == 2:
    #                    labels.append(4)
    #                elif n_point[3] == 3 or n_point[3] == 4:
    #                    labels.append(2)
    #            break
    #    else:
    #        points.append(o_point[:3])
    #        if o_point[4] == -1:
    #            labels.append(5)
    #        elif o_point[4] == 1:
    #            labels.append(6)
    #
    #points = np.array(points).reshape((-1, 3)) 
    #labels = np.array(labels).reshape((-1, 1))
   
    
    
    points = []
    labels = []
    used_points = [False] * len(new_labels)
    for o_point in original_labels:
        for index, n_point in enumerate(new_labels):
            if o_point[0] + 0.001 >= n_point[0] and o_point[0] - 0.001 <= n_point[0] and o_point[1] + 0.001 >= n_point[1] and o_point[1] - 0.001 <= n_point[1] and o_point[2] + 0.001 >= n_point[2] and o_point[2] - 0.001 <= n_point[2]:
                points.append(o_point[:3])
                used_points[index] = True
                if o_point[4] == -1:
                    if n_point[3] == 1:
                        labels.append(-11)
                    elif n_point[3] == 2:
                        labels.append(-12)
                    elif n_point[3] == 3:
                        labels.append(-13)
                    elif n_point[3] == 4:
                        labels.append(-14)
                elif o_point[4] == 1:
                    if n_point[3] == 1:
                        labels.append(11)
                    elif n_point[3] == 2:
                        labels.append(12)
                    elif n_point[3] == 3:
                        labels.append(13)
                    elif n_point[3] == 4:
                        labels.append(14)
                break
        else:
            points.append(o_point[:3])
            if o_point[4] == -1:
                labels.append(-10)
            elif o_point[4] == 1:
                labels.append(10)
    
    unused_points = used_points
    
    for index, point in enumerate(new_labels):
        if not used_points[index]:
            points.append(point[:3])
            used_points[index] = True
            if point[3] == 1:
                labels.append(1)
            elif point[3] == 2:
                labels.append(2)
            elif point[3] == 3:
                labels.append(3)
            elif point[3] == 4:
                labels.append(4)
    
    
    points = np.array(points).reshape((-1, 3)) 
    labels = np.array(labels).reshape((-1, 1))
    
    c_11 = np.count_nonzero(labels == -11)
    c_12 = np.count_nonzero(labels == -12)
    c_13 = np.count_nonzero(labels == -13)
    c_14 = np.count_nonzero(labels == -14)
    c11 = np.count_nonzero(labels == 11)
    c12 = np.count_nonzero(labels == 12)
    c13 = np.count_nonzero(labels == 13)
    c14 = np.count_nonzero(labels == 14)
    c_10 = np.count_nonzero(labels == -10)
    c10 = np.count_nonzero(labels == 10)
    c1 = np.count_nonzero(labels == 1)
    c2 = np.count_nonzero(labels == 2)   
    c3 = np.count_nonzero(labels == 3)  
    c4 = np.count_nonzero(labels == 4)
    tp = c13 + c14
    fp = c_13 + c_14 + c3 + c4
    fn = c11 + c12 + c10
    tn = c_11 + c_12 + c_10 + c1 + c2
    total = labels.shape[0]
    original_total = original_labels.shape[0]
    new_label_total = new_labels.shape[0]
    
    if not os.path.exists(os.path.join(path_save, sequence)):
        os.makedirs(os.path.join(path_save, sequence))
    with open(path_save + '/counts.txt', 'a') as f:
        f.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (c_11, c_12, c_13, c_14, c11, c12, c13, c14, c_10, c10, c1, c2, c3, c4, tp, fp, fn, tn, total, original_total, new_label_total))
    points.astype('float32').tofile(os.path.join(path_save, sequence, sequence + '_points.bin'))
    labels.astype('float32').tofile(os.path.join(path_save, sequence, sequence + '_labels.bin'))
        
    if points.shape[0] == 0:
        return
    
    #print(points.shape[0] / original_labels.shape[0])
    
    
    #points = np.vstack((original_labels[:, :3], new_labels[:, :3]))
    #labels = np.hstack((np.full((original_labels.shape[0]), fill_value = -1, dtype=np.int16), new_labels[:,3]))
    #labels = labels.astype(int)
    #print(original_labels[0])
    #print(new_labels[0])
    
    #{-1:[0,0,0], 1:[34,139,34], 2:[65,105,225], 3:[178,34,34], 4:[225,225,0]}
    color_dict = {-1:[0,0,0], 13:[34,139,34], 14:[34,139,34], -13:[178,34,34], -14:[178,34,34], 3:[178,34,34], 4:[178,34,34], 11:[225,225,0], 12:[225,225,0], 10:[225,225,0], -11:[65,105,225], -12:[65,105,225], -10:[65,105,225], 1:[65,105,225], 2:[65,105,225]}
    sem_label_color = get_labels_and_colors(labels, color_dict)
    
    rgb = cv2.cvtColor(cv2.imread(os.path.join(path_img)), cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = rgb.shape

    calib_dic = {}
    calib = read_calib_file(path_calibrations, calib_dic)

    img = get_point_camerafov(points, calib, rgb, img_width, img_height, sem_label_color, dataset)
    
    if points.shape[0] != 0:
        save_img_to_file(img, path_save, sequence)

def main(seq):
    seq = '{0:04d}'.format(int(seq))
    # load config file
    config_filename = '/disk/no_backup/ju878/model_contradictions/supervised_unsupervised_anomaly/anomaly_detection/config/config_paths.yaml'
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    
    path_dataset = config['path_dataset']
    path_inference = config['path_inference']
    path_anomalies_sup_self = os.path.join(path_inference, 'anomalies_sup_self')
    path_original_labels = os.path.join(path_inference, 'original_labels')
    path_groundtruth_eval = os.path.join(path_inference, 'groundtruth_eval_all')

    frames = os.listdir(os.path.join(path_anomalies_sup_self, seq, 'anomaly_labels'))
    frames = sorted(frames, key=lambda x: int(re.findall(r'\d+', x)[-1]))
    
    once = r'^\d{6}_\d{13}_\d{1,2}\.bin$'
    
    
    for frame in range(len(frames)):
        frame_label = frames[frame]
        if int(frame_label.split('.')[0].split('_')[2]) != 8:
            continue
        dataset = ''
        if re.match(once, frame_label):
            dataset = 'once'
        elif 'kitti' in frame_label:
            dataset = 'kitti'
        elif 'nuscenes' in frame_label:
            dataset = 'nuscenes'
        frame_image = frames[frame].split('.')[0] +'.png'
        frame_clusters_incon_static_dyn_3 = frames[frame].split('.')[0] +'_cluster_incon_static_dyn_3.bin'
        frame_clusters_incon_dyn_static_4 = frames[frame].split('.')[0] +'_cluster_incon_dyn_static_4.bin'
        
        path_calib_frame = os.path.join(path_dataset ,seq, 'calib.txt')
        path_pc_velo_camerafov_frame = os.path.join(path_anomalies_sup_self, seq, 'pc_velo_camerafov', frame_label)
        path_clusters_frame_3 = os.path.join(path_anomalies_sup_self, seq, 'clusters', frame_clusters_incon_static_dyn_3)
        path_clusters_frame_4 = os.path.join(path_anomalies_sup_self, seq, 'clusters', frame_clusters_incon_dyn_static_4)
        path_anomaly_labels_frame = os.path.join(path_anomalies_sup_self, seq, 'anomaly_labels', frame_label)
        path_image_frame = os.path.join(path_dataset, seq, 'image_2', frame_image)
        path_original_label_frame = os.path.join(path_original_labels, seq + '.bin')

        map_anomalies_on_image(path_pc_velo_camerafov_frame, path_anomaly_labels_frame, path_image_frame, path_clusters_frame_3, path_clusters_frame_4, path_original_label_frame, path_calib_frame, path_groundtruth_eval, seq, frame_image, dataset)
        
if __name__ == '__main__':
    # load config file
    config_filename = '/disk/no_backup/ju878/model_contradictions/supervised_unsupervised_anomaly/anomaly_detection/config/config_paths.yaml'
    config = yaml.load(open(config_filename), Loader=yaml.FullLoader)
    
    sequences = config['sequences']
    #with Pool(12) as p:
    #    p.map(main, sequences)
    for seq in tqdm(sequences):
        main(seq)