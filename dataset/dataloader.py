import os
import numpy as np
import logging

from dataset.data_util import *
logger = logging.getLogger(__package__)

def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def parse_llff_pose(pose):
    """
    convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)
    Args:
        pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4]
    """
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    c2w_4x4[:, 1:3] *= -1
    intrinsics = np.array([[f, 0, w / 2.0, 0], [0, f, h / 2.0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return intrinsics, c2w_4x4


def batch_parse_llff_poses(poses):
    all_intrinsics = []
    all_c2w_mats = []
    for pose in poses:
        intrinsics, c2w_mat = parse_llff_pose(pose)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats

def recenter_poses(poses):
    poses_ = poses + 0
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:, :3, :4] = poses[:, :3, :4]
    poses = poses_
    return poses

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

# Resize all images of the scene to the same size
image_size_table ={
    "tat_intermediate_Playground": [548, 1008],
    "tat_intermediate_Family": [1084, 1957],
    "tat_intermediate_Francis": [1086, 1959],
    "tat_intermediate_Horse": [1084, 1958],
    "tat_training_Truck": [546, 980]
}

def load_style_meta_data():
    import pickle
    with open('/mnt/hdd/data/wikiart/wikiart_data.pickle', 'rb') as f:
        data = pickle.load(f)
        
    return data['train_data'], data['val_data']

def load_data_split(basedir, scene, split, try_load_min_depth=True, only_img_files=False, seed=None):
    """
    :param split train | validation | test
    """
    scenes = sorted(os.listdir(basedir))
    all_ray_samplers = []
    
    scene_dir = os.path.join(basedir, scene, split)
    
    # camera parameters files
    intrinsics_files = find_files(os.path.join(scene_dir, "intrinsics"), exts=['*.txt'])
    pose_files = find_files(os.path.join(scene_dir, "pose"), exts=['*.txt'])
    img_files = find_files(os.path.join(scene_dir, "rgb"), exts=['*.png', '*.jpg'])
    
    logger.info('raw intrinsics_files: {}'.format(len(intrinsics_files)))
    logger.info('raw pose_files: {}'.format(len(pose_files)))
    logger.info('raw img_files: {}'.format(len(img_files)))

    cam_cnt = len(pose_files)
    logger.info("Dataset len is {}".format(cam_cnt))
    assert(len(img_files) == cam_cnt)

    # img files
    style_dir = os.path.join("./wikiart", split)
    style_img_files = find_files(style_dir, exts=['*.png', '*.jpg'])
    logger.info("Number of style images is {}".format(len(style_img_files)))
    
    # create ray samplers
    ray_samplers = []
    H, W = image_size_table[scene]
    
    if seed != None:
        np.random.seed(seed)

    for i in range(cam_cnt):
        intrinsics = parse_txt(intrinsics_files[i])
        pose = parse_txt(pose_files[i])

        ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsics, c2w=pose,
                                                img_path = img_files[i],
                                                mask_path=None,
                                                min_depth_path=None,
                                                max_depth=None,
                                                style_imgs = style_img_files
                                                ))
        
    logger.info('Split {}, # views: {}'.format(split, cam_cnt))
    
    return ray_samplers


def load_poses(scene):
    a = np.load(f'/home/meric/GNT_Style/data/nerf_llff_data/{scene}/poses_bounds.npy')
    
    poses_arr = a
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
    bds = poses_arr[:, -2:].transpose([1, 0])

    sfx = ""
    factor = 8
    img0 = [
            os.path.join(f'/home/meric/GNT_Style/data/nerf_llff_data/{scene}/images', f)
            for f in sorted(os.listdir(f'/home/meric/GNT_Style/data/nerf_llff_data/{scene}/images'))
            if f.endswith("JPG") or f.endswith("jpg") or f.endswith("png")
        ][0]
    sh = imageio.imread(img0).shape

    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1.0 / factor

    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    bd_factor=0.75
    # Rescale if bd_factor is provided
    sc = 1.0 if bd_factor is None else 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    poses = recenter_poses(poses)

    c2w = poses_avg(poses)
    # print('Data:')
    # print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)
    i_test = np.argmin(dists)
    # print('HOLDOUT view is', i_test)
    poses = poses.astype(np.float32)
    intrinsics, c2w_mats = batch_parse_llff_poses(poses)
    return intrinsics, c2w_mats

def load_data_split_custom(scene, try_load_min_depth=True, only_img_files=False, seed=None):
    """
    :param split train | validation | test
    """
    
    if seed != None:
        np.random.seed(seed)
        
    scene_dir = os.path.join('/home/meric/GNT_Style/data/nerf_llff_data', scene)
    
    img_files = find_files(os.path.join(scene_dir, "images_8"), exts=['*.png', '*.jpg'])

    # img files
    #style_dir = os.path.join("./wikiart", split)
    #style_img_files = find_files(style_dir, exts=['*.png', '*.jpg'])
    #logger.info("Number of style images is {}".format(len(style_img_files)))
    
    # create ray samplers
    train_ray_samplers = []
    val_ray_samplers = []
    H, W = 378, 504
    
    if seed != None:
        np.random.seed(seed)

    intrinsics, poses = load_poses(scene)
    
    i_test = list(np.arange(poses.shape[0])[::8])
    i_train = [
                j
                for j in np.arange(int(poses.shape[0]))
                if (j not in i_test and j not in i_test)
            ]
    
    intrinsics[:, :2, :3] *= 1/8
    
    train_styles, val_styles = load_style_meta_data()
    
    for i in range(poses.shape[0]):
        intrinsic = intrinsics[i]
        pose = poses[i]
        
        if i in i_train:
            train_ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsic, c2w=pose,
                                                    img_path = img_files[i],
                                                    mask_path=None,
                                                    min_depth_path=None,
                                                    max_depth=None,
                                                    style_imgs = train_styles
                                                    ))
        else:
            val_ray_samplers.append(RaySamplerSingleImage(H=H, W=W, intrinsics=intrinsic, c2w=pose,
                                                    img_path = img_files[i],
                                                    mask_path=None,
                                                    min_depth_path=None,
                                                    max_depth=None,
                                                    style_imgs = val_styles
                                                    ))
    
    return train_ray_samplers, val_ray_samplers