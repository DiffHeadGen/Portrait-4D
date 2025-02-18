# Inference script for Portrait4D and Portrait4D-v2

"""Generate images and shapes using pretrained network pickle."""
from functools import cached_property
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import cv2
import imageio
import json

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.reconstructor.triplane_reconstruct import TriPlaneReconstructorNeutralize
from training.utils.preprocess import estimate_norm_torch_pdfgc
from models.pdfgc.encoder import FanEncoder
from kornia.geometry import warp_affine
import torch.nn.functional as F
import pickle
from pytorch3d.io import load_obj, save_obj
from shape_utils import convert_sdf_samples_to_ply
from expdataloader import *
from expdataloader.P4DLoader import P4DRowData

# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    parts = s.split(",")
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f"cannot parse 2-vector {s}")


# ----------------------------------------------------------------------------


def get_motion_feature(pd_fgc, imgs, lmks, crop_size=224, crop_len=16, reverse_y=False):

    trans_m = estimate_norm_torch_pdfgc(lmks, imgs.shape[-1], reverse_y=reverse_y)
    imgs_warp = warp_affine(imgs, trans_m, dsize=(224, 224))
    imgs_warp = imgs_warp[:, :, : crop_size - crop_len * 2, crop_len : crop_size - crop_len]
    imgs_warp = torch.clamp(F.interpolate(imgs_warp, size=[crop_size, crop_size], mode="bilinear"), -1, 1)

    out = pd_fgc(imgs_warp)
    motions = torch.cat([out[1], out[2], out[3]], dim=-1)

    return motions


# ----------------------------------------------------------------------------


def pose2rot(pose):
    rot_mats = batch_rodrigues(pose.view(-1, 3), dtype=pose.dtype).view([pose.shape[0], 3, 3])

    return rot_mats


# ----------------------------------------------------------------------------


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    return samples.unsqueeze(0), voxel_origin, voxel_size


# ----------------------------------------------------------------------------

class ImageGenerator:
    use_simplified = True
    def __init__(self):
        self.device = torch.device("cuda")
        
        torch.manual_seed(42)

    def load_model(self, model_path="./pretrained_models/portrait4d-v2-vfhq512.pkl", reload_modules=True):
        print(f'Loading networks from "{model_path}"...')
        device = torch.device("cuda")
        with dnnlib.util.open_url(model_path) as f:
            Network = legacy.load_network_pkl(f)  # type: ignore
            G = Network["G_ema"].to(device)

        # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
        if reload_modules:
            print("Reloading Modules!")
            G_new = TriPlaneReconstructorNeutralize(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
            misc.copy_params_and_buffers(G, G_new, require_all=False)
            G_new.neural_rendering_resolution = G.neural_rendering_resolution
            G_new.rendering_kwargs = G.rendering_kwargs
            G = G_new

        # load motion encoder
        pd_fgc = FanEncoder()
        weight_dict = torch.load("models/pdfgc/weights/motion_model.pth")
        pd_fgc.load_state_dict(weight_dict, strict=False)
        pd_fgc = pd_fgc.eval().to(device)
        self.pd_fgc = pd_fgc
        self.G = G

    def generate_image(self, row: P4DRowData, use_neck=True, shape=False):
        if not hasattr(self, "G"):
            self.load_model()
        G = self.G
        img_app, lmks_app, motion_app, params_app, shape_params_app, _, _, _ = self.load_params(row.source_output.crop_dir, row.source_img_path)

        for tar_idx, target_img_path in tqdm(enumerate(row.target.img_paths), desc="Generating images"):
            target_name = os.path.splitext(os.path.basename(target_img_path))[0]
            # print(target_img_path)
            img_mot, lmks_mot, motion_mot, params_mot, shape_params_mot, exp_params_mot, pose_params_mot, eye_pose_mot = self.load_params(row.output.crop_dir, target_img_path)
            c = params_mot[:, :25]
            intrinsics = c[:, 16:]

            if use_neck:
                extrinsics = torch.eye(4).to(self.device)
                extrinsics[1, 3] = 0.01
                extrinsics[2, 3] = 4.2
                c = torch.cat([extrinsics.reshape(1, -1), intrinsics.reshape(1, -1)], dim=-1)
            else:
                pose_params_mot[:, :3] *= 0

            _deformer = G._deformer(shape_params_app, exp_params_mot, pose_params_mot, eye_pose_mot, use_rotation_limits=False, smooth_th=3e-3)
            out = G.synthesis(img_app, img_mot, motion_app, motion_mot, c, _deformer=_deformer, neural_rendering_resolution=128, motion_scale=1)

            img = out["image_sr"]

            img_ = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_mot_ = (img_mot.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img_app_ = (img_app.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            img = torch.cat([img_app_, img_, img_mot_], dim=2)

            PIL.Image.fromarray(img_[0].cpu().numpy(), "RGB").save(f"{row.output.ori_output_dir}/{target_name}.jpg", quality=95)
            PIL.Image.fromarray(img[0].cpu().numpy(), "RGB").save(f"{row.output.ori_output_comp_dir}/{target_name}.jpg", quality=95)

            if shape:
                os.makedirs(f"{row.output_dir}/shapes", exist_ok=True)

                max_batch = 1000000
                shape_res = 128  # Marching cube resolution

                samples, voxel_origin, voxel_size = create_samples(
                    N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs["box_warp"] * 1
                )  # .reshape(1, -1, 3)
                samples = samples.to(c.device)
                sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=c.device)

                head = 0
                with tqdm(total=samples.shape[1]) as pbar:
                    with torch.no_grad():
                        while head < samples.shape[1]:
                            torch.manual_seed(0)
                            out = G.sample_mixed(
                                img_app,
                                img_mot,
                                motion_app,
                                motion_mot,
                                samples[:, head : head + max_batch],
                                torch.zeros_like(samples[:, head : head + max_batch]),
                                shape_params_app,
                                exp_params_mot,
                                pose_params_mot,
                                eye_pose_mot,
                            )
                            sigma = out["sigma"]
                            sigmas[:, head : head + max_batch] = sigma
                            head += max_batch
                            pbar.update(max_batch)

                sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
                sigmas = np.flip(sigmas, 0)

                convert_sdf_samples_to_ply(
                    np.transpose(sigmas, (2, 1, 0)), voxel_origin, voxel_size, f"{row.output_dir}/shapes/{tar_idx:05d}.ply", level=25
                )

    def load_params(self, base_dir, img_name):
        img_name = os.path.basename(img_name)
        img = np.array(PIL.Image.open(os.path.join(base_dir, "align_images", img_name)))
        data_name = os.path.splitext(img_name)[0]
        img = torch.from_numpy((img.astype(np.float32) / 127.5 - 1)).to(self.device)
        img = img.permute([2, 0, 1]).unsqueeze(0)
        # source landmarks: y axis points downwards
        lmks = np.load(os.path.join(base_dir, "3dldmks_align", data_name + ".npy"))
        lmks = torch.from_numpy(lmks).to(self.device).unsqueeze(0)

        # calculate motion embedding
        motion = get_motion_feature(self.pd_fgc, img, lmks)

        # source flame params
        if self.use_simplified:
            params = np.load(os.path.join(base_dir, "bfm2flame_params_simplified", data_name + ".npy"))
        else:
            params = np.load(os.path.join(base_dir, "flame_optim_params", data_name + ".npy"))
        params = torch.from_numpy(params).to(self.device).reshape(1, -1)

        shape_params = params[:, 25:325]
        exp_params = params[:, 325:425]
        pose_params = params[:, 425:431]
        eye_pose = params[:, 431:437]
        return img, lmks, motion, params, shape_params, exp_params, pose_params, eye_pose
