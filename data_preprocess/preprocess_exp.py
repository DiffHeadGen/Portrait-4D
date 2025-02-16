from functools import cached_property
import os
import pickle
import cv2
import numpy as np
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

##### self defined package
from lib.config.config import cfg
from lib.inferencer import Tester as Tester

from expdataloader import *
from cropping.crop_images_portrait_model import retarget_img
from PIL import Image


class P4DRowData(RowData):
    @property
    def is_processed(self):
        return len(self.ori_img_paths) == len(self.croped_img_paths)

    @cached_property
    def output_crop_dir(self):
        return get_sub_dir(self.output_dir, "crop")

    @cached_property
    def croped_img_paths(self):
        return get_image_paths(self.croped_imgs_dir)

    @cached_property
    def croped_imgs_dir(self):
        return os.path.join(self.output_crop_dir, "align_images")

    @cached_property
    def crop_params_dir(self):
        return os.path.join(self.output_crop_dir, "crop_params")

    @cached_property
    def retarget_imgs_dir(self):
        return get_sub_dir(self.output_dir, "retarget_imgs")

    def merge_croped_frames(self):
        merge_video(f"{self.output_crop_dir}/align_images/%6d.png", f"{self.output_crop_dir}/crop.mp4")

    def retarget_test(self):
        self.retarget_imgs(self.croped_imgs_dir, self.retarget_imgs_dir)
    
    def retarget_imgs(self, croped_imgs_dir, output_dir):
        for ori_img_path, croped_img_path in tqdm(zip(self.ori_img_paths, get_image_paths(croped_imgs_dir)), total=len(self.ori_img_paths)):
            name = os.path.basename(ori_img_path)
            crop_params = pickle.load(open(os.path.join(self.crop_params_dir, change_extension(name, ".pkl")), "rb"))
            result = retarget_img(Image.open(ori_img_path), Image.open(croped_img_path), crop_params)
            output_path = os.path.join(output_dir, change_extension(name, ".jpg"))
            result.save(output_path)


class CropLoader(HeadGenLoader):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name)

    @cached_property
    def model(self):
        return Tester(cfg)

    def create_row(self, data_name) -> P4DRowData:
        return P4DRowData(data_name, self.output_dir)

    def run_video(self, row: P4DRowData):
        self.model.inference(row.ori_imgs_dir, row.output_crop_dir, video=True)


def main():
    loader = CropLoader()
    # for row in loader.all_data_rows:
        # loader.merge_croped_frames(row)
        # print(row.is_processed)
    loader.run_all()


def retarget_test():
    row = P4DRowData(data_name="id1", base_output_dir="../test_data", base_dir="../test_data")
    if not row.is_processed:
        model = Tester(cfg)
        model.inference(row.ori_imgs_dir, row.output_crop_dir, video=True)
    row.retarget_test()


if __name__ == "__main__":
    main()
