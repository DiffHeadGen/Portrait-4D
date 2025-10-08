from functools import cached_property
import os
import pickle
import shutil
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

##### self defined package
from lib.config.config import cfg
from lib.inferencer import Tester as Tester

from expdataloader import *
from expdataloader.P4DLoader import *

from PIL import Image
from bfm2flame_simplified import bfm2flame_simplified



class PreProcessOutput:
    def __init__(self, data_name, base_dir = "./output"):
        self.base_dir = base_dir
        self.data_name = data_name

    @cached_property
    def output_dir(self):
        return get_sub_dir(self.base_dir, self.data_name)

    @cached_property
    def crop_dir(self):
        return get_sub_dir(self.output_dir, "crop")

    @cached_property
    def align_images_dir(self):
        return get_sub_dir(self.crop_dir, "align_images")

    @cached_property
    def align_image_paths(self):
        return get_image_paths(self.align_images_dir)

    @cached_property
    def cropped_imgs_dir(self):
        return os.path.join(self.crop_dir, "align_images")

    @property
    def num_cropped_imgs(self):
        return count_images(self.cropped_imgs_dir)

    @property
    def cropped_img_paths(self):
        # This is dynamic, cannot use cached_property
        return get_image_paths(self.cropped_imgs_dir)

    @cached_property
    def bfm2flame_params_dir(self):
        return get_sub_dir(self.crop_dir, "bfm2flame_params_simplified")

    @property
    def num_bfm2flame_params(self):
        return count_files(self.bfm2flame_params_dir)
    

class CropLoader(P4DLoader):
    @cached_property
    def model(self):
        return Tester(cfg)

    def run_video(self, row: P4DRowData):
        self.run_source(row)
        if not row.is_img_aligned:
            self.model.inference(row.target.imgs_dir, row.output.crop_dir, video=True)
        if not row.is_bfm2flame_params_ready:
            bfm2flame_simplified(row.output.crop_dir)

    def run_source(self, row: P4DRowData):
        input_dir = get_sub_dir(row.source_output.output_dir, "ori_imgs")
        shutil.copy(row.source_img_path, input_dir)
        self.model.inference(input_dir, row.source_output.crop_dir, video=True)
        bfm2flame_simplified(row.source_output.crop_dir)


def test_202050218():
    loader = CropLoader()
    row = loader.test_20250218_row_data
    row.output = row.source_output
    row.target = row.source
    # print(row.output.crop_dir)
    loader.run_video(row)
    # loader.run_video(TEST_ROW_DATA_ID1)


def main():
    loader = CropLoader()
    loader.run_all()


def bfm2flame():
    loader = CropLoader()
    row: P4DRowData
    for row in tqdm(loader.all_data_rows, desc="bfm2flame"):
        if row.is_bfm2flame_params_ready:
            print(f"Skip {row.data_name}")
            continue
        bfm2flame_simplified(row.output_crop_dir)


if __name__ == "__main__":
    main()
