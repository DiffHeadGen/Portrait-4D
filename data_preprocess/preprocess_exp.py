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


class CropLoader(P4DLoader):
    @cached_property
    def model(self):
        return Tester(cfg)

    def run_video(self, row: P4DRowData):
        if not row.is_img_aligned:
            self.model.inference(row.target.imgs_dir, row.output.crop_dir, video=True)
        if not row.is_bfm2flame_params_ready:
            bfm2flame_simplified(row.output.crop_dir)


def main():
    loader = CropLoader()
    row = loader.test_20250218_row_data
    row.output = row.source_output
    row.target = row.source
    # print(row.output.crop_dir)
    loader.run_video(row)
    # loader.run_video(TEST_ROW_DATA_ID1)


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
