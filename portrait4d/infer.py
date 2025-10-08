from functools import cached_property
import os
import shutil
from gen_images_portrait4d import ImageGenerator
from expdataloader import *
from expdataloader.P4DLoader import P4DLoader


class GenLoader(P4DLoader):
    def __init__(self, name="Protrait4Dv2"):
        super().__init__(name)

    @cached_property
    def model(self):
        return ImageGenerator()

    def run_video(self, row):
        if not row.is_img_generated:
            self.model.generate_image(row)
            row.output.merge_ori_output_video()
            row.output.merge_ori_output_comp_video()
            self.retarget_video(row)

    def get_run_data_rows(self):
        for row in self.all_data_rows:
            if not row.is_img_generated:
                yield row


def clear_all_output():
    loader = GenLoader()
    for row in loader.all_data_rows:
        print(row)
        row.output.clear_ori_output()


def test_retarget():
    retargeter = Retargeter(use_cache=True)
    source_img_path = "source.jpg"
    target_img_path = "target.jpg"
    output_path = "output.jpg"
    retargeter.retarget(source_img_path, target_img_path, output_path)


def main():
    loader = GenLoader()
    loader.run_all()
    # for row in loader.all_data_rows:
    #     print(row.source_output.crop_dir, row.source_img_path)
        


if __name__ == "__main__":
    main()

