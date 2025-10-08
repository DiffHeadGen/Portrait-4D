import time
from preprocess_exp import P4DRowData, CropLoader

loader = CropLoader()
for row in loader.all_data_rows:
    print(row.source_img_path)
# loader.print_summary()