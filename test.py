import pickle
import numpy as np

data = {
    "crop_params": {
        "crop_size": (256, 256),
        "crop_center": (128, 128),
    }
}
pickle.dump(data, open("test_data/crop/crop_params/000000.pkl", "wb"))
data = pickle.load(open("test_data/crop/crop_params/000000.pkl", "rb"))
print(data)
