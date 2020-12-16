import random
import glob
import os
import shutil

dataset_root = "/root/TrainingDataset/TrainingData/"
original_folder = "Images"
parts = {
    "train": {"portion": 0.7, "folder": "ImagesTrain"},
    "val": {"portion": 0.1, "folder": "ImagesVal"},
    "test": {"portion": 0.2, "folder": "ImagesTest"},
}

images = glob.glob(os.path.join(dataset_root, original_folder, "*.png"))
image_num, cursor = len(images), 0
print(f"Total images: {image_num}")
random.shuffle(images)  # shuffle all images before division
for name, config in parts.items():
    cnt = int(config["portion"] * image_num)
    print(f"Images for {name}: {cnt}")
    folder = os.path.join(dataset_root, config["folder"])
    if not os.path.exists(folder) or not os.path.isdir(folder):
        os.mkdir(folder)
    for _ in range(cnt):
        shutil.copy(images[cursor], folder)
        cursor += 1
    print("copied.")
