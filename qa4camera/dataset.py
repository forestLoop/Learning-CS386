import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageDataset(Dataset):

    def __init__(self, dataset_root, scenes, crop_size=(500, 500)):
        self.images = dict()
        self.data = []
        print("Loading dataset...")
        sort_result_dir = os.path.join(
            dataset_root, "score_and_sort", "Training", "sort")
        scenes_dir = os.path.join(dataset_root, "Training")
        for scene_id in scenes:
            print(f"Scene {scene_id:03d}")
            # rankings[image_name] = (Color, Exposure, Noise, Texture)
            rankings = dict()
            with open(os.path.join(sort_result_dir, "{:03d}.csv".format(scene_id))) as f:
                reader = csv.reader(f)
                next(reader)  # skip header row
                for row in reader:
                    rankings[row[0]] = list(map(float, row[1:]))
            image_folder = os.path.join(scenes_dir, "{:03d}".format(scene_id))
            for image_name in rankings.keys():
                image_path = os.path.join(image_folder, image_name)
                image = cv2.imread(image_path)
                # crop the center region
                h, w = image.shape[:2]
                h = (h - crop_size[0]) // 2
                w = (w - crop_size[1]) // 2
                image = image[h:h + crop_size[0], w:w + crop_size[1], :]
                assert image.shape[:2] == crop_size
                # (height, weight, channel) -> (channel, weight, height)
                image = np.transpose(image, (2, 0, 1))
                self.images[image_name] = image
                r1 = np.array(rankings[image_name])
                for another_image_name in rankings.keys():
                    if image_name == another_image_name:
                        continue
                    r2 = np.array(rankings[another_image_name])
                    self.data.append(
                        (image_name, another_image_name, (r1 < r2) * 1)
                    )
        print(
            f"Dataset loadeded: {len(self.images)} images and {len(self.data)} pairs.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        item = self.data[idx]
        image1 = self.images[item[0]]
        image2 = self.images[item[1]]
        image = np.concatenate((image1, image2))
        assert image.shape[0] == 6
        labels = item[2]
        return image, labels


if __name__ == "__main__":
    print("Note: this file is not expected to be run directly!")
    print("Test ImageDataset:")
    d = ImageDataset("C:\\MyCode\\DIP\\DIP1\\dataset", tuple(range(1, 10)))
    print(len(d))
    for i in range(100):
        x = d[i]
        print(x[0].shape, x[1])
