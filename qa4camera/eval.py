import models
import os
import torch.nn as nn
import torch
import time
import cv2
import csv
import numpy as np
from collections import Counter, defaultdict

is_cuda = True


def compute_SRCC(ranking1, ranking2):
    srcc = [0, 0, 0, 0]
    for k in ranking1.keys():
        for i in range(4):
            diff = ranking1[k][i] - ranking2[k][i]
            srcc[i] += diff * diff
    N = len(ranking1.keys())
    for i in range(4):
        srcc[i] = 1 - 6 * srcc[i] / N / (N * N - 1)
    return srcc


def eval_scene(model, scene_folder, crop_size=(500, 500), sort_result=None):
    image_names = os.listdir(scene_folder)
    assert (
        len(image_names) == 15
    ), f"number of images in {scene_folder} is {len(image_names)}"
    images = dict()
    for image_name in image_names:
        image_path = os.path.join(scene_folder, image_name)
        image = cv2.imread(image_path)
        # crop the center region
        h, w = image.shape[:2]
        h = (h - crop_size[0]) // 2
        w = (w - crop_size[1]) // 2
        image = image[h : h + crop_size[0], w : w + crop_size[1], :].copy()
        assert image.shape[:2] == crop_size
        image = np.transpose(image, (2, 0, 1))
        images[image_name] = image
    model = model.eval()
    votes = [Counter(), Counter(), Counter(), Counter()]
    with torch.no_grad():
        for n1 in image_names:
            x = images[n1]
            for n2 in image_names:
                if n1 == n2:
                    continue
                y = images[n2]
                inputs = torch.from_numpy(
                    np.expand_dims(np.concatenate((x, y)), axis=0)
                ).float()
                if is_cuda:
                    inputs = inputs.cuda()
                outputs = model(inputs).cpu().numpy()
                outputs = np.reshape(outputs, (4,))
                # print(n1, n2, outputs)
                for i in range(4):
                    votes[i][n1] += int(outputs[i] > 0.5)
    result = defaultdict(list)
    for v in votes:
        rank = 0
        for image, _ in v.most_common():
            rank += 1
            result[image].append(rank)
    if sort_result is None:
        return result
    ground_truth = dict()
    with open(sort_result) as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            ground_truth[row[0]] = list(map(float, row[1:]))
    srcc = compute_SRCC(result, ground_truth)
    return result, srcc


if __name__ == "__main__":
    is_cuda = True
    model = torch.load("./model/resnet50-test-1e-5.pth").cuda()
    scenes = range(81, 101)
    all_srcc = []
    for s in scenes:
        print(f"--- Scene {s:03d} ---")
        result, srcc = eval_scene(
            model,
            f"/root/dataset/Training/{s:03d}/",
            (500, 500),
            f"/root/dataset/score_and_sort/Training/sort/{s:03d}.csv",
        )
        # print(result)
        print(f"SRCC: {srcc}")
        all_srcc.append(srcc)
    print(f"Overall SRCC: {np.mean(all_srcc, axis=0)}")
