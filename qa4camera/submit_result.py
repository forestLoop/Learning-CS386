from eval import eval_scene
import torch
import os
import csv

if __name__ == "__main__":
    while True:
        try:
            model_path = input("model path:") or "./model/resnet101-full-1e-5.pth"
            print(f"Use model: {model_path}")
            model = torch.load(model_path)
            break
        except Exception as e:
            print("failed to load model:", e)
    if torch.cuda.is_available():
        print("Use CUDA.")
        model = model.cuda()
    test_folder = input("folder for testing scenes:") or "/root/dataset/Test"
    output_folder = input("folder for testing results:") or "./submission/"
    for scene in range(1, 20 + 1):
        scene_folder = os.path.join(test_folder, f"{scene:03d}")
        output_file = os.path.join(output_folder, f"{scene:03d}.csv")
        result = eval_scene(model, scene_folder)
        with open(output_file, mode="w") as f:
            writer = csv.writer(f)
            writer.writerow([" ", "Color", "Exposure", "Noise", "Texture"])
            for k in sorted(result.keys()):
                writer.writerow(
                    [
                        k,
                    ]
                    + result[k]
                )
