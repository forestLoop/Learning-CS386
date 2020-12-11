import torch
import torch.nn as nn
import torchvision
import numpy as np


class Resnet101(nn.Module):
    def __init__(self):
        super(Resnet101, self).__init__()
        self.pretrained_resnet = torchvision.models.resnet101(pretrained=True)
        # set input channels to 6 as we stack up two images
        self.pretrained_resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # set output dimension to 4
        self.pretrained_resnet.fc = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.pretrained_resnet(x)
        # add sigmoid function before output
        x = torch.sigmoid(x)
        return x


class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.pretrained_resnet = torchvision.models.resnet50(pretrained=True)
        # set input channels to 6 as we stack up two images
        self.pretrained_resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # set output dimension to 4
        self.pretrained_resnet.fc = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.pretrained_resnet(x)
        # add sigmoid function before output
        x = torch.sigmoid(x)
        return x


class Resnet50Extended(nn.Module):
    def __init__(self):
        super(Resnet50Extended, self).__init__()
        self.pretrained_resnet = torchvision.models.resnet50(pretrained=True)
        # set input channels to 6 as we stack up two images
        self.pretrained_resnet.conv1 = nn.Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.pretrained_resnet.fc = nn.Linear(2048, 2048)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 4)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.pretrained_resnet(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sg(x)
        return x


if __name__ == "__main__":
    import time

    print("Note: this file is not expected to be run directly!")
    enable_cuda = True
    for name, model in (
        ("Resnet50", Resnet50()),
        ("Resnet50Extended", Resnet50Extended()),
        ("Resnet101", Resnet101()),
    ):
        print(f"--- {name} ---")
        tik = time.time()
        sample_input = np.random.randint(0, 256, (10, 6, 300, 400))
        sample_input = torch.from_numpy(sample_input).float()
        if enable_cuda and torch.cuda.is_available():
            print("Use CUDA")
            model = model.cuda()
            sample_input = sample_input.cuda()
        print("input:", sample_input.shape)
        sample_output = model(sample_input)
        print("output:", sample_output.shape)
        print(f"time: {time.time() - tik:.4f}s")
