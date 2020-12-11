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
        self.pretrained_resnet = torchvision.models.resnet101(pretrained=True)
        # set input channels to 6 as we stack up two images
        self.pretrained_resnet.conv1 = Conv2d(
            6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        # set output dimension to 4
        self.pretrained_resnet.fc = nn.Linear(2048, 4)

    def forward(self, x):
        x = self.pretrained_resnet(x)
        # add sigmoid function before output
        x = torch.sigmoid(x)
        return x


if __name__ == "__main__":
    import time

    print("Note: this file is not expected to be run directly!")
    print("Test Resnet101:")
    model = Resnet101()
    for shape in ((10, 6, 300, 400), (1, 6, 3000, 4000)):
        tik = time.time()
        sample_input = np.random.randint(0, 256, shape)
        sample_input = torch.from_numpy(sample_input).float()
        print("input:", sample_input.shape)
        sample_output = model(sample_input)
        print("output:", sample_output)
        print("time: ", time.time() - tik)
