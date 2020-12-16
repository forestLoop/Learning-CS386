import torch
from torch.autograd import Variable
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

import glob
import os

from data_loader import RescaleT
from data_loader import RandomCrop
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET


def validate(net, val_dataloader):
    print("--- Validation ---")
    net.eval()
    batch_num = 0
    running_loss, running_tar_loss = 0.0, 0.0
    with torch.no_grad():
        for _, data in enumerate(val_dataloader):
            batch_num += 1

            inputs, labels = data["image"], data["label"]
            inputs = inputs.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            # wrap them in Variable
            if torch.cuda.is_available():
                inputs_v, labels_v = (
                    Variable(inputs.cuda(), requires_grad=False),
                    Variable(labels.cuda(), requires_grad=False),
                )
            else:
                inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                    labels, requires_grad=False
                )
            # forward
            d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
            loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            # # print statistics
            running_loss += loss.item()
            running_tar_loss += loss2.item()
            # del temporary outputs and loss
            del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    avg_loss, avg_tar = running_loss / batch_num, running_tar_loss / batch_num
    print(f"val loss: {avg_loss:.3f}, tar: {avg_tar:.3f}")
    return avg_loss, avg_tar


def load_dataset(image_dir, image_ext, label_dir, batch_size, name=""):
    img_name_list = glob.glob(os.path.join(image_dir, "*" + image_ext))
    lbl_name_list = []
    for img_path in img_name_list:
        img_name = img_path.split(os.sep)[-1]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1, len(bbb)):
            imidx = imidx + "." + bbb[i]
        lbl_name_list.append(os.path.join(label_dir, imidx + "_s" + label_ext))

    print("---")
    print(f"{name} images: {len(img_name_list)}")
    print(f"{name} labels: {len(lbl_name_list)}")
    print("---")

    salobj_dataset = SalObjDataset(
        img_name_list=img_name_list,
        lbl_name_list=lbl_name_list,
        transform=transforms.Compose(
            [RescaleT(320), RandomCrop(288), ToTensorLab(flag=0)]
        ),
    )
    salobj_dataloader = DataLoader(
        salobj_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    return salobj_dataloader, len(img_name_list)


def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

    bce_loss = nn.BCELoss(size_average=True)
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, loss


# ------- 2. set the directory of training dataset --------

data_dir = "/root/TrainingDataset/TrainingData"
tra_image_dir = os.path.join(data_dir, "ImagesTrain")
tra_label_dir = os.path.join(data_dir, "TD_FixMaps")
val_image_dir = os.path.join(data_dir, "ImagesVal")
val_label_dir = tra_label_dir

image_ext = ".png"
label_ext = ".png"

best_model = "./saved_models/u2net-trained.pth"
pretrained_model_path = "./saved_models/u2net.pth"

epoch_num = 100000
batch_size_train = 8
batch_size_val = 15

train_dataloader, train_num = load_dataset(
    tra_image_dir, image_ext, tra_label_dir, batch_size_train, "train"
)
val_dataloader, val_num = load_dataset(
    val_image_dir, image_ext, val_label_dir, batch_size_val, "val"
)

# ------- 3. define model --------
# define the net
net = U2NET(3, 1)
if pretrained_model_path:
    print(f"Try to load pretrained model from {pretrained_model_path}")
    net.load_state_dict(torch.load(pretrained_model_path))
    print("Success")

if torch.cuda.is_available():
    print("Use CUDA")
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(
    net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0
)

# ------- 5. training process --------
print("---start training...")
ite_num = 0

min_loss = float("inf")

for epoch in range(0, epoch_num):
    val_loss, _ = validate(net, val_dataloader)
    if val_loss < min_loss:
        print("--- Save best model ---")
        print(f"min loss: {min_loss:.3f} -> {val_loss:.3f}")
        min_loss = val_loss
        torch.save(net.state_dict(), best_model)
    net.train()
    running_loss = 0.0
    running_tar_loss = 0.0
    batch_num = 0
    for i, data in enumerate(train_dataloader):
        ite_num, batch_num = ite_num + 1, batch_num + 1

        inputs, labels = data["image"], data["label"]

        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(
                labels.cuda(), requires_grad=False
            )
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(
                labels, requires_grad=False
            )

        # y zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        # # print statistics
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        print(
            f"[epoch: {epoch + 1:3d}/{epoch_num:3d}, batch: {(i + 1) * batch_size_train:3d}/{train_num:3d}, ite: {ite_num}] train loss: {loss.item():.3f}, tar: {loss2.item()}",
            flush=True,
        )

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4, d5, d6, loss2, loss
    print("--- Epoch Summary ---")
    print(
        f"Average loss: {running_loss / batch_num:.3f}, average tar: {running_tar_loss / batch_num:.3f}",
        flush=True,
    )
