from model import MyModel
from dataset import ImageDataset
from torch.utils.data import DataLoader
import os
import torch.nn as nn
import torch
import time

is_cuda = False


def validate(dataloader, model):
    print("------ Validation ------")
    model.eval()
    loss_function = torch.nn.BCELoss()
    total_loss = 0.0
    with torch.no_grad():
        tik = time.time()
        for inputs, labels in dataloader:
            inputs, labels = inputs.float(), labels.float()
            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            predicted = model(inputs)
            loss = loss_function(predicted, labels)
            print(loss.item(), end=",")
            total_loss += loss.item()
        tok = time.time()
        print(
            f"Validation finished in {tok-tik:.3f}s, total loss is {total_loss:.5f}.")
    return total_loss


def train(dataset_dir, model_path, batch_size, epochs, train_scenes, val_scenes):
    train_dataset = ImageDataset(dataset_dir, train_scenes)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    val_dataset = ImageDataset(dataset_dir, val_scenes)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True
    )
    if os.path.exists(model_path):
        print("------ Load Model --------")
        model = torch.load(model_path)
    else:
        print("------ Create Model ------")
        model = MyModel()
    model = nn.DataParallel(model)
    if is_cuda:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = torch.nn.BCELoss()
    min_loss = validate(val_dataloader, model)
    val_loss_history = [min_loss, ]
    for e in range(epochs):
        model.train()
        print(f"------ Epoch {e} --------")
        tik = time.time()
        for inputs, labels in train_dataloader:
            print(inputs.shape, labels.shape)
            # prepare
            model.zero_grad()
            inputs, labels = inputs.float(), labels.float()
            if is_cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # forward
            predicted = model(inputs)
            # compute loss
            loss = loss_function(predicted, labels)
            print(loss.item(), end=",")
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
        tok = time.time()
        print()
        print(f"Epoch {e} finished in {tok-tik:.3f}s")
        val_loss = validate(val_dataloader, model)
        val_loss_history.append(val_loss)
        if val_loss < min_loss:
            print("--------- Save ----------")
            print(f"- Val loss: {min_loss:.5f} -> {val_loss:.5f}")
            min_loss = val_loss
            torch.save(model, model_path)
        elif val_loss > 2 * min_loss:
            print("Early stopping!")
            break
    print(f"Training finished after {e+1} epochs.")
    print(f"Validation loss history:")


if __name__ == "__main__":
    train("C:/MyCode/DIP/DIP1/dataset", "model/test.pth", 16, 4, {1, }, {2, })
