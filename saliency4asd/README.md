# Saliency4ASD

## Code Structure

```bash
$ tree
.
├── LICENSE
├── README.md
├── code_forMetrics
│   ├── eval_on_dataset.m # Matlab script to evaluate multiple metrics
│   └── eval_result.txt # evaluation result on my testing dataset
├── data_loader.py # PyTorch dataloader
├── divide_dataset.py # Python script to divide dataset into train, val and test
├── model # U-2 Net definition
│   ├── __init__.py
│   └── u2net.py
├── saved_models
│   ├── u2net-trained.pth # my model trained on training dataset
│   └── u2net.pth # the pretained model
├── u2net_test.py # test
└── u2net_train.py # train
```

## How to use

If you want to train the model:

1. Modify `u2net_train.py` to make sure that it knows  the correct path of your training data and pretrained model. Also you can set some hyperparameters.
2. Run `python3 u2net_train.py`.

If you want to test the model:

1. Modify `u2net_test.py` to set your model location and folder for testing images.
2. Run `python3 u2net_test.py`.
3. Then you will see the predicted results in the folder you specified in Step 1.