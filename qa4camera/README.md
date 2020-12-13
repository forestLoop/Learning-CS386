# README

Folder structure:
```bash
$ tree .
├── dataset.py  # custom pytorch dataset loader
├── eval.py # given a model, run validation with it on scene 81 - 100
├── model/ # folder for trained models
├── models.py   # model definitions
├── README.md
├── submit_result.py # apply models to testing scenes and write results to csv files
└── train.py # train models
```

If you'd like to use these models:

```bash
python3 submit_result.py
```
