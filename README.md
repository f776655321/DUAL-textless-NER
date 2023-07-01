# DUAL-textless-NER

## Data Process
### URL: https://drive.google.com/drive/folders/1BL0WpXcZf1O6_cjIyI8bOpU5vUPZOLUr?usp=drive_link

### Get Data
```
unzip code-data.zip
```

## Before Training
* modify args.json to change the training parameter
* modify baseline.yaml's data/data_dir to specify the data directory

* modify the train.py's SQADataset to specify the data you want to train with

## Start Training
modify run.sh WANDB_PROJECT to your wandb project or you can delete run.sh "report_to" to prevent wandb
```
bash run.sh
```

## Evaluate
modify ner_inference.py parser to specify the data directory and model path

```
python ner_inference.py
```






