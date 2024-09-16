# PaliGemma Model Training and Inference

This repository provides scripts for training and performing inference using the PaliGemma model. The model is designed for visual question answering (VQA) tasks. The scripts were made by our team "Attack On Python". 


# Task

Given the images of online products on Amazon with various measurements of physical quantities (e.g., height, width, weight) specified, **extract the numerical values corresponding to the physical quantities given as input**.

# Results

Our solution achieved a maximum F1-Score of `0.661` and secured a Top-30 finish (Public Leaderboard) among over 2000 participating teams.

## Table of Contents
- [Requirements](#requirements)
- [Training](#training)
- [Inference](#inference)

## Requirements

Ensure you have the following dependencies installed:

Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Training

- Download the images in a directory by passing the list of links from train.csv to ```util.download_images(<list of link of images>)```
- Add the path of the images directory in the ``` data_dir ``` and the path to train.csv in ```csv_filename```
- Run the ```PaliGemma_Training_AttackOnPython.py``` file after setting up the ```NUM_EPOCHS``` and ```BATCH_SIZE``` hyperparameter



## Testing

- Download the images in a directory by passing the list of links from test.csv to ```util.download_images(<list of link of images>)```
- Add the path of the images directory in the ``` data_dir ``` and the path to train.csv in metadata_df's ```pd.read_csv(<test.csv path>)```
- Run the ```PaliGemma_Training_AttackOnPython.py``` file after setting up the ```batch_size``` and ```test_id``` hyperparameter.
