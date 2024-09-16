import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
import io
import os
import torch
from tqdm import tqdm
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, AutoModelForPreTraining
import time
import requests
from datetime import datetime


#Set the batch-size
batch_size = 8
# We already divided the test-set into 4-parts. 
# !!! If you have a central name please update the path !!!
test_id = 4
# 1. Path to update
data_dir = f"./test_data/test{test_id}/"
# 2. Path to update
metadata_df = pd.read_csv(f"./test_data/test{test_id}.csv")
metadata_df["image"] = [x.split('/')[-1] for x in metadata_df["image_link"]]
metadata_df = metadata_df.drop(columns=["image_link", "group_id", "Unnamed: 0"])

device = "cuda"
model_id = "google/paligemma-3b-ft-docvqa-448"
config = PeftConfig.from_pretrained("anuraktK/paligemma_vqav2_full_ft")
processor = AutoProcessor.from_pretrained(model_id)
base_model = AutoModelForPreTraining.from_pretrained("google/paligemma-3b-ft-docvqa-448").to(device)
model = PeftModel.from_pretrained(base_model, "anuraktK/paligemma_vqav2_full_ft").to(device)
model.eval()
print("Locked and loaded!")

# Record start time
start_time = datetime.now()
print(f"Start time: {start_time}")

pred_vals = {"preds":[], "img_name": []}

# Prepare batch inference
error_files = ""

#The for-loop was used for batch-inference as using test-loaders was giving errors
for batch_start in tqdm(range(0, len(metadata_df), batch_size)):
    batch_end = min(batch_start + batch_size, len(metadata_df))

    #Updating the batch
    batch_images = []
    batch_prompts = []
    for idx in range(batch_start, batch_end):
        img_path =  data_dir +  metadata_df["image"][idx]
        pred_vals["img_name"].append(metadata_df["image"][idx])
        prompt = f'What is the {metadata_df["entity_name"][idx]}'

        # Using img_path to load the image
        image = Image.open(img_path).convert('RGB')

        # (IMG,PROMPT) batch
        batch_images.append(image)
        batch_prompts.append(prompt)

    #Pre-processing the batch
    model_inputs = processor(text=batch_prompts, images=batch_images, return_tensors="pt", padding=True).to(device)
    
    with torch.inference_mode():
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        
    decoded = processor.batch_decode(generation, skip_special_tokens=True)
    pred_vals["preds"].extend(decoded)

    #Saving the intermediate dataset
    if batch_start % 200 == 0 and batch_start > 0:
        pred_df = pd.DataFrame(pred_vals)
        pred_df.to_csv(f"test_set_{test_id}_pred_palli_final.csv", index=False)

#Saving the final predictions
pred_df = pd.DataFrame(pred_vals)
pred_df.to_csv(f"test_set_{test_id}_pred_palli_final.csv", index=False)

# Saves the errors if any
with open("error_files.txt", "w") as ef:
    ef.write(error_files)

#Prints the execution time
end_time = datetime.now()
print(f"End time: {end_time}")
execution_time = end_time - start_time
print(f"Total execution time: {execution_time}")
