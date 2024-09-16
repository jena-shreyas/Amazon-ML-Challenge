import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from PIL import Image
import io
import os
from bisect import bisect_left
import torch
from tqdm import tqdm
import wandb
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoProcessor, AutoModelForPreTraining

from transformers import BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#Add the file-path here
# 1. Path to images folder
NUM_EPOCHS = 1
BATCH_SIZE = 1
data_dir = "/home/anurakt/ak_work_etc/my_expts/dataaa/train/" 
# 2. Path to test.csv
csv_filename = "/home/anurakt/ak_work_etc/my_expts/sampled_df.csv"
metadata_df = pd.read_csv(csv_filename)
metadata_df["image"] = [x.split('/')[-1] for x in metadata_df["image_link"]]

metadata_df = metadata_df.drop(columns=["image_link", "group_id"])

sample_sizes = {
    'depth': 3000,
    'width': 3000,
    'item_weight': 3000,
    'height': 3000,
    'voltage': 750,
    'wattage': 750,
    'item_volume': 750,
    'maximum_weight_recommendation': 750
}

def sample_from_group(group, size):
    return group.sample(n=size, random_state=1)

metadata_df = metadata_df.groupby('entity_name').apply(lambda x: sample_from_group(x, sample_sizes.get(x.name, len(x)))).reset_index(drop=True)
new_eval = [f'What is the {x}?' for x in metadata_df["entity_name"]]
metadata_df["entity_name"] = new_eval

#Dataset class for training the model
class VQADataset(Dataset):
    def __init__(self, df_val, img_folder, transform=None):
        self.data = df_val
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image']
        prefix = self.data.iloc[idx]['entity_name']
        suffix = self.data.iloc[idx]['entity_value']
        
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return {"image": image, "entity_name": prefix, "entity_value":suffix}
    

transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

#Initializing the dataset
train_dataset = VQADataset(df_val=metadata_df, img_folder=data_dir, transform=transform)
dataset_size = len(train_dataset)
print(f"Train size: {dataset_size}")

#Loading the model
device = "cuda"
model_id = "google/paligemma-3b-ft-docvqa-448"
processor = AutoProcessor.from_pretrained(model_id, do_rescale=False)

def collate_fn(examples):
    
    prefixes = [example['entity_name'] for example in examples]
    suffixes = [example['entity_value'] for example in examples]
    images = [example["image"] for example in examples]

    images = torch.stack(images)
    tokens = processor(text=prefixes, images=images, suffix=suffixes,
                       return_tensors="pt", padding="longest",
                       tokenize_newline_separately=False)

    tokens = tokens.to(torch.bfloat16).to(device)

    return tokens

#Quantising the model
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_type=torch.bfloat16,
    skip_modules_not_needed=True,
    offload_to_cpu=True,
)
#Configuring LoRa adapter
lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = AutoModelForPreTraining.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
model = get_peft_model(model, lora_config)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False
    
print(model.print_trainable_parameters())

#wandb logging parameters
wandb.init(
    project="pali_gemma_ft_amz",
    config={
    "learning_rate": 2e-5,
    "architecture": "Pali_Gemma896_docvqa",
    "dataset": "amazon_mlc",
    "epochs": 1,
    }
)

#Setting up the training
from transformers import Trainer, TrainingArguments
args=TrainingArguments(
            num_train_epochs=NUM_EPOCHS,
            remove_unused_columns=False,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            warmup_steps=2,
            learning_rate=5e-6,
            weight_decay=1e-6,
            adam_beta2=0.999,
            logging_steps=100,
            optim="adamw_hf",
            save_strategy="steps",
            save_steps=1000,
            push_to_hub=True,
            save_total_limit=1,
            output_dir="paligemma_vqav2_full_new_mod",
            bf16=True,
            report_to=["wandb"],
            dataloader_pin_memory=False
        )


trainer = Trainer(
        model=model,
        train_dataset=train_dataset ,
        data_collator=collate_fn,
        args=args
        )

#Training the model
trainer.train()

trainer.push_to_hub()