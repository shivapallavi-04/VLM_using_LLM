import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.models as models
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

# Check for GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU.")

# File paths
train_path = "/kaggle/input/vqa-dataset/archive (6)/train_data.csv"
eval_path = "/kaggle/input/vqa-dataset/archive (6)/eval_data.csv"
image_path = "/kaggle/input/vqa-dataset/archive (8)/dataset/images"

# Load and preprocess data
dataframe = pd.DataFrame(pd.read_csv(train_path))
eval_dataframe = pd.DataFrame(pd.read_csv(eval_path))
dataframe['image_id'] = dataframe['image_id'] + '.png'
eval_dataframe['image_id'] = eval_dataframe['image_id'] + '.png'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


