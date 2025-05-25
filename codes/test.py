import pandas as pd
import numpy as np
import torch
import argparse
import random
import torch.nn as nn
import torch.optim as optim
import json
import os
import math
import time
from utils import preparedata,create_dataloader,datasplit, test_explanation, generate_explanation
from models import NRTPlus2, ContrastiveLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer
from models import NRTPlus2
import warnings
warnings.filterwarnings("ignore")

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df=pd.read_csv("../data/Yelp/yelp.csv")

print(df.info())

ahlds


review_text="the swimming pool is fantastic"

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

num_users, num_items= 7209,3018
src_vocab_size, tgt_vocab_size= 30522,30522
d_model, num_heads, num_layers, d_ff= 256,8,6,1024
max_seq_length=128
dropout=0.1
hidden_dim= 64
criteria= False
calculate_aspRep=True

model = NRTPlus2(num_users,num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, criteria, calculate_aspRep)

model.to(device)
best_model_path = f"{criteria}_{calculate_aspRep}best_model.pth"

print(best_model_path)
model.load_state_dict(torch.load(best_model_path,weights_only=True))
model.eval()


max_length=len(review_text.split())
criteria=False
calculate_aspRep=True
user_id=0
item_id=0

text_input_ids = tokenizer(review_text, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")["input_ids"].to(device)

words=[]
for i in range(max_length):
    with torch.no_grad():  # No gradient calculation for inference

        user_id_tensor = torch.tensor([user_id], device=device)
        item_id_tensor = torch.tensor([item_id], device=device)
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)

        output, pred_rating, pred_criteria = model(user_id_tensor, item_id_tensor, text_input_ids, decoder_input_ids)
        print(i, output)

    probs = torch.softmax(output[:, -1, :], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    print(next_token)
    decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)  # Append token
    print(decoder_input_ids)
    if next_token.item() == tokenizer.eos_token_id:
        break # Stop if
    
    explanation=tokenizer.decode(decoder_input_ids.squeeze().tolist(), skip_special_tokens=True)
    words.append(explanation)
print(f"Ground Truth: {review_text}")
print(f"Prediction :{words}")






