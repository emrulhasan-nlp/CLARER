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
from utils import preparedata,create_dataloader,datasplit, generate_and_test_explanations
from models import NRTPlus2, ContrastiveLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type = str, default = "yelp", help = "Dataset category (Default: TripAdvisor Review Data)")
parser.add_argument("-rs", '--random_seed', dest = "random_seed", type = int, metavar = "<int>", default = 1702, help = 'Random seed (Default: 1702)')
parser.add_argument("-seq", '--seq_length', dest = "seq_length", type = int, metavar = "<int>", default = 128, help = 'Maximum Sequence Length (Default: 128)')
parser.add_argument("-src_vocab", '--src_vocab', dest = "src_vocab_size", type = int, metavar = "<int>", default = 30522, help = 'Vocabulary size for source review (Default: 20000)')
parser.add_argument("-tgt_vocab", '--tgt_vocab', dest = "tgt_vocab_size", type = int, metavar = "<int>", default = 30522, help = 'Vocabulary size for starget review (Default: 20000')
parser.add_argument("-bs", '--batch_size', dest = "batch_size", type = int, metavar = "<int>", default = 64, help = 'Batch size (Default: 64)')
parser.add_argument("-ft", '--feat_type', dest = "feat_type", type = bool, default = False, help = 'criteria (multiple criteria)')
parser.add_argument("-asRep", '--aspect_rep', dest = "aspect_type", type = bool, default = True, help = 'Aspect based representation (multiple criteria)')
args = parser.parse_args()

CATEGORY = args.dataset.strip().lower()
print("\nDataset: {}".format( CATEGORY ))

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Random seed
np.random.seed(args.random_seed)
random.seed(args.random_seed)

startTime = time.time()


SOURCE_FOLDER="../data/Yelp/"

filePath = "{}{}.csv".format( SOURCE_FOLDER,CATEGORY )

max_seq_length=args.seq_length
src_vocab_size=args.src_vocab_size
tgt_vocab_size=args.tgt_vocab_size
batch_size=args.batch_size
tripadvisor=False
data,num_users,num_items=preparedata(filePath, max_seq_length, src_vocab_size,tgt_vocab_size, tripadvisor)

#print(num_users,num_items)
#ahksjd
train_data,val_data, test_data=datasplit(data,random_state=1702)

train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)

d_model = 256
num_heads = 8
num_layers = 6
d_ff = 1024
output_dim=1
max_seq_length = 128
dropout = 0.1
hidden_dim = 64
criteria=args.feat_type
calculate_aspRep=args.aspect_type

print(criteria)
print(calculate_aspRep)
# Define weights for each loss term
λ1, λ2, λ3 = 1.0,0.5,1.0   # Weight for explanation loss, # Weight for rating prediction loss, Weight for contrastive loss (typically small)

model = NRTPlus2(num_users,num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, criteria, calculate_aspRep)
model.to(device)

# Load the best model before testing
best_model_path=best_model_path = f"{CATEGORY}_{criteria}_{calculate_aspRep}best_model.pth"


model.load_state_dict(torch.load(best_model_path,weights_only=True))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#haksjd
output_file=f"{CATEGORY}_{λ1}_criteria{criteria}_aspect{calculate_aspRep}_xpltest.txt"

print(criteria, calculate_aspRep)
test_xplanationPairs= generate_and_test_explanations(test_data, model, device, criteria, calculate_aspRep, output_file)





