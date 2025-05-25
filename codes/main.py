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
from utils import preparedata,create_dataloader,datasplit, test_explanation,bleu_score, rouge_score
from models import NRTPlus
from sklearn.metrics import mean_squared_error, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type = str, default = "trip_review_xplns", help = "Dataset category (Default: TripAdvisor Review Data)")
parser.add_argument("-rs", '--random_seed', dest = "random_seed", type = int, metavar = "<int>", default = 1702, help = 'Random seed (Default: 1702)')
parser.add_argument("-seq", '--seq_length', dest = "seq_length", type = int, metavar = "<int>", default = 256, help = 'Maximum Sequence Length (Default: 128)')
parser.add_argument("-src_vocab", '--src_vocab', dest = "src_vocab_size", type = int, metavar = "<int>", default = 20000, help = 'Vocabulary size for source review (Default: 20000)')
parser.add_argument("-tgt_vocab", '--tgt_vocab', dest = "tgt_vocab_size", type = int, metavar = "<int>", default = 20000, help = 'Vocabulary size for starget review (Default: 20000')
parser.add_argument("-bs", '--batch_size', dest = "batch_size", type = int, metavar = "<int>", default = 64, help = 'Batch size (Default: 64)')
args = parser.parse_args()


# Dataset, e.g. amazon_instant_video
CATEGORY = args.dataset.strip().lower()
print("\nDataset: {}".format( CATEGORY ))


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Random seed
np.random.seed(args.random_seed)
random.seed(args.random_seed)

startTime = time.time()

# ============================================================================= INPUT =============================================================================
# Source Folder
SOURCE_FOLDER = "../data/TripAdvisor/"


filePath = "{}{}.csv".format( SOURCE_FOLDER,CATEGORY )

best_model_path="best_model.pth"
max_seq_length=args.seq_length
src_vocab_size=args.src_vocab_size
tgt_vocab_size=args.tgt_vocab_size
batch_size=args.batch_size

data,num_users,num_items=preparedata(filePath, max_seq_length, src_vocab_size,tgt_vocab_size)

train_data,val_data, test_data=datasplit(data,random_state=1702)

train_loader = create_dataloader(train_data, batch_size=batch_size, shuffle=True)
val_loader = create_dataloader(val_data, batch_size=batch_size, shuffle=False)
test_loader = create_dataloader(test_data, batch_size=batch_size, shuffle=False)


print("filePath")

#ahskjdh
########Hyper parameters
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 1024
rating_dim=1
criteria_dim=5
dropout = 0.1
hidden_dim = 128


#############initialize the model#####################
model = NRTPlus(num_users,num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, rating_dim, criteria_dim)
model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss(ignore_index=0) # This is for generation
rating_criterion = nn.MSELoss() # This crierion is for rating prediction
# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


#################################Trianing ############################

start_time=time.time()

num_epochs = 20
patience = 3  # Number of epochs to wait before early stopping
best_val_loss = float('inf')
early_stop_counter = 0
best_model_path = "best_model.pth"

for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0

    for batch in train_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

        criteria=batch[5][:,0:5].type(torch.float32).to(device)
        ratings = batch[5][:,5:6].type(torch.float32).to(device)

        optimizer.zero_grad()
        explanation, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])  # Shift target sequence left

        exp_loss = criterion(explanation.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
        rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
        criteria_loss=rating_criterion(pred_criteria.reshape(-1), criteria.reshape(-1))
        loss = exp_loss + rating_loss+criteria_loss

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
        #print(loss.item())
        #break

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            userId = batch[0].to(device)
            itemId = batch[1].to(device)
            src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

            criteria=batch[5][:,0:5].type(torch.float32).to(device)
            ratings = batch[5][:,5:6].type(torch.float32).to(device)

            explanation, pred_rating, pred_criteria = model(userId, itemId, src_batch, tgt_batch[:, :-1])  # Shift target sequence left

            exp_loss = criterion(explanation.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
            rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
            criteria_loss=rating_criterion(pred_criteria.reshape(-1), criteria.reshape(-1))
            loss = exp_loss + rating_loss+criteria_loss

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch: {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Check for early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0  # Reset counter if validation loss improves
        torch.save(model.state_dict(), best_model_path)  # Save the best model
        print("Best model saved!")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    if early_stop_counter >= patience:
        print("Early stopping triggered. Stopping training.")
        break

end_time=time.time()

total_train_valtime=end_time-start_time
hours,remainder = divmod(total_train_valtime, 3600)
minutes, sec=divmod(remainder, 60)

print(f"\nTime taken for training and valiation, hours:{hours}, minutes:{minutes}")

#jhasjk
# Load the best model before testing
start_time=time.time()

model.load_state_dict(torch.load(best_model_path,weights_only=True))
model.eval()

y_trues=[]
y_preds=[]
with torch.no_grad():
    for batch in test_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

        criteria=batch[5][:,0:5].type(torch.float32).to(device)
        ratings = batch[5][:,5:6].type(torch.float32).to(device)

        _, pred_rating, _ = model(userId, itemId, src_batch, tgt_batch[:, :-1])  # Shift target sequence left
        y_trues.extend(ratings.cpu().numpy().flatten())
        y_preds.extend(pred_rating.cpu().numpy().flatten())


mse = mean_squared_error(y_trues, y_preds)
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_trues, y_preds)

# Print results rounded to 4 decimal places
print(f"\n\nMean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

end_time=time.time()

total_train_valtime=end_time-start_time
hours,remainder = divmod(total_train_valtime, 3600)
minutes, sec=divmod(remainder, 60)

print(f"\nTime taken for testing, hours:{hours}, minutes:{minutes}")


################################################evaluation of explanation###############
start_time=time.time()
save_explanation="test_explanation.txt"
test_xplanationPairs=test_explanation(test_data, model, device, save_explanation)

# Store results
bleu_scores = []
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Compute scores for each pair
for explanation, predicted_explanation in test_xplanationPairs:
    # Compute BLEU score
    bleu = bleu_score([explanation], [predicted_explanation])
    bleu_scores.append(bleu)

    # Compute ROUGE scores
    rouge_scores = rouge_score([explanation], [predicted_explanation])
    rouge1_scores.append(rouge_scores['rouge_1/f_score'])
    rouge2_scores.append(rouge_scores['rouge_2/f_score'])
    rougeL_scores.append(rouge_scores['rouge_l/f_score'])

# Compute average scores
avg_bleu = sum(bleu_scores) / len(bleu_scores)
avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
avg_rougeL = sum(rougeL_scores) / len(rougeL_scores)

# Print results
print(f"\nAverage BLEU Score: {avg_bleu:.4f}")
print(f"Average ROUGE-1 Score: {avg_rouge1:.4f}")
print(f"Average ROUGE-2 Score: {avg_rouge2:.4f}")
print(f"Average ROUGE-L Score: {avg_rougeL:.4f}")


end_time=time.time()

total_train_valtime=end_time-start_time
hours,remainder = divmod(total_train_valtime, 3600)
minutes, sec=divmod(remainder, 60)

print(f"\nTime taken for explanation evaluation, hours:{hours}, minutes:{minutes}")


