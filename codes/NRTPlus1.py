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
from utils import preparedata,create_dataloader,datasplit,generate_and_test_explanations, test_explanation
from models import NRTPlus2, ContrastiveLoss
from sklearn.metrics import mean_squared_error, mean_absolute_error
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type = str, default = "trip_review_xplns", help = "Dataset category (Default: TripAdvisor Review Data)")
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

# ============================================================================= INPUT =============================================================================
SOURCE_FOLDER = "../data/TripAdvisor/"

#SOURCE_FOLDER="../data/Amazon/MoviesAndTV/"
#SOURCE_FOLDER="../data/Yelp/"

filePath = "{}{}.csv".format( SOURCE_FOLDER,CATEGORY )

print(filePath)
max_seq_length=args.seq_length
src_vocab_size=args.src_vocab_size
tgt_vocab_size=args.tgt_vocab_size
batch_size=args.batch_size
tripadvisor=False
data,num_users,num_items=preparedata(filePath, max_seq_length, src_vocab_size,tgt_vocab_size, tripadvisor)

print(num_users,num_items)
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
#dhfjdk
# Define weights for each loss term
λ1, λ2, λ3 = 1.0,0.5,1.0   # Weight for explanation loss, # Weight for rating prediction loss, Weight for contrastive loss (typically small)

model = NRTPlus2(num_users,num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, criteria, calculate_aspRep)
model.to(device)

contrastive_loss = ContrastiveLoss()
criterion = nn.CrossEntropyLoss(ignore_index=0)
rating_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

print("Training started")
###############start training ##########
asp_emb=np.load('../data/TripAdvisor/aspect_embeddings.npy')

#asp_emb=np.load("../data/Amazon/MoviesAndTV/aspect_emb.npy")
#asp_emb=np.load("../data/Yelp/ylp_asp_embeddings.npy")

asp_emb=torch.from_numpy(asp_emb).to(device)
start_time=time.time()

num_epochs = 20
patience = 3  # Number of epochs to wait before early stopping
best_val_loss = float('inf')
early_stop_counter = 0
best_model_path = f"{CATEGORY}_{criteria}_{calculate_aspRep}best_model.pth"

print(best_model_path)
#hskjdh
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    total_train_loss = 0

    for batch in train_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

        # Convert ratings to float before calculating rating_loss
        if tripadvisor:
            ratings = batch[4][:,5:6].type(torch.float32).to(device)
            criteria_ratings = batch[4][:,0:5].type(torch.float32).to(device)
        else:
            ratings=batch[4].type(torch.float32).to(device)

        #print(ratings)
        optimizer.zero_grad()
        batch_len=len(userId)

        if criteria:
          output,pred_rating, pred_criteria=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
          xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
          rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
          criteria_loss = criteria_loss=rating_criterion(pred_criteria.reshape(-1), criteria_ratings.reshape(-1))

          loss = λ1 * xpln_loss + λ2 * rating_loss + λ3 * criteria_loss
        elif calculate_aspRep:
          asp_embed=torch.stack([asp_emb for _ in range(batch_len)],dim=0).to(device)
          output,pred_rating, aspRep=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
          xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
          rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
          contrast_loss = contrastive_loss(asp_embed, asp_embed)
          loss = λ1 * xpln_loss + λ2 * rating_loss + λ3 * contrast_loss
        else:
          output,pred_rating=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
          xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
          rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
          loss = λ1 * xpln_loss + λ2 * rating_loss

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation loop
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            userId = batch[0].to(device)
            itemId = batch[1].to(device)
            src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

            # Convert ratings to float before calculating rating_loss
            if tripadvisor:

                ratings = batch[4][:,5:6].type(torch.float32).to(device)
                criteria_ratings = batch[4][:,0:5].type(torch.float32).to(device)
            else:
                ratings=batch[4].type(torch.float32).to(device)

            if criteria:
              output,pred_rating, pred_criteria=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
              xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
              rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
              criteria_loss = criteria_loss=rating_criterion(pred_criteria.reshape(-1), criteria_ratings.reshape(-1))

              loss = λ1 * xpln_loss + λ2 * rating_loss + λ3 * criteria_loss
            elif calculate_aspRep:
              asp_embed=torch.stack([asp_emb for _ in range(batch_len)],dim=0).to(device)
              output,pred_rating, aspRep=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
              xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
              rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
              contrast_loss = contrastive_loss(asp_embed, asp_embed)
              loss = λ1 * xpln_loss + λ2 * rating_loss + λ3 * contrast_loss
            else:
              output,pred_rating=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep

              xpln_loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_batch[:, 1:].contiguous().view(-1))
              rating_loss = rating_criterion(pred_rating.squeeze(), ratings.squeeze())
              loss = λ1 * xpln_loss + λ2 * rating_loss
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

# Load the best model before testing
model.load_state_dict(torch.load(best_model_path,weights_only=True))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
y_trues=[]
y_preds=[]
with torch.no_grad():

    for batch in test_loader:
        userId = batch[0].to(device)
        itemId = batch[1].to(device)
        src_batch, tgt_batch = batch[2].to(device), batch[3].to(device)

        # Convert ratings to float before calculating rating_loss
        if tripadvisor:

            ratings = batch[4][:,5:6].type(torch.float32).to(device)
            criteria_ratings = batch[4][:,0:5].type(torch.float32).to(device)
        else:
            ratings=batch[4].type(torch.float32).to(device)

        if criteria:
          output,pred_rating, pred_criteria=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
        elif calculate_aspRep:
          output,pred_rating, aspRep=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep
        else:
          output,pred_rating=model(userId, itemId, src_batch, tgt_batch[:, :-1]) #Output: Decoder output for explanation, rating is the predicted ratings, aspect_rep is review rep

        y_trues.extend(ratings.cpu().numpy().flatten())
        y_preds.extend(pred_rating.cpu().numpy().flatten())

        # Replace tokenizer.bos_token_id with tokenizer.cls_token_id
        decoder_input_ids = torch.tensor([[tokenizer.cls_token_id]] * userId.shape[0], device=device)

        next_token = torch.argmax(output[:, -1, :], dim=-1, keepdim=True)  # Get next word ID
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)  # Append token

        if tokenizer.eos_token_id in next_token.tolist():
          break # Stop if

        output = tokenizer.decode(decoder_input_ids.squeeze().flatten().tolist(), skip_special_tokens=True)

mse = mean_squared_error(y_trues, y_preds)
rmse = np.sqrt(mse)  # Root Mean Squared Error
mae = mean_absolute_error(y_trues, y_preds)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")

start_time=time.time()

#haksjd
output_file=f"{CATEGORY}_{λ1}_criteria{criteria}_aspect{calculate_aspRep}_xpltest.txt"

print(criteria, calculate_aspRep)

test_xplanationPairs= generate_and_test_explanations(test_data, model, device, criteria, calculate_aspRep, output_file)


