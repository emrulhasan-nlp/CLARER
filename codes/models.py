
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

###############Multihead Attention#############################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

#################Pointwise FeedForward Network ##########################
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

#################################Positional Encoding #################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

######################################Encoder Layer####################################
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

#######################################Decoder Layer########################
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

####################User and Item ID encoder################################
class IDEncoder(nn.Module):
    def __init__(self, num_users, num_items, emb_dim, hidden_dim):
        super(IDEncoder, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self.linear1 = nn.Linear(emb_dim * 2, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # Change here: in_features should match the output of linear1 and relu
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_emb(user_ids)
        item_emb = self.item_emb(item_ids)
        combined = torch.cat([user_emb, item_emb], dim=1)
        x = self.linear1(combined)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)

    def forward(self, review_embeds, aspect_embeds):

        review_embeds = review_embeds.mean(dim=1, keepdim=True) # shape (16, 1, 384)
        pos_score = self.cosine_similarity(review_embeds, aspect_embeds)
        neg_embeds = aspect_embeds[torch.randperm(aspect_embeds.size(0))]
        neg_score = self.cosine_similarity(review_embeds, neg_embeds)

        loss = -torch.log(torch.exp(pos_score / self.temperature) / (torch.exp(pos_score / self.temperature) + torch.exp(neg_score / self.temperature)))
        return loss.mean()

###########################NRTPlus++ model, consider review and id information#####################
class NRTPlus(nn.Module):
    def __init__(self, num_users, num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, criteria=True, calculate_aspRep=False):
        super(NRTPlus, self).__init__()

        self.criteria = criteria 
        self.calculate_aspRep = calculate_aspRep
        # Encoder and decoder embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.norm_layer = nn.LayerNorm(d_model)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
      
        self.enc_projection = nn.Linear(d_model, d_model)  
        self.gate_layer = nn.Linear(d_model, d_model)  

        # Final decoder output layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        ########### Rating prediction part ###########
        self.userid_emb = IDEncoder(num_users, num_items, d_model, hidden_dim)  # User-item embedding
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool to shape (batch_size, word_dim, 1), Pooling layer for review embeddings
        self.linear = nn.Linear(d_model, hidden_dim)
       
        self.projection = nn.Linear(hidden_dim, d_model)
        self.linearId = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_combined = nn.Linear(d_model + hidden_dim, hidden_dim)  # To reduce combined size to hidden_dim
        self.ratingPred_layer = nn.Linear(hidden_dim, 1)  # Rating prediction output
        
        if criteria:
          self.citeria_layer = nn.Linear(hidden_dim, 5)
          
        #########Alignement for aspect Representation Layer
        if self.calculate_aspRep:
          self.aspect_projection = nn.Linear(d_model, 384)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)  # Move src_mask to the device of src
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device=tgt.device), diagonal=1)).bool()  # Create nopeak_mask on the device of tgt
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, users, items, src, tgt):
        # Generate masks for the src and tgt
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed the input sequences with positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Pass through encoder layers
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)
        
        enc_output = self.norm_layer(self.enc_projection(enc_output))
        ########### Integrate User-Item Embeddings into enc_output ############
        userItem_emb = self.userid_emb(users, items)  # Shape: (batch_size, hidden_dim)
        userItem_emb = F.normalize(userItem_emb, p=2, dim=1)

        # Expand userItem_emb and concatenate with enc_output
        userItem_emb_expanded = userItem_emb.unsqueeze(1).repeat(1, enc_output.size(1), 1)  # Shape: (batch_size, seq_length, hidden_dim)
        
        userItem_emb_expanded = self.projection(userItem_emb_expanded)
        userItem_emb_expanded = self.relu(userItem_emb_expanded)
        userItem_emb_expanded = self.dropout(userItem_emb_expanded)
        gate = torch.sigmoid(self.gate_layer(enc_output))
        enc_output = gate * enc_output + (1 - gate) * userItem_emb_expanded

        #enc_output = torch.cat((enc_output, userItem_emb_expanded), dim=2)  # Shape: (batch_size, seq_length, d_model + hidden_dim)

        #enc_output = self.enc_projection(enc_output)  # Shape: (batch_size, seq_length, d_model)

        ########### Pass modified enc_output to the decoder ############
        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        ########### Rating Prediction Layer ###################
        # Pooling and reshaping for rating prediction
        enc_pooled = self.pool(enc_output.permute(0, 2, 1)).squeeze(-1)  # Shape: (batch_size, d_model)
        combined_emb = torch.cat((userItem_emb, enc_pooled), dim=1)  # Shape: (batch_size, hidden_dim + d_model)
        combined_emb = self.linear_combined(combined_emb)  # Reduce to hidden_dim
        combined_emb = self.relu(combined_emb)
        combined_emb = self.dropout(combined_emb)

        # All four outputs of the model
      
        explanation = self.fc(dec_output)
        rating = self.ratingPred_layer(combined_emb)  # Output rating score
        if self.criteria:
          criteria_pred = self.citeria_layer(combined_emb)
          return explanation, rating, criteria_pred

        elif self.calculate_aspRep:
          aspect_rep=self.aspect_projection(enc_output)
          return explanation, rating, aspect_rep
        else:
          return explanation, rating

