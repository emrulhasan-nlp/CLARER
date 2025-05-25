import torch
import torch.nn as nn
import torch.nn.functional as F


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

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

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


class NRTPlus(nn.Module):
    def __init__(self, num_users, num_items, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, hidden_dim, rating_dim=1, criteria_dim=5):
        super(NRTPlus, self).__init__()

        # Encoder and decoder embeddings
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder and decoder layers
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        # Final output layer for explanation generation
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        ########### Rating Prediction Part ###########
        self.userid_emb = IDEncoder(num_users, num_items, d_model, hidden_dim)  # User-item embedding
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(d_model, hidden_dim)
        self.linearId = nn.Linear(128, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, rating_dim)
        self.criteria_layer = nn.Linear(hidden_dim, criteria_dim)

        ########### Additional Layers ###########
        self.enc_projection = nn.Linear(d_model + hidden_dim, d_model)
        self.linear_combined = nn.Linear(d_model + hidden_dim, hidden_dim)
        
        # Initialize decoder with user-item embeddings
        self.decoder_init_layer = nn.Linear(hidden_dim, d_model)
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1).bool().to(tgt.device)
        tgt_mask = tgt_mask & ~nopeak_mask
        return src_mask.to(src.device), tgt_mask.to(tgt.device)

    def forward(self, users, items, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embed input sequences
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        # Encoder forward pass
        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        # User-Item Embeddings
        userItem_emb = self.userid_emb(users, items)
        userItem_emb_expanded = userItem_emb.unsqueeze(1).repeat(1, enc_output.size(1), 1)
        enc_output = torch.cat((enc_output, userItem_emb_expanded), dim=2)
        enc_output = self.enc_projection(enc_output)

        ########### Decoder Improvement ###########
        # Initialize decoder input with transformed user-item embedding
        decoder_init = self.decoder_init_layer(userItem_emb)
        tgt_embedded[:, 0, :] += decoder_init  # Modify first decoder token

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        explanation = self.fc(dec_output)

        ########### Rating Prediction ###########
        enc_pooled = self.pool(enc_output.permute(0, 2, 1)).squeeze(-1)
        combined_emb = torch.cat((userItem_emb, enc_pooled), dim=1)
        combined_emb = self.linear_combined(combined_emb)
        combined_emb = self.relu(combined_emb)
        combined_emb = self.dropout(combined_emb)
        rating = self.output_layer(combined_emb)
        criteria = self.criteria_layer(combined_emb)

        return explanation, rating, criteria

