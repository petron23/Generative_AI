import torch 
import torch.nn as  nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape (seq_len, d_model)
        self.pe = torch.zeros(seq_len, d_model)

        # Vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()) / d_model * (-math.log(10000.))
        #div_term = torch.exp((torch.log(10000.**(-torch.arange(0,64,2).float() / 64))))

        # Giving values to the positional encoding matrix
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)


        # Adding another dimension to the pe matrix as we will use batches
        self.pe = self.pe.unsqueeze(0) #(1, seq_len, d_model)

        # Register the tensor in the buffer of the model, we want this to be saved not as a learned param
        # I.e., it is saved in the file along with the model
        self.register_buffer("pe", self.pe)

    def forward(self, x):
        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x-mean)/(std + self.eps) + self.bias
        

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int , d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
"""
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super(FeedForwardBlock, self).__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.feedforward(x)
"""

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        """Initializing the multiheadattentiaon layer

        Args:
            d_model (int): the dimension of encoding
            h (int): number of heads it must divide d_model
            dropout (float): dropout parameter
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert self.d_model % self.h == 0, "d_model must be divisble by h"

        self.d_k = self.d_model // h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_0 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = value.shape[-1]
        

        attention_scores = (query @ key.transpos(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, 10**-9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_q(q)
        value = self.w_v(v)

        # Splitting into h heads each matrix
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, seq_len, d_k) --> (Batch, seq_len, d_model = d_k * h)
        x = x.transpose(1,2).contigous().view(x.shape[0], -1, self.d_k * self.h)

        # (Batch, seq_len, d_model)
        return self.w_o(x) 


