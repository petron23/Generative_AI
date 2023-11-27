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
        div_term = torch.exp(torch.arange(torch.arange(0,d_model,2).float()) / d_model * (-math.log(10000.)))
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