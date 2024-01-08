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
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()) / d_model * (-math.log(10000.))
        #div_term = torch.exp((torch.log(10000.**(-torch.arange(0,64,2).float() / 64))))

        # Giving values to the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)


        # Adding another dimension to the pe matrix as we will use batches
        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        # Register the tensor in the buffer of the model, we want this to be saved not as a learned param
        # I.e., it is saved in the file along with the model
        self.register_buffer("pe", pe)

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
        x = x.float()
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
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2,-1))/math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, 10**-9)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) #(Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        # Splitting into h heads each matrix
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # (Batch, seq_len, d_k) --> (Batch, seq_len, d_model = d_k * h)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.d_k * self.h)

        # (Batch, seq_len, d_model)
        return self.w_o(x) 


# Add and Norm
class ResidualConnection(nn.Module):

    def __init__(self, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__() 
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) 

    def forward(self, x, src_mask):
        return self.residual_connections[1](self.residual_connections[0](x,
                                                                        lambda x: self.self_attention_block(x ,x , x, src_mask)),
                                            self.feed_forward_block)
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.normalization = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.normalization(x)
    

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x



        #x = self.residual_connections[2](
        #        self.residual_connections[1](
        #            self.residual_connections[0](
        #                x, self.self_attention_block(x, x, x, tgt_mask)),
        #            self.cross_attention_block(x,encoder_output, encoder_output, src_mask)),
        #        self.feed_forward_block) 
        #return x
    

class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization
    
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_soft_max(self.proj(x), dim=-1)

class Transformer(nn.Module): 

    def __init__(self,
                encoder: Encoder,
                decoder: Decoder, 
                src_embedding: InputEmbeddings, 
                tgt_embedding: InputEmbeddings,
                src_position: PositionalEncoding,
                tgt_position: PositionalEncoding,
                projection_layer: ProjectionLayer
                ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_position = src_position
        self.tgt_position = tgt_position
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        return self.encoder(self.src_position(self.src_embedding(src)), src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_position(self.tgt_embedding(tgt)), encoder_output, src_mask, tgt_mask) 

    def project(self, x):
        return self.projection_layer(x)


# Creating the transformer from the classes with given params
def build_transformer(src_vocab_size: int,
                      tgt_vocab_size: int,
                      src_seq_len: int,
                      tgt_seq_len: int,
                      d_model: int = 512,
                      N: int = 6,
                      h: int =8,
                      dropout: float = 0.1,
                      d_ff = 2048
                      ):
    
    # Creating embedding layers
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)

    # Creating positional encoding layers
    src_position = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_position = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Creating the encoder blocks (x N):
    encoder_blocks = []
    for _ in range(N):
        # Needs to define these before the encoder block
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout) 
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        # The encoder block
        encoder_block = EncoderBlock(self_attention_block=encoder_self_attention_block, feed_forward_block=feed_forward_block, dropout=dropout)
        encoder_blocks.append(encoder_block)

    # Creating the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model=d_model, h=h ,dropout=dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model=d_model, h=h ,dropout=dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(self_attention_block=decoder_self_attention_block,
                                     cross_attention_block=decoder_cross_attention_block,
                                     feed_forward_block=feed_forward_block,
                                     dropout=dropout)
        decoder_blocks.append(decoder_block)

    # Creating encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating the projection layer
    projection_layer =  ProjectionLayer(d_model=d_model, vocab_size=tgt_vocab_size)

    # Creating the transformer with the Transformer class
    transformer = Transformer(encoder=encoder,
                              decoder=decoder,
                              src_embedding=src_embedding,
                              tgt_embedding=tgt_embedding,
                              src_position=src_position,
                              tgt_position=tgt_position,
                              projection_layer=projection_layer)
    
    # Parameter initialization
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer