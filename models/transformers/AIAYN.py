# Attention is all you need
import math
import os

import torch
from torch import nn


class AIAYN(nn.Module):
    """Transformer class as described in Attention is all you need"""

    def __init__(self, input_dictionary_size, output_dictionary_size, embedding_dim_size=32, max_sentences=9999):
        super(AIAYN, self).__init__()
        # Embeddings transform tokens to vectors
        self.input_embedding = nn.Embedding(num_embeddings=input_dictionary_size, embedding_dim=embedding_dim_size)
        self.output_embedding = nn.Embedding(num_embeddings=output_dictionary_size, embedding_dim=embedding_dim_size)

        self.positional_encoding = PositionalEncoding(max_sentences=max_sentences)

        self.encoder = Encoder(embedding_dim_size=embedding_dim_size)
        self.decoder = Decoder(embedding_dim_size=embedding_dim_size)

        self.output_linear = nn.Linear(embedding_dim_size, output_dictionary_size)
        # self.soft_max = nn.Softmax(dim=-1)

    def forward(self, source, target):
        # Convert to embeddings
        source_embedding = self.input_embedding(source)
        target_embedding = self.output_embedding(target)

        # Add Positional Embeddings
        source_embedding = self.positional_encoding(source_embedding)
        target_embedding = self.positional_encoding(target_embedding)

        # Pass through encoder
        memory = self.encoder(source_embedding)

        # Pass through decoder with encoder memory
        output = self.decoder(target_embedding, memory)
        output = self.output_linear(output)
        # output = self.soft_max(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim_size=32, max_sentences=9999):
        super(PositionalEncoding, self).__init__()
        # For even = sin(pos/10000^(2*i/dmodel))
        # For odd = cod(pos/10000^(2*i/dmodel))

        # For each embedding dimension i we add the encoding meaning 0,1 -> odd encoding ->2 even 1,0 even
        pe = torch.zeros(max_sentences, embedding_dim_size)
        position = torch.arange(0, max_sentences, dtype=torch.float).unsqueeze(1)

        # TODO
        div_term = torch.exp(
            torch.arange(0, embedding_dim_size, 2).float() * (-math.log(10000.0) / embedding_dim_size)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension and register as a buffer (not a parameter to be learned)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, embedding):
        # Embedding shape: [batch_size, seq_len, embedding_dim_size]
        seq_len = embedding.size(1)  # The second dimension is the sequence length
        # Add positional encoding to embeddings (ensure seq_len matches)
        output = embedding + self.pe[:, :seq_len, :]
        return output


class Encoder(nn.Module):
    """Encoder class for the AIAYN Transformer"""

    def __init__(self, embedding_dim_size=32):
        super(Encoder, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(num_heads=4, embed_dim=embedding_dim_size)

        self.normalization1 = nn.LayerNorm(embedding_dim_size)
        self.normalization2 = nn.LayerNorm(embedding_dim_size)

        # 3.3 feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim_size, embedding_dim_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim_size * 4, embedding_dim_size)
        )

    def forward(self, embedding):
        # Attention -> outpus Attention -> weights
        attn, _ = self.multi_head_attention(embedding, embedding, embedding)

        # Add & Norm
        attn_embedding = self.normalization1(embedding + attn)

        # Feed Forward
        output = self.feed_forward(attn_embedding)

        # Add & Norm
        output = self.normalization2(attn_embedding + output)

        return output


class Decoder(nn.Module):
    """Decoder class for the AIAYN Transformer"""

    def __init__(self, embedding_dim_size=32):
        super(Decoder, self).__init__()
        self.masked_multi_head_attention = nn.MultiheadAttention(num_heads=4, embed_dim=embedding_dim_size)
        self.multi_head_attention = nn.MultiheadAttention(num_heads=4, embed_dim=embedding_dim_size)

        self.normalization1 = nn.LayerNorm(embedding_dim_size)
        self.normalization2 = nn.LayerNorm(embedding_dim_size)
        self.normalization3 = nn.LayerNorm(embedding_dim_size)

        # 3.3 feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim_size, embedding_dim_size * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim_size * 4, embedding_dim_size)
        )

    def forward(self, embedding, memory):
        # Embedding is the input , memory is the output from encoder

        seq_len = embedding.size(0)
        mask = self.generate_mask(seq_len)  # Create the mask for the self-attention

        # masked multi head attn
        masked_attn, _ = self.masked_multi_head_attention(embedding, embedding, embedding, attn_mask=mask)
        out1 = self.normalization1(masked_attn + embedding)

        out1 = out1.transpose(0, 1)  # [seq_len, batch_size, embedding_dim_size]
        memory = memory.transpose(0, 1)  # [seq_len, batch_size, embedding_dim_size]
        out1_attn, _ = self.multi_head_attention(memory, memory, out1)

        out2 = self.normalization2(out1_attn + out1)

        out3 = self.feed_forward(out2)
        output = self.normalization3(out3 + out2)

        return output

    def generate_mask(self, seq_len):
        """
        This makes mask that is triangle so first time max-1 are covered and by the end none are covered.


        Generate a mask for the self-attention in the decoder to prevent attending to future tokens.
        The mask will be a lower triangular matrix, where all values above the diagonal are True (masked).
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=0)  # Upper triangular matrix
        mask = mask == 0  # Invert: True for allowed positions, False for masked positions
        return mask

def save_model(path,model):
    absolute_path = os.path.abspath(path)
    torch.save(model.state_dict(), absolute_path)
    print(f"Model saved")


def load_model(path,model):
    try:
        absolute_path = os.path.abspath(path)
        model.load_state_dict(torch.load(absolute_path))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")