# Attention is all you need
from torch import nn


class AIAYN(nn.Module):
    """Transformer class as described in Attention is all you need"""

    def __init__(self, input_dictionary_size, output_dictionary_size, embedding_dim_size=32):
        super(AIAYN, self).__init__()
        # Embeddings transform tokens to vectors
        self.input_embedding = nn.Embedding(num_embeddings=input_dictionary_size, embedding_dim=embedding_dim_size)
        self.output_embedding = nn.Embedding(num_embeddings=output_dictionary_size, embedding_dim=embedding_dim_size)

        self.encoder = Encoder(embedding_dim_size=embedding_dim_size)
        self.decoder = Decoder(embedding_dim_size=embedding_dim_size)

        self.output_linear = nn.Linear(embedding_dim_size, output_dictionary_size)

    def forward(self, source, target):
        # Convert to embeddings
        source_embedding = self.input_embedding(source)
        target_embedding = self.output_embedding(target)

        # Pass through encoder
        memory = self.encoder(source_embedding)

        # Pass through decoder with encoder memory
        output = self.decoder(target_embedding, memory)

        return self.output_linear(output)


class Encoder(nn.Module):
    """Encoder class for the AIAYN Transformer"""

    def __init__(self, embedding_dim_size=32):
        super(Encoder, self).__init__()

    def forward(self, x):
        return x


class Decoder(nn.Module):
    """Decoder class for the AIAYN Transformer"""

    def __init__(self, embedding_dim_size=32):
        super(Decoder, self).__init__()

    def forward(self, x, mem):
        return x
