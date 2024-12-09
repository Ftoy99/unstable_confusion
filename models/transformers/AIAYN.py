# Attention is all you need
from torch import nn


class AIAYN(nn.Module):
    """Transformer class as described in Attention is all you need"""
    def __init__(self):
        super(AIAYN, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Encoder(nn.Module):
    """Encoder class for the AIAYN Transformer"""
    def __init__(self):
        super(Encoder, self).__init__()


class Decoder(nn.Module):
    """Decoder class for the AIAYN Transformer"""
    def __init__(self):
        super(Decoder, self).__init__()
