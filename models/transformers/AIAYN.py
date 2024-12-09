# Attention is all you need
from torch import nn


class AIAYN(nn.Module):

    def __init__(self):
        super(AIAYN, self).__init__()
        self.test: nn.Linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.test(x)
