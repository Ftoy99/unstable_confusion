import torch


class EMA:
    def __init__(self, model, beta=0.999):
        self.model = model
        self.beta = beta
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                self.shadow[name] = self.beta * self.shadow[name] + (1.0 - self.beta) * param.detach()

    def apply(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = self.shadow[name]

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                param.data = self.shadow[name]