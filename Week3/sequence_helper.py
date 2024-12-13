import torch


def pad_sequences(sequence, max_length, padding_value,device):
    return torch.tensor(
        [seq + [padding_value] * (max_length - len(seq)) for seq in sequence],
        dtype=torch.int64
    ).to(device)