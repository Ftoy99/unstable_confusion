import torch
from torch import nn, optim

from .TransformerDictionary import TransformerDictionary
from .prepare_dataset import get_gutenberg_generator
from .translate import translate_functional
from models import AIAYN
import random

english_dictionary = TransformerDictionary(name="english")
made_up_dictionary = TransformerDictionary(name="made_up")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "arcade/Week3/weights/AIAYN.pth"  # The file where the model will be saved


def save_model(model):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved")


def load_model(model):
    try:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("No saved model found. Starting from scratch.")


def dataset_generator():
    for sentence in get_gutenberg_generator():
        tokenized_sentence = [english_dictionary.to_token(x.lower()) for x in sentence]
        func_translation = [made_up_dictionary.to_token(x) for x in translate_functional(sentence)]
        output_sequence = [0 for _ in sentence]  # Replace with your actual logic
        yield tokenized_sentence, output_sequence, func_translation


def create_batches(generator, batch_size):
    batch = []
    for item in generator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch  # Yield a full batch
            batch = []
    if batch:  # Yield the last batch if it's not empty
        yield batch


def pad_batch(batch, padding_value=0, device=device):
    input_sequences = [item[0] for item in batch]
    output_sequences = [item[1] for item in batch]
    predicted_sequences = [item[2] for item in batch]

    # Determine the max length for padding
    max_length = max(
        max(len(seq) for seq in input_sequences),
        max(len(seq) for seq in predicted_sequences)
    )

    def pad_sequences(sequences, max_length, padding_value):
        return torch.tensor(
            [seq + [padding_value] * (max_length - len(seq)) for seq in sequences],
            dtype=torch.int64
        ).to(device)

    in_tensor = pad_sequences(input_sequences, max_length, padding_value)
    out_tensor = pad_sequences(output_sequences, max_length, padding_value)
    predicted_tensor = pad_sequences(predicted_sequences, max_length, padding_value)

    return in_tensor, out_tensor, predicted_tensor


def train():
    model = AIAYN(input_dictionary_size=len(english_dictionary.dictionary) + 1,
                  output_dictionary_size=len(made_up_dictionary.dictionary)).to(device)

    # Load the model if it exists
    load_model(model)

    criterion = nn.CrossEntropyLoss()  # Common loss function for sequence-to-sequence tasks
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 100
    num_epochs = 10
    data_gen = dataset_generator()

    subset = 100
    batches = []
    print("Preparing Batches")
    for batch in create_batches(data_gen, batch_size):
        batches.append(batch)
    random.shuffle(batches)
    batches = batches[:subset]
    print("Done with Batches")

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for index, batch in enumerate(batches):
            print(f"Processing batch {index}")
            # Pad the current batch
            in_tensor, out_tensor, predicted_tensor = pad_batch(batch)

            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            output = model(in_tensor, out_tensor)

            output_flat = output.view(-1, output.shape[-1])
            predicted_tensor_flat = predicted_tensor.view(-1)

            loss = criterion(output_flat, predicted_tensor_flat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (len(batch) * batch_size)}")
        save_model(model)


if __name__ == '__main__':
    train()
