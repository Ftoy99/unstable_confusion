import torch
from torch import nn, optim

from arcade.Week3.sequence_helper import pad_sequences
from models.transformers.AIAYN import load_model, save_model
from arcade.Week3.TransformerDictionary import TransformerDictionary
from arcade.Week3.prepare_dataset import get_gutenberg_generator
from arcade.Week3.translate import translate_functional
from models import AIAYN
import random

english_dictionary = TransformerDictionary(name="english")
made_up_dictionary = TransformerDictionary(name="made_up")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "weights/AIAYN.pth"  # The file where the model will be saved


def dataset_generator():
    dataset = []
    for sentence in get_gutenberg_generator():
        tokenized_sentence = [english_dictionary.to_token(x.lower()) for x in sentence]
        func_translation = [made_up_dictionary.to_token(x) for x in translate_functional(sentence)]
        output_sequence = [0 for _ in sentence]  # Replace with your actual logic
        dataset.append((tokenized_sentence, output_sequence, func_translation))
    return dataset


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

    in_tensor = pad_sequences(input_sequences, max_length, padding_value, device)
    out_tensor = pad_sequences(output_sequences, max_length, padding_value, device)
    predicted_tensor = pad_sequences(predicted_sequences, max_length, padding_value, device)

    return in_tensor, out_tensor, predicted_tensor


def train():
    model = AIAYN(input_dictionary_size=len(english_dictionary.dictionary) + 1,
                  output_dictionary_size=len(made_up_dictionary.dictionary) + 1).to(device)
    torch.autograd.set_detect_anomaly(True)
    # Load the model if it exists
    load_model(model_path, model)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    batch_size = 100
    num_epochs = 10
    data_gen = dataset_generator()

    batches = []
    print("Preparing Batches")
    for sentence_pair in create_batches(data_gen, batch_size):
        batches.append(sentence_pair)
    print("Done with Batches")

    batches = batches[:100]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        random.shuffle(batches)
        for index, batch in enumerate(batches):
            # Pad the current batch
            in_tensor, out_tensor, predicted_tensor = pad_batch(batch)

            # Debug
            # print(in_tensor.shape)
            # print(out_tensor.shape)
            # print(predicted_tensor.shape)

            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            output = model(in_tensor, out_tensor)

            _, indices = torch.max(output, dim=-1)
            transformer_translation = indices.squeeze().tolist()
            transformer_translation = [made_up_dictionary.to_word(token) for token in transformer_translation]

            output_flat = output.view(-1, output.shape[-1])
            predicted_tensor_flat = predicted_tensor.view(-1)

            loss = criterion(output_flat, predicted_tensor_flat)

            # Backpropagation
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (len(batch) * batch_size)}")
        save_model(model_path, model)


if __name__ == '__main__':
    train()
