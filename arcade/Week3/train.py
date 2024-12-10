import torch
from torch import nn, optim

from arcade.Week3.TransformerDictionary import TransformerDictionary
from arcade.Week3.prepare_dataset import get_gutenberg_sentence
from arcade.Week3.translate import translate_functional
from models import AIAYN


def train ():
    english_dictionary = TransformerDictionary(name="english")
    made_up_dictionary = TransformerDictionary(name="made_up")
    model = AIAYN(input_dictionary_size=100000, output_dictionary_size=10)
    criterion = nn.CrossEntropyLoss()  # Common loss function for sequence-to-sequence tasks
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for sentence in get_gutenberg_sentence():  # or use your batch generator
            # Tokenize and create tensors
            tokenized_sentence = [english_dictionary.to_token(x.lower()) for x in sentence]
            in_tensor = torch.tensor(tokenized_sentence, dtype=torch.int64).unsqueeze(0)  # Add batch dimension
            out_tensor = torch.tensor([0 for _ in sentence], dtype=torch.int64).unsqueeze(0)  # Example output sequence
            func_translation = [made_up_dictionary.to_token(x) for x in translate_functional(sentence)]
            predicted_tensor = torch.tensor(func_translation,dtype=torch.int64).unsqueeze(0)

            # Forward pass
            optimizer.zero_grad()  # Clear previous gradients
            output = model(in_tensor, out_tensor)

            # Compute the loss (we only consider the output sequence, not padding)
            loss = criterion(output.view(-1, output.shape[-1]), predicted_tensor.view(-1))  # Flatten for cross-entropy loss

            # Backpropagation and optimization step
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(get_gutenberg_sentence())}")



if __name__ == '__main__':
    train()