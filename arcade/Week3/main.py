"""
Sequence to sequence with an implementation of transformed ->
Translate from english to a made up simple language that 1 word in english may consist 2 in madeupLanguage words
For made up language use


https://jujutsu-kaisen.fandom.com/wiki/Toge_Inumaki#cite_note-vol0-23
Toge's Safe Words:
Salmon (しゃけ Shake?) — Used for affirmation.[23]
Fish Flakes (おかか Okaka?) — Used for negation, and usually negative.[23]
Kelp (こんぶ Konbu?) — Used as a greeting.[23]
Mustard leaf (たかな Takana?) — Used to show concern/worry, "I'm on it".
Salmon Roe (すじこ Sujiko?) — "Well, well" or "My, my".
Caviar (いくら Ikura?) — Used as an expletive/curse word.
Spicy Cod Roe (めんたいこ Mentaiko?) — Used motivationally.
Tuna (ツナ Tsuna?), Tuna Tuna (ツナツナ Tsuna Tsuna?) — Used to call attention to something, "look".
Tuna Mayo (ツナマヨ Tsuna Mayo?) — Used as general talk, "do something" (about the situation).
Everything else is gibberish.[23]

Going to set rules on how these words are used in english gonna keep punctuations as is or may cut words after a specific length so they dont appear eg 13letters words -> none


eg.
3 Letter words -> Caviar
4 Letter words -> Salmon ...


Allowed Words

# Unique
Fish Flakes
Kelp
Mustard leaf
Caviar

# Duplicate salmor / roe
Salmon
Spicy Cod Roe
Salmon Roe

# Duplicate Tuna
Tuna
Tuna Mayo


Uniques
Tuna
Mayo
Salmon
Spicy
Cod
Roe
Fish
Flakes
Kelp
Mustard
Leaf
Caviar
"""
import torch
from torch import Tensor, dtype

from arcade.Week3.TransformerDictionary import TransformerDictionary
from models.transformers import AIAYN
from prepare_dataset import get_gutenberg_sentence


def main():
    # Dictionaries to convert to tokens
    english_dictionary = TransformerDictionary(name="english")
    made_up_dictionary = TransformerDictionary(name="made_up")

    model = AIAYN(input_dictionary_size=100000, output_dictionary_size=10)

    for sentence in get_gutenberg_sentence():
        sentence = [english_dictionary.to_token(x.lower()) for x in sentence]
        in_tensor = torch.tensor(sentence, dtype=torch.int64)
        out_tensor = torch.tensor([],dtype=torch.int64)
        x = model(in_tensor, out_tensor)
        print(x)
    # english_dictionary.save()


if __name__ == '__main__':
    main()
