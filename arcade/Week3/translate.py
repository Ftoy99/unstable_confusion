"""
To translate from english to made up language defined in main

1) Take input english
2) Convert to tokens
3) feed to model
4) take output tokens and convert to made up language


Need 2 dictionaries
1 for english
1 for madeup language
"""
import nltk

from arcade.Week3.TransformerDictionary import TransformerDictionary
from arcade.Week3.prepare_dataset import get_gutenberg_sentence


def main():
    english_dictionary = TransformerDictionary(name="english")
    made_up_dictionary = TransformerDictionary(name="made_up")


if __name__ == '__main__':
    main()
