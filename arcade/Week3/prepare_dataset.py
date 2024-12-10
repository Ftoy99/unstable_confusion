import nltk
from nltk.corpus import gutenberg

from arcade.Week3.TransformerDictionary import TransformerDictionary


def nltk_gui():
    nltk.download_gui()


def get_gutenberg_id():
    for id in gutenberg.fileids():
        yield id


def get_gutenberg_sentence():
    for id in get_gutenberg_id():
        for sentence in gutenberg.sents(id):
            yield sentence


def prepare_made_up():
    english_dictionary = TransformerDictionary(name="english")
    made_up_dictionary = TransformerDictionary(name="made_up")
    for x in ["Tuna",
              "Mayo",
              "Salmon",
              "Spicy",
              "Cod",
              "Roe",
              "Fish",
              "Flakes",
              "Kelp",
              "Mustard",
              "Leaf",
              "Caviar"]:
        made_up_dictionary.learn_word(x.lower())
    made_up_dictionary.save()

    print(made_up_dictionary.dictionary)


if __name__ == '__main__':
    # prepare_made_up()
    # nltk_gui()
    for sentence in get_gutenberg_sentence():
        print(sentence)
