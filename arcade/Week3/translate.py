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
from arcade.Week3.TransformerDictionary import TransformerDictionary

len_to_words = {
    2: ["kelp"],
    3: ["fish", "flakes"],
    4: ["salmon"],
    5: ["mustard", "leaf"],
    6: ["salmon", "roe"],
    7: ["caviar"],
    8: ["tuna"],
    9: ["tuna", "mayo"],
    10: ["spicy", "cod", "roe"]
}


def main():
    english_dictionary = TransformerDictionary(name="english")
    made_up_dictionary = TransformerDictionary(name="made_up")


def translate_functional(sentence):
    """
    2 -> Kelp
    3 -> Fish Flakes
    4 -> Salmon
    5 -> Mustard leaf
    6- > Salmon Roe
    7 -> Caviar
    8-> Tuna
    9 -> Tuna Mayo
    10 -> Spicy Cod Roe
    no tokens for everything else
    :param sentence:
    :return:
    """

    output = []
    for word in sentence:
        if len(word) in len_to_words:
            output.extend(len_to_words[len(word)])

    return output


if __name__ == '__main__':
    main()
