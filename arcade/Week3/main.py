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
"""

from torch import Tensor

from models.transformers import AIAYN


def main():
    model = AIAYN(input_dictionary_size=100000, output_dictionary_size=10)


if __name__ == '__main__':
    main()
