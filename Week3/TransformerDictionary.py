import os
import pickle


class TransformerDictionary:
    def __init__(self, name, path_to_dictionaries="dictionaries"):
        self.name = name

        # Get the absolute path of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script

        # Use os.path.join to ensure the dictionaries path is correctly resolved
        self.path_to_dictionaries = os.path.join(script_dir, path_to_dictionaries)

        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")

        # Init dictionary
        self.dictionary = {}
        if os.path.exists(dictionary_path):
            print(f"Dictionary {dictionary_path} exists. Loading")
            self.load()
        else:
            print(f"The file {dictionary_path} does not exist. Creating")
            self.save()
        self.reverse_dictionary = {v: k for k, v in self.dictionary.items()}

    def learn_word(self, word):
        if word not in self.dictionary:
            self.dictionary[word] = len(self.dictionary) + 1
        return self.dictionary[word]

    def to_token(self, word):
        if word in self.dictionary:
            return self.dictionary[word]
        else:
            return self.learn_word(word)

    def to_word(self, token):
        if token in self.reverse_dictionary:
            return self.reverse_dictionary[token]
        raise RuntimeError("Word not found in dictionary")

    def save(self):
        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")
        with open(dictionary_path, "wb") as f:
            pickle.dump(self.dictionary, f)

    def load(self):
        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")
        with open(dictionary_path, "rb") as f:
            self.dictionary = pickle.load(f)
