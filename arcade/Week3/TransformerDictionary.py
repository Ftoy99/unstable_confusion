import os
import pickle


class TransformerDictionary:
    def __init__(self, name, path_to_dictionaries="dictionaries"):
        self.name = name
        self.path_to_dictionaries = path_to_dictionaries
        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")

        # Init dictionary
        self.dictionary = {1: "hehe"}
        if os.path.exists(dictionary_path):
            print(f"Dictionary {dictionary_path} exists. Loading")
            self.load()
        else:
            print(f"The file {dictionary_path} does not exist. Creating")
            self.save()

    def learn_word(self, word):
        pass

    def save(self):
        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")
        with open(dictionary_path, "wb") as f:
            pickle.dump(self.dictionary, f)

    def load(self):
        dictionary_path = os.path.join(self.path_to_dictionaries, self.name + ".pkl")
        with open(dictionary_path, "rb") as f:
            self.dictionary = pickle.load(f)
