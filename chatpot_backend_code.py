import json

from collections import OrderedDict, Counter

from pathlib import Path
from pprint import pprint

import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchtext
from torchtext.vocab import Vocab


# from torchtext.data.utils import get_tokenizer

import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


device = "gpu" if torch.cuda.is_available() else "cpu"

torch.rand(3, 5).to(device)
print(device)

# def Tokenize_and_Lemmatize(sentence, ignore_words=["?", "!"]):
#     return [
#         lemmatizer.lemmatize(word.lower())
#         for word in nltk.word_tokenize(sentence)
#         if word not in ignore_words
#     ]


# path_to_file = Path(__file__).parent

# intents_wd = f"{path_to_file}/intents3.json"

# intents = []
# intents_file = open(intents_wd).read()
# intents = json.loads(intents_file)


# # tokenizer = get_tokenizer("basic_english")
# words = []
# documents = []
# classes = []

# for intent in intents["intents"]:
#     for pattern in intent["patterns"]:
#         # words.extend(Tokenize_and_Lemmatize(pattern))
#         # w = tokenizer(pattern)
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)

#         documents.append((w, intent["tag"]))
#         if intent["tag"] not in classes:
#             classes.append(intent["tag"])


# ignore_words = ["?", "!"]
# words = [
#     lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_words
# ]
# words = sorted(list(set(words)))
# # sort classes
# classes = sorted(list(set(classes)))


# print(len(documents), "documents")
# # classes = intents
# print(len(classes), "classes", classes)
# # words = all words, vocabulary
# print(len(words), "unique lemmatized words", words)

# pickle.dump(words, open("words.pkl", "wb"))
# pickle.dump(classes, open("classes.pkl", "wb"))

# # Create empty lists for training data
# train_x = []
# train_y = []

# for doc in documents:
#     bag = []
#     output_row = [0] * len(classes)

#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

#     # Create the bag of words
#     for word in words:
#         bag.append(1) if word in pattern_words else bag.append(0)

#     # Set the appropriate output value for the current tag
#     output_row[classes.index(doc[1])] = 1

#     # Append the bag of words and output row to the training data lists
#     train_x.append(bag)
#     train_y.append(output_row)

# train_x = np.array(train_x)
# train_y = np.array(train_y)

# print("Training data created")


# # Define Dataset
# class IntentDataset(Dataset):
#     def __init__(self, examples):
#         self.examples = examples

#     def __len__(self):
#         return len(self.examples)

#     def __getitem__(self, idx):
#         return self.examples[idx]


# # Define the tokenizer
# # tokenizer = get_tokenizer("basic_english")

# # Example sentence
# # sentence = "PyTorch is a deep learning framework."

# # Tokenize the sentence


# # tokens = tokenizer(sentence)
# # print(tokens)


# # device = "gpu" if torch.cuda.is_available() else 'cpu'
# # arr = torch.rand(5,3).to(device)


# # def tokenize_and_lemmatize(sentence):
# #     return [lemmatizer.lemmatize(words.lower()) for words in nltk.word_tokenize(sentence)]

# # # TEXT = Field(tokenize=tokenize_and_lemmatize)
# # # LABEL = Field(sequential=False)

# # path_to_file = Path(__file__).parent
# # intents_wd = f"{path_to_file}/intents3.json"

# # intents = []
# # intents_file = open(intents_wd).read()
# # intents = json.loads(intents_file)

# # examples = []
# # for intent in intents['intents']:
# #     for pattern in intent['patterns']:
# #         examples.append(Example.fromlist([pattern, intent['tag']], fields=[('text', TEXT), ('label', LABEL)]))

# # print(examples)
