import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchtext
from torchtext.data.utils import get_tokenizer

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


def tokenize_and_lemmatize(sentence):
    return [lemmatizer.lemmatize(words.lower()) for words in nltk.word_tokenize(sentence)]

# TEXT = Field(tokenize=tokenize_and_lemmatize)
# LABEL = Field(sequential=False)

path_to_file = Path(__file__).parent
intents_wd = f"{path_to_file}/intents3.json"

intents = []
intents_file = open(intents_wd).read()
intents = json.loads(intents_file)

examples = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        examples.append(Example.fromlist([pattern, intent['tag']], fields=[('text', TEXT), ('label', LABEL)]))

print(examples)

# path_to_file = Path(__file__).parent

# intents_wd = f"{path_to_file}/intents3.json"

# intents = []
# intents_file = open(intents_wd).read()
# intents = json.loads(intents_file)


# tokenizer = get_tokenizer("basic_english")
# words = []
# documents = []
# classes = []

# for intent in intents['intents']:
#     for pattern in intent["patterns"]:
#         w = tokenizer(pattern)
#         words.extend(w)

#         documents.append((w,intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])


# ignore_words = ['?', '!']
# lemmatizer = WordNetLemmatizer()
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]


print(words)

# Define the tokenizer
# tokenizer = get_tokenizer("basic_english")

# Example sentence
# sentence = "PyTorch is a deep learning framework."

# Tokenize the sentence


# tokens = tokenizer(sentence)
# print(tokens)



# device = "gpu" if torch.cuda.is_available() else 'cpu'
# arr = torch.rand(5,3).to(device)



