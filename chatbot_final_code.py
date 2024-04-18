import json
import pickle
import random

import nltk
import numpy as np
from keras import Model
from keras.layers import Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents3.json").read()
intents = json.loads(data_file)

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent["tag"]))

        # add to our classes list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])


# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open("chatbot_cache/words.pkl", "wb"))
pickle.dump(classes, open("chatbot_cache/classes.pkl", "wb"))

# Create empty lists for training data
train_x = []
train_y = []

# Iterate over each document in the 'documents' list
for doc in documents:
    # Initialize the bag of words and output row
    bag = []
    output_row = [0] * len(classes)

    # Tokenize and lemmatize the words in the pattern
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in doc[0]]

    # Create the bag of words
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # Set the appropriate output value for the current tag
    output_row[classes.index(doc[1])] = 1

    # Append the bag of words and output row to the training data lists
    train_x.append(bag)
    train_y.append(output_row)

# Convert the training data lists to NumPy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Training data created")


# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# fitting and saving the model
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose="1")
model.save("chatbot_cache/chatbot_model.h5")

print("model created")

chatbot_model: Model = load_model("chatbot_cache/chatbot_model.h5")  # type: ignore
assert isinstance(chatbot_model, Model)

intents = json.loads(open("intents3.json").read())
words = pickle.load(open("chatbot_cache/words.pkl", "rb"))
classes = pickle.load(open("chatbot_cache/classes.pkl", "rb"))


def clean_up_sentence(sentence: str) -> list[str]:
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model: Model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]["intent"]
    list_of_intents = intents_json["intents"]
    assert len(list_of_intents) > 0
    result = None
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res
