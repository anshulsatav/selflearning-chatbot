import random
import json
import pickle
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK files for tokenization and lemmatization
nltk.download('punkt')
nltk.download('wordnet')

# Initialize the lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file, which contains patterns and responses
intents = json.load(open('intents.json'))

# Initialize lists to store words, classes (intents), and documents (tokenized patterns)
words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',']  # Characters to ignore during tokenization

# Tokenize and lemmatize each pattern in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize the sentence into words
        tok_lem_sent = nltk.word_tokenize(pattern)
        # Extend the list of words with tokenized words
        words.extend(tok_lem_sent)
        # Append the tokenized sentence and its associated tag to the documents list
        documents.append((tok_lem_sent, intent['tag']))
        # Add the tag to the classes list if it's not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize each word and remove punctuation characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_chars]
# Remove duplicate words and sort the list
words = sorted(set(words))
# Sort the classes list
classes = sorted(set(classes))

# Create the 'chatbot' directory if it does not exist
if not os.path.exists('chatbot'):
    os.makedirs('chatbot')

# Save the processed words and classes to pickle files for later use
pickle.dump(words, open('chatbot/words.pkl', 'wb'))
pickle.dump(classes, open('chatbot/classes.pkl', 'wb'))

# Prepare the training data
training = []
output_empty = [0] * len(classes)  # Create an empty output vector for each class

# Convert each document into a bag-of-words format
for document in documents:
    bag = []
    word_patterns = document[0]
    # Lemmatize each word in the tokenized sentence
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    # Create a bag-of-words vector for the current sentence
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create the output vector with a 1 for the correct class and 0s for the rest
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Append the bag-of-words and output vector to the training data
    training.append([bag, output_row])

# Shuffle the training data to ensure randomness
random.shuffle(training)
# Convert the training data to a numpy array
training = np.array(training, dtype=object)

# Separate the training data into features (X) and labels (Y)
train_x = np.array([item[0] for item in training], dtype=np.float32)
train_y = np.array([item[1] for item in training], dtype=np.float32)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Input layer with 128 neurons and ReLU activation
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting
model.add(Dense(64, activation='relu'))  # Hidden layer with 64 neurons and ReLU activation
model.add(Dropout(0.5))  # Another dropout layer to prevent overfitting
model.add(Dense(len(train_y[0]), activation='softmax'))  # Output layer with softmax activation for multi-class classification

# Compile the model with categorical crossentropy loss and SGD optimizer
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model on the training data
model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=5, verbose=1)

# Save the trained model to a file
model.save('chatbot/chatbot_model.keras')
