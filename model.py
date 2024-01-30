import tensorflow as tf
from tensorflow.keras.layers import Embedding, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Fetch data from /kaggle/input/spam-mails-dataset/spam_ham_dataset.csv
data = pd.read_csv("spam_ham_dataset.csv")

# Separate texts and labels in variables
texts = data["text"].values
labels = data["label_num"].values
# Preparation of our dataset with Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

max_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_length)

# Separate 80% for train and 20% for test
test = int(len(sequences)*0.2)
np.random.seed(42)

indices = list(range(len(sequences)))

train_indices = indices[test:]
test_indices = indices[:test]

x_train = np.array([sequences[i] for i in train_indices])
y_train = np.array([labels[i] for i in train_indices])

x_test = np.array([sequences[i] for i in test_indices])
y_test = np.array([labels[i] for i in test_indices])


# Create our Model
class Net(tf.keras.Model):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=32, input_length=max_length)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(1, activation="sigmoid")
        
    def call(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x
    
model = Net()
# Config our loss function and optimizer
bce = tf.keras.losses.BinaryCrossentropy()
adam = tf.keras.optimizers.Adam()
# Compile our model for training and test
model.compile(
    optimizer=adam,
    loss=bce,
    metrics=["accuracy"]
)
# Train and test our model
result = model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))