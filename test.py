import numpy as np
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Path to the text file
file_path = r"D:\Summer Research\translation_new.txt"

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize the text into individual characters and ignore spaces and \n
char_tokens = [char for char in text if char != ' ' and char != '\n']

# Convert list of characters back to a string for Tokenizer
char_text = ''.join(char_tokens)

# Use Keras Tokenizer to convert characters to sequences of numbers
tokenizer = Tokenizer(char_level=True)  # Set char_level=True for character-level tokenization
tokenizer.fit_on_texts([char_text])
print("Vocabulary:", tokenizer.word_index)

# Convert tokens to sequences of numbers
sequences = tokenizer.texts_to_sequences([char_text])[0]

# Prepare sequences for the model
sequence_length = 10
X = []
Y = []

for i in range(sequence_length, len(sequences)):
    X.append(sequences[i-sequence_length:i])  # Sequence of characters
    Y.append(sequences[i])                    # Next character (the target)

# Convert lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

# Build the LSTM model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size (including padding token)

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=sequence_length))  # Embedding layer
model.add(LSTM(512))  # LSTM layer without activation function
model.add(Dense(vocab_size, activation='softmax'))  # Output layer for classification

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(X, Y, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X)

# Convert predictions to indices
predicted_indices = np.argmax(predictions, axis=-1)

# Convert indices to characters
predicted_chars = [tokenizer.index_word.get(idx, '') for idx in predicted_indices]

print("Predicted Characters:")
print(predicted_chars[:10])  # Display first 10 predictions

   