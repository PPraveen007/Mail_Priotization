# %%
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('preprocessed_data.csv')

# %%
# creating a bag of words model
cv = CountVectorizer()
X = cv.fit_transform(data['heading']).toarray()
y = data['target'].values

# 3.1.2. TF-IDF
# creating a TF-IDF model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(X).toarray()
print(f"{X=}")

# 3.1.3. Word Embeddings
embeddings_index = {}
with open('glove.6B.100d.txt', encoding='utf8') as f:
    for line in f:
        values = line.split()
        # The first value is the word, the rest are the values of the embedding
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')  # Load embedding
        # Add embedding to our embedding dictionary
        embeddings_index[word] = coefs
print('Found %s word vectors.' % len(embeddings_index))

# creating a word embeddings matrix
embedding_matrix = np.zeros((len(embeddings_index), 100))
for i, word in enumerate(embeddings_index):
    embedding_vector = embeddings_index[word]
    embedding_matrix[i] = embedding_vector

# creating a word embeddings model
model = Sequential()
model.add(Embedding(len(embeddings_index), 100, weights=[
          embedding_matrix], input_length=1000, trainable=False))
model.compile('rmsprop', 'mse')
