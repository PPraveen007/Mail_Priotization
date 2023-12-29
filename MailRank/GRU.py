# %%
from analysis import generate_report, save_model
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('preprocessed_data.csv')
df['content'].fillna('', inplace=True)
df['heading'].fillna('', inplace=True)
df['sender_name'].fillna('', inplace=True)
df['target'].fillna(0, inplace=True)

print(f"{df=}")
print(f"{df.axes=}")
X = df['content']
y = df['target']
# %%
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)

num_features_to_select = 1000
selector = SelectKBest(chi2, k=num_features_to_select)
X_selected = selector.fit_transform(X_tfidf, y)

# Convert the sparse matrix to a NumPy array
X_np = X_selected.toarray()
# %%

# Reshape features for GRU input
X_gru = X_np.reshape(X_np.shape[0], 1, X_np.shape[1])

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_gru, y_encoded, test_size=0.2, random_state=20)

# Build GRU model
model = Sequential()
model.add(GRU(100, input_shape=(X_gru.shape[1], X_gru.shape[2])))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# %%

# Evaluate on the test set
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Decode labels if needed
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
# %%

# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test_decoded, y_pred_decoded))
print("Classification Report:\n", classification_report(
    y_test_decoded, y_pred_decoded))

# %%
generate_report(y_test_decoded, y_pred_decoded, "GRU")
