# %%
# Feature Extraction
# %%
from analysis import generate_report, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('preprocessed_data.csv')
df['content'].fillna('', inplace=True)
df['heading'].fillna('', inplace=True)
df['sender_name'].fillna('', inplace=True)
df['target'].fillna(0, inplace=True)

X = df['content']
print(f"{X=}")
y = df['target']
print(f"{y=}")

# %%
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf_vectorizer.fit_transform(X)
print(f"{X_tfidf=}")

num_features_to_select = 1000
selector = SelectKBest(chi2, k=num_features_to_select)
X_selected = selector.fit_transform(X_tfidf, y)

# %%
X_lstm = X_selected.toarray().reshape(
    X_selected.shape[0], 1, X_selected.shape[1])

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_lstm, y_encoded, test_size=0.2, random_state=42)

model = Sequential()
model.add(LSTM(100, input_shape=(X_lstm.shape[1], X_lstm.shape[2])))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# %%
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# %%
generate_report(y_test_decoded, y_pred_decoded, "LSTM")
