# %%
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
df = pd.read_csv('data.csv')

print("Original Data:")
print(df.head())


def clean_text(text):
    if pd.isnull(text):
        return ""
    text = str(text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text, re.I | re.A)
    text = text.lower()
    return text


for col_name in df.columns.drop(['target', 'sender_email']):
    df[col_name] = df[col_name].apply(clean_text)
    df[col_name] = df[col_name].apply(word_tokenize)
    stop_words = set(stopwords.words('english'))

    def remove_stopwords(tokens):
        return [word for word in tokens if word.lower() not in stop_words]

    df[col_name] = df[col_name].apply(remove_stopwords)

# %%
df = pd.read_csv('preprocessed_data.csv')


stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def preprocess_text(tokens):
    if pd.isna(tokens):
        return ""
    tokens = tokens.replace('[', '')
    tokens = tokens.replace(']', '')
    tokens = tokens.replace('\'', '')
    tokens = tokens.replace(',', '')
    tokens = tokens.split(' ')
    # print(f"{tokens=}")

    stemmed_words = [stemmer.stem(word) for word in tokens]
    # print(f"{stemmed_words=}")

    lemmatized_words = [lemmatizer.lemmatize(word) for word in stemmed_words]
    # print(f"{lemmatized_words=}")

    preprocessed_text = ' '.join(lemmatized_words)
    # print(f"{preprocessed_text=}")

    return preprocessed_text


columns_to_preprocess = df.columns.drop(['target', 'sender_email'])

for column in columns_to_preprocess:
    df[column] = df[column].apply(preprocess_text)


def remove_braces(txt):
    # print(f"{txt=}")
    # print(f"{type(txt)=}")
    return str(txt).replace('<', '').replace('>', '')


df['sender_email'] = df['sender_email'].apply(remove_braces)
# df = df.fillna("null")
df.to_csv('preprocessed_data.csv', index=False)
print("Preprocessed Data is written to 'preprocessed_data.csv'")
