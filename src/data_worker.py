import pandas as pd
import re
import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

df = pd.read_csv('data/spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({ 'ham': 0, 'spam': 1 })

def preprocess_text(text):
    text = text.lower() # Переводить текст у нижній регістр (щоб "Spam" = "spam")
    text = re.sub(r'\d+', '', text) # Видаляє всі цифри (вони рідко мають значення у спамі)
    text = text.translate(str.maketrans('', '', string.punctuation)) # Видаляє розділові знаки
    text = text.strip() # Видаляє зайві пробіли на початку і в кінці
    return text

df['clean_text'] = df['text'].apply(preprocess_text)

def remove_stopwords(text):
    words = text.split() # Розбиває рядок на окремі слова
    filtered_words = [word for word in words if word not in stop_words] # Видаляє стоп-слова
    return ' '.join(filtered_words) # Склеює слова назад у рядок

df['clean_text'] = df['clean_text'].apply(remove_stopwords)

df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']
df['clean_text'] = df['clean_text'].astype(str)

df.to_csv('./data/cleaned_spam.csv', index=False)

