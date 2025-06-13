import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('data/cleaned_spam.csv')

# Векторизація, створення матриці TF-IDF
# TF (Term Frequency) — скільки разів слово зустрічається у повідомленні.
# IDF (Inverse Document Frequency) — наскільки це слово рідкісне серед усіх повідомлень.
# Перетворення очищенного тексту у числові вектори, які можуть бути подані на вхід ML/NN-моделям.
# Проблема в тому, що моделі не розуміють рядкові дані, тобто "Вигравай 50% зники..." - для моделі буде
# просто набір символів, для людей це спам. Тому потрібно перетворити текст у числові вектори.
# Тобто кожне слово після векторизації буде мати вагу TF-IDF, яка показує, наскільки важливе це слово.
df = df.dropna(subset=['clean_text'])
df['clean_text'] = df['clean_text'].astype(str)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# У цій частині коду ми зберігаємо векторизатор та дані для подальшого використання.
# Так нам потрібно було б кожного разу векторизувати дані перед тренуванням моделі,
# а після збереження ми можемо їх використовувати просто завантаживши.
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump((X_train, y_train, X_test, y_test), 'models/train_test_data.pkl')

