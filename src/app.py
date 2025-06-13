import streamlit as st
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from classes._mlp import MLP
from classes._knn import KNN
from classes._naive_bayes import NaiveBayes

# Заголовок
st.title("🔎 Виявлення спаму у повідомленнях")

# Вибір моделі
model_choice = st.selectbox("Оберіть модель:", ["MLP", "KNN", "Naive Bayes"])
model_file = f"models/{model_choice.lower().replace(' ', '_')}_custom.pkl"

# Завантаження моделі та векторизатора
model = joblib.load(model_file)
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Завантаження метрик
metrics = joblib.load("models/model_metrics.pkl")

# Вивід метрик
st.subheader("Метрики моделі:")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics[model_choice]['Accuracy']:.2%}")
col2.metric("Precision", f"{metrics[model_choice]['Precision']:.2%}")
col3.metric("Recall", f"{metrics[model_choice]['Recall']:.2%}")
col4.metric("F1-score", f"{metrics[model_choice]['F1-score']:.2%}")

# Ввід користувача
st.subheader("Перевірка повідомлення:")
user_input = st.text_area("Введіть текст повідомлення:")

if st.button("Аналізувати"):
    if user_input.strip() == "":
        st.warning("Будь ласка, введіть текст повідомлення.")
    else:
        X_input = vectorizer.transform([user_input]).toarray()
        prediction = model.predict(X_input)[0]
        label = "СПАМ 🚫" if prediction == 1 else "НЕ СПАМ ✅"
        st.success(f"Результат: {label}")

# Побудова Confusion Matrix
st.subheader("Матриця неточностей:")

try:
    X_train, y_train, X_test, y_test = joblib.load("models/train_test_data.pkl")
    if hasattr(model, "predict"):
        if hasattr(X_test, "toarray"):  # для MLP
            y_pred = model.predict(X_test.toarray())
        else:
            y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        plt.xlabel("Прогноз")
        plt.ylabel("Справжнє")
        st.pyplot(fig)
except Exception as e:
    st.warning(f"⚠️ Немає даних для побудови матриці: {e}")

# Кнопка "Тестувати"
if st.button("🧪 Тестувати на прикладах"):
    test_messages = [
        "Congratulations! You've won a free ticket to Bahamas. Click here to claim.",
        "Can we meet tomorrow to finish our assignment?",
        "You've been selected for a $500 reward. Call now.",
        "Don't forget mom's birthday gift.",
        "URGENT: Your bank account is blocked. Verify immediately."
    ]
    for msg in test_messages:
        X_test = vectorizer.transform([msg]).toarray()
        pred = model.predict(X_test)[0]
        res = "СПАМ 🚫" if pred == 1 else "НЕ СПАМ ✅"
        st.write(f"**{msg}** → {res}")
