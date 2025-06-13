import numpy as np
from collections import defaultdict
import joblib
import math

# Наївний Байєс — це ймовірнісна модель для класифікації, яка базується на формулі Баєса:
# P(C | X) = P(X | C) * P(C) / P(X)
# В нашому випадку ми рахуємо лише чисельник, оскільки P(X) однаковий для всіх класів.
# P(X | C) * P(C) — ймовірність класу C за умови, що ознаки X спостерігались.
# Модель припускає, що всі слова незалежні одне від одного.


# У випадку нашої задачі це викладатиме так:
# X — набір слів у повідомленні
# C — клас (spam / ham)

class NaiveBayes:
    def __init__(self):
        # Зберігає log-імовірності класів (P(C))
        self.class_log_prior = {}

        # Зберігає log-імовірності ознак (слів) у кожному класі (P(w|C))
        self.feature_log_prob = {}

        # Список можливих класів (наприклад, [0, 1])
        self.classes = []

        # Лексикон (можна розширити)
        self.vocab = set()

        # Кількість слів у кожному класі (матриця)
        self.word_counts = {}

        # Кількість документів у кожному класі
        self.class_counts = {}

    def fit(self, X, y):
        """
        Навчає модель:
        - підраховує ймовірності кожного класу
        - підраховує ймовірність кожного слова у кожному класі
        Працює з матрицею X (схожою на tf-idf або count-вектором)
        """
        self.classes = np.unique(y)
        n_docs, n_features = X.shape

        for cls in self.classes:
            idx = np.where(np.asarray(y).ravel() == cls)[0]
            self.class_counts[cls] = len(idx)
            self.word_counts[cls] = X[idx].sum(axis=0)

        for cls in self.classes:
            self.class_log_prior[cls] = math.log(self.class_counts[cls] / n_docs)

            total_words = self.word_counts[cls].sum()
            probs = (self.word_counts[cls] + 1) / (total_words + n_features)
            self.feature_log_prob[cls] = np.log(probs)

    def predict(self, X):
        """
        Повертає прогнозовані класи для кожного прикладу
        """
        return [self._predict_single(x) for x in X]
    
    def _predict_single(self, x):
        """
        Обчислює лог-оцінку для кожного класу
        і повертає той, де вона найбільша
        """
        scores = {}
        for cls in self.classes:
            score = self.class_log_prior[cls]
            score += x.dot(self.feature_log_prob[cls].T).item()
            scores[cls] = score
        return max(scores, key=scores.get)
        
if __name__ == "__main__":
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

    X_train, y_train, X_test, y_test = joblib.load('models/train_test_data.pkl')

    model = NaiveBayes()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'models/naive_bayes_custom.pkl')