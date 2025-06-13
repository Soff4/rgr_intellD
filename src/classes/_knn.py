import numpy as np
import joblib
from collections import Counter
from sklearn.metrics import classification_report, accuracy_score
from scipy.spatial.distance import cosine

# K-Nearest Neighbors (KNN) — це алгоритм класифікації (або регресії), який:
#   - нічого не вчить на етапі "навчання";
# а при класифікації нового прикладу:
#   - вимірює відстань до всіх прикладів з тренувальної вибірки;
#   - обирає k найближчих сусідів;
#   - і визначає клас нового прикладу за голосуванням більшості.
# Його називають алгоритмом пам’яті (instance-based learning).

# Ми працюємо з TF-IDF векторами, для них найбільш відстань — косинусна.
# 📐 Косинусна відстань:
# Це не "відстань" у прямому сенсі, а міра відмінності напрямку двох векторів:

# cosine_similarity(A, B) = A * B / |A| * |B|

# А відстань:

# cosine_distance(A, B) = 1 - cosine_similarity(A, B)

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = np.ravel(y)

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        distances = []
        for i in range(self.X_train.shape[0]):
            vec = self.X_train[i]
            a = x.ravel() if isinstance(x, np.ndarray) else x.toarray().ravel()
            b = vec.toarray().ravel()

            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                dist = 1.0  # максимально далека відстань
            else:
                dist = cosine(a, b)

            distances.append((dist, self.y_train[i]))

        nearest = sorted(distances, key=lambda d: d[0])[:self.k]
        nearest_labels = [label for _, label in nearest]

        most_common = Counter(nearest_labels).most_common(1)[0][0]
        return most_common

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = joblib.load('models/train_test_data.pkl')

    # Створюємо і тренуємо власний KNN
    model = KNN(k=5)
    model.fit(X_train, y_train)

    # Прогноз
    y_pred = model.predict(X_test)

    # Оцінка
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Зберігаємо модель (опціонально)
    joblib.dump(model, 'models/knn_custom.pkl')
