import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib

class MLP:
    def __init__(self, input_size, hidden1_size=300, hidden2_size=150, learning_rate=0.01, epochs=200):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Ініціалізація ваг (He)
        self.W1 = np.random.randn(self.input_size, self.hidden1_size) * np.sqrt(2 / self.input_size)
        self.b1 = np.zeros((1, self.hidden1_size))
        self.W2 = np.random.randn(self.hidden1_size, self.hidden2_size) * np.sqrt(2 / self.hidden1_size)
        self.b2 = np.zeros((1, self.hidden2_size))
        self.W3 = np.random.randn(self.hidden2_size, 1) * np.sqrt(2 / self.hidden2_size)
        self.b3 = np.zeros((1, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, a):
        return a * (1 - a)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1, 1)
        class_weight = np.where(y == 1, 6.55, 1.0).reshape(-1, 1)

        for epoch in range(self.epochs):
            # Пряме поширення
            Z1 = X @ self.W1 + self.b1
            A1 = self.relu(Z1)

            Z2 = A1 @ self.W2 + self.b2
            A2 = self.relu(Z2)

            Z3 = A2 @ self.W3 + self.b3
            A3 = self.sigmoid(Z3)

            # Втрата
            loss = -np.mean(y * np.log(A3 + 1e-8) + (1 - y) * np.log(1 - A3 + 1e-8))
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss:.4f}")

            # Зворотне поширення
            dZ3 = (A3 - y) * class_weight
            dW3 = A2.T @ dZ3 / X.shape[0]
            db3 = np.mean(dZ3, axis=0, keepdims=True)

            dA2 = dZ3 @ self.W3.T
            dZ2 = dA2 * self.relu_deriv(Z2)
            dW2 = A1.T @ dZ2 / X.shape[0]
            db2 = np.mean(dZ2, axis=0, keepdims=True)

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * self.relu_deriv(Z1)
            dW1 = X.T @ dZ1 / X.shape[0]
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            # Оновлення ваг
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            self.W3 -= self.learning_rate * dW3
            self.b3 -= self.learning_rate * db3

    def predict(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = self.relu(Z2)
        Z3 = A2 @ self.W3 + self.b3
        A3 = self.sigmoid(Z3)
        return (A3 > 0.5).astype(int).flatten()


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = joblib.load('models/train_test_data.pkl')
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    model = MLP(input_size=X_train.shape[1], learning_rate=0.05, epochs=170)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, 'models/mlp_custom.pkl')
