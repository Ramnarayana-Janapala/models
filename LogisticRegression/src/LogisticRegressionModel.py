import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_1 = None
        self.theta_0 = None
        self.cost_per_epochs = []

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """Train the logistic regression model using gradient descent."""
        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.theta_1 = np.zeros(n_features)
        self.theta_0 = 0


        # Gradient descent
        for _ in range(self.epochs):
            cost = 0
            # Linear model
            linear_model = np.dot(X, self.theta_1) + self.theta_0
            # Apply sigmoid function
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients for weight
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            #Compute the gradient for bias
            db = (1 / n_samples) * np.sum(y_predicted - y)
            cost = -1/n_samples * np.sum(y * np.log(y_predicted) + (1-y) * np.log(1- y_predicted))
            # Update weights and bias
            self.theta_1 -= self.learning_rate * dw
            self.theta_0 -= self.learning_rate * db
            self.cost_per_epochs.append(cost)
    def predict(self, X):
        """Make predictions using the trained model."""
        linear_model = np.dot(X, self.theta_1) + self.theta_0
        y_predicted = self.sigmoid(linear_model)
        # Convert probabilities to binary class labels, boundary line is 0.5
        return np.where(y_predicted >= 0.5, 1, 0)




X, Y = make_classification(n_samples=1000, n_features=2, n_informative=2,
                           n_redundant=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression(learning_rate=0.1, epochs=1000)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
print(model.cost_per_epochs)

plt.figure()
plt.plot(X_test,y_pred,"o",color='red')
plt.plot(X_test,y_test,"*",color='green')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

plt.figure()
plt.plot(model.cost_per_epochs,"o",color='red')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
