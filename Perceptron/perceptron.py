import numpy as np

class Perceptron:

    def __init__(self, learning_rate = 0.1, epochs = 1000):

        self.learning_rate = learning_rate
        self.epochs = epochs

        self.coeff_        = None
        self.y_intercept_  = None

    def step(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples, n_features = X.shape


        self.coeff_        = np.zeros(n_features)
        self.y_intercept_  = 0.0

        for epoch in range(self.epochs):
            for idx in range(n_samples):
                
                z = np.dot(X[idx], self.coeff_) + self.y_intercept_
                y_pred = self.step(z)

                gw =  (y[idx] - y_pred) * X[idx]
                gb =  (y[idx] - y_pred)
               
                self.coeff_       += self.learning_rate * gw
                self.y_intercept_ += self.learning_rate * gb

        return self.coeff_, self.y_intercept_
    
    def predict(self, X):
        X = np.asarray(X)
        z = np.dot(X, self.coeff_) + self.y_intercept_
        return np.array([self.step(val) for val in z])
