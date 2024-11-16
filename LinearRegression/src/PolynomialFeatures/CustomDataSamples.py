import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

def generate_data():
    np.random.seed(0)
    X = np.arange(-5,5,0.2) + 0.1 * np.random.normal(0,1,50)
    # Three degree polynomial
    Y = -1*(X**2) + 5*X + 13* (X**4) - 14*(X**3) - 24 + 10*np.random.normal(-1,1,50)
    print(f"Magnitude of X : {len(X)} and Y {len(Y)}")
    return X,Y


def generate_new_dimensions(X,Y,degree):
    X1 = X[:,np.newaxis]
    Y1 = Y[:, np.newaxis]
    polynomial_features = PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(X1)
    return x_poly,Y1

def polynomial_linear_regression(X, Y,degree=1):
    model = LinearRegression()
    model.fit(X,Y)
    y_predicate = model.predict(X)
    rmse = np.sqrt(mean_squared_error(Y,y_predicate))
    r2 = r2_score(Y,y_predicate)

    model.degree = degree
    model.RMSE = rmse
    model.r2= r2
    return model



def plot_results(X,Y,model,axis,degree):
    axis.plot(X[:, 1], model.predict(X),
              color=color[degree],
              label=str("Model Degree: %d" % model.degree)
                    + str("; RMSE:%.3f" % model.RMSE)
                    + str("; R2 Score: %.3f" % model.r2))
    axis.legend()

if __name__=="__main__":
    print(f'Start Model data generation...!')
    X,Y = generate_data()
    plt.figure(figsize=(10, 5))
    plt.scatter(X, Y, color='blue', s=25, label='data')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.grid()
    plt.show()

    _, axis = plt.subplots()
    color = ['black', 'green', 'blue', 'purple','brown','pink']
    axis.grid()
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.scatter(X[:,np.newaxis],Y[:np.newaxis], color='red', s=25,label='data')
    axis.set_title('LinearRegression')

    for i in range(1,5):
        x_poly,y = generate_new_dimensions(X,Y,i)
        model = polynomial_linear_regression(x_poly,y,i)
        plot_results(x_poly,y,model,axis,i)
    plt.show()