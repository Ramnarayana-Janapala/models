import numpy as np
from matplotlib import pyplot as plt
'''
 Data in the X is called features and data in the Y_actual is called labels
 Formula using in the Model Y = θ₃X³+-θ₂X² -θ₁*X + θ₀, with MSE as calculating the accuracy
'''

X = np.arange(-1,1,0.01)
np.random.shuffle(X)
# this is the noise were are creating to the data
Y_actual = 3*X**3-2*X**2 -1*X + 2 + 0.2 * np.random.normal(0, 1, len(X))

def split_dataset(x, y, test_data_percentage=30 ):
    idx = int(len(x)*test_data_percentage/100)
    x_train = x[idx:]
    y_train = y[idx:]
    x_test = x[:idx]
    y_test = y[:idx]

    return x_train,y_train,x_test,y_test

x_train_data,y_train_data,x_test_data,y_test_data = split_dataset(X,Y_actual)


# Step 1: Get the Random values to start the predication process.
# Different ramdom values provide different predications
def get_random_numbers(total_nums):
    """
    Get the random numbers between -1 to 1 total of the given size
    :param total_nums:
    :return:
    """
    return np.random.uniform(-1,1,size=total_nums)

#Calculate the Y predication value
def predict_y(x,params):
    y_predict = params[0] + params[1] * x + params[2] * x**2 + params[3] * x**3
    return y_predict


# Definition to train the model
def model_training(x,y,lr=0.1,epochs=1000):
    """
    :param x: feature data set
    :param y: label data set
    :param lr: learning rate
    :param epochs: number of iterations
    :return:
    """
    # Step 1 , Random Parameters
    params = get_random_numbers(4)
    print(f"Random Values : {params}")

    cost_per_epoch_l = []
    # Here there is no convergence point, but in real world we have threshold where the
    # loops get stop of the value have no deviation beyond the threshold after certain loops then we break it.
    for _ in range(epochs):
        cost = 0
        # For a 2 degree polynomial you have three theta value (θ₂X^2+θ₁*X + θ₀)
        dj_dtheta_0, dj_dtheta_1, dj_dtheta_2, dj_dtheta_3 = 0, 0, 0, 0
        for i in range(len(x)):
            # Step 2 Calculate Y predicate
            y_predict = predict_y(x[i],params=params)
            # Step 3 find the error value for a given record/observation/sample
            error = y_predict - y[i]
            cost = cost + error**2
            dj_dtheta_0 = dj_dtheta_0 + error
            dj_dtheta_1 = dj_dtheta_1 + error * x[i]
            dj_dtheta_2 = dj_dtheta_2 + error * x[i]**2
            dj_dtheta_3 = dj_dtheta_3 + error * x[i]**3
        cost = cost/len(x)
        cost_per_epoch_l.append(cost)
        params[0] = params[0] - (lr * 2 * dj_dtheta_0) / len(x)
        params[1] = params[1] - (lr * 2 * dj_dtheta_1) / len(x)
        params[2] = params[2] - (lr * 2 * dj_dtheta_2) / len(x)
        params[3] = params[3] - (lr * 2 * dj_dtheta_3) / len(x)
    return params, cost_per_epoch_l

cal_params, cost_per_epoch = model_training(x_train_data,y_train_data)
print(f"predicted values : {cal_params}")

# Verify the params with test data
y_predict_test = []
for i in range(len(x_test_data)):
    y_predict_test.append(predict_y(x_test_data[i],cal_params))


plt.figure()
plt.plot(x_test_data,y_predict_test,"*",color='red')
plt.plot(x_test_data,y_test_data,"*",color='green')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()