import numpy as np


def data_process(data):
    data = data.split('\n')
    
    data = [line.split(',') for line in data][:-2]
       
    data = np.array([np.array(line) for line in data])
    x_ = data[:, :-1]
    x = np.zeros(x_.shape)
    y = data[:, -1:]
    (u, v) = x.shape
    for i in range(u):
        for j in range(v):
            x[i, j] = float(x_[i, j])
    return x, y


def set_main(y, s):
    (u, v) = y.shape
    y_ = np.zeros((u, v))
    for i in range(u):
        if y[i, 0] == s:
            y_[i, 0] = 1
        else:
            y_[i, 0] = 0
    return y_


def feture_normalize(x):

    for i in range(n):
        mu = np.mean(x[:, i])
        sigma = np.max(x[:, i]) - np.min(x[:, i])
        x[:, i] = (x[:, i] - mu) / sigma
    return x


def sigmoid(theta, X):
    return 1/(1 + np.exp(-np.dot(X, theta)))


def compute_cost(X, y, theta):
    tmp = sigmoid(theta, X)
    return -np.mean(y * np.log(tmp) + (1 - y) * np.log(1 - tmp))


def gradient_descent(X, y, theta, alpha):
    cost = compute_cost(X, y, theta)
    while(1):
        dif = sigmoid(theta, X) - y
        for j in range(n + 1):
            theta[j, 0] -= alpha * np.dot(dif.T, X[:, j]) / m
        
        J = compute_cost(X, y, theta)
      
        if cost - J < 0.00001:
            cost = J
            break
        else:
            cost = J

    return cost, theta


def train(X, y):
    alpha = 0.001
    
    theta1 = np.zeros((n + 1, 1))
    y1 = set_main(y, 'Iris-setosa')
    _, theta1 = gradient_descent(X, y1, theta1, alpha)

    theta2 = np.zeros((n + 1, 1))
    y2 = set_main(y, 'Iris-versicolor')
    _, theta2 = gradient_descent(X, y2, theta2, alpha)

    theta3 = np.zeros((n + 1, 1))
    y3 = set_main(y, 'Iris-virginica')
    _, theta3 = gradient_descent(X, y3, theta3, alpha)
    
    return np.concatenate((theta1, theta2, theta3), axis=1)


def predict(x, Theta):
    
    x = np.concatenate((np.ones((1,1)), x), axis=1)
    ans = np.argmax(np.dot(x, Theta))
    if ans == 0:
        return 'Iris-setosa'
    elif ans == 1:
        return 'Iris-versicolor'
    else:
        return 'Iris-virginica'


if __name__ == "__main__":
    f = open('iris.data', 'r')
    data = f.read()
    f.close()
    print(data)

    x, y = data_process(data)
    (m, n) = x.shape
    x = feture_normalize(x)
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    
    Theta = train(X, y)
    
    # test
    # for i in range(m-1):
    #     ans = predict(..., Theta)
    #     print(ans)
