import numpy as np

def data_process(data):
    data = data.split('\n')
    data = [line.split() for line in data][:-1]
    for i in range(len(data)):
        data[i] = np.array([float(e) for e in data[i]])
    data = np.array(data)
    X = data[:, :-1]
    y = data[:, -1:]
    return X, y


def feture_normalize(x):
    for i in range(n):
        mu = np.mean(x[:, i])
        sigma = np.max(x[:, i]) - np.min(x[:, i])
        x[:, i] = (x[:, i] - mu) / sigma
    return x


def compute_cost(X, y, theta):
    return 0.5 * np.mean((np.dot(X, theta) - y) ** 2)


def gradient_descent(X, y, theta, alpha):
    cost = compute_cost(X, y, theta)
    while(1):
        dif = np.dot(X, theta) - y
        for j in range(n + 1):
            theta[j, 0] -= alpha * np.dot(dif.T, X[:, j]) / m
        
        J = compute_cost(X, y, theta)
    
        if cost - J < 0.00000001:
            cost = J
            break
        else:
            cost = J
    return cost, theta

def predict(x, theta):
    return (np.dot(x, theta[1:, 0]) + theta[0, 0])[0]

if __name__ == "__main__":
    f = open('housing.data', 'r')
    data = f.read()
    f.close()

    x, y = data_process(data)
    (m, n) = x.shape
    x = feture_normalize(x)
    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    theta = np.ones((n + 1, 1))
    alpha = 0.001

    cost, theta = gradient_descent(X, y, theta, alpha)
    
    # test
    # ans = predict(..., theta)
    # 


    
