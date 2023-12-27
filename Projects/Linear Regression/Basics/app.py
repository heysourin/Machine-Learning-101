import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0]
w = 0
b = 0


def calculate_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb_x = w * x[i] + b
        dj_dw_i = (f_wb_x - y[i]) * x[i]
        dj_db_i = (f_wb_x - y[i])

        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw/m
    dj_db = dj_db/m

    return dj_dw, dj_db


def calculate_gradient_descent(x, y, w_current, b_current, alpha):
    dj_dw, dj_db = calculate_gradient(x, y, w_current, b_current)
    w_updated = w_current - alpha * dj_dw
    b_updated = b_current - alpha * dj_db

    return w_updated, b_updated


#! Performing the linear gradient:
for _ in range(10000):  # Choose the number of iterations
    w, b = calculate_gradient_descent(x_train, y_train, w, b, 0.01)

print(f"Final value of w = {w:0.2f} and Final value of b = {b:0.2f}")

x_i = 1.5
cost_1500sqft = w * x_i + b
print(f"Cost of 1500 sqft house is: {cost_1500sqft:0.2f} USD")
