import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0,])
y_train = np.array([300.0, 500.0])
# print(f"x_train = {x_train}")
# print(f"y_train = {y_train}")

# m is the number of training examples
# print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# m is the number of training examples
# m = len(x_train)
# print(f"Number of training examples is: {m}")

# i = 1  # Change this to 1 to see (x^1, y^1)
for i in range(m):
    x_i = x_train[i]
    y_i = y_train[i]
    print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
# plt.show()

# ----------------------------------------------------------------#
# 𝑓𝑤,𝑏(𝑥(𝑖))=𝑤𝑥(𝑖)+𝑏

print("Geometric intuition:")
w = 100  # given 100 initially
b = 100
print(f"w: {w}")
print(f"b: {b}")


# this function just draws the straightline
def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]  # total training models
    f_wb = np.zeros(m)  # creating array with m elements (all values = 0)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb


temp_f_wb = compute_model_output(x_train, w, b)
print("Values of temp_f_wb: ", temp_f_wb)
# Plot our model prediction
plt.plot(x_train, temp_f_wb, c='b', label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
# plt.show()

print("Mathematical intuition:")
w = 200
b = 100
x_i = 1.2
cost_1200sqft = w * x_i + b

print(f"${cost_1200sqft:.0f} thousand dollars")


"""
In linear regression, the line that connects two existing points is indeed a valid representation of a linear relationship between those points. However, in the context of machine learning, we often deal with multiple data points and we aim to find a line (or hyperplane in higher dimensions) that best fits all these points. This is done by minimizing the sum of squared residuals (the differences between the observed and predicted values).

f_wb is a point that is y-cap, or predicted output.

The value of b is an arbitrary choice in this example. In a real-world scenario, b would be determined through the training process of the machine learning model, where the model learns the best values of w and b that minimize the error of the model 
"""
