def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
#     for i in range(m):
#         z_i = np.dot(X[i],w) + b
#         f_wb_i = sigmoid(z_i)
#         cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
#     cost = cost / m #0.3668667864055175
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        expo = np.exp(-z_i)
        f_wb_i = 1/(1+expo)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
        
    cost = cost/m  # 0.36686678640551745

    return cost
"""
There is a slight difference in the precision, what Andrew Ng sir provided: 0.3668667864055175
                                              what my code returns: 0.36686678640551745
"""
