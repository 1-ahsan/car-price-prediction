#
# # Importing required libraries
#
import matplotlib.pyplot as plt
import numpy as np
import copy,math 

np.set_printoptions(precision=2)
#
# # fetching data
#

data = np.loadtxt('./Machine_learning/Car_price_pridiction/data/Training Data.txt'  , delimiter=',' , skiprows=1)
names = np.loadtxt('./Machine_learning/Car_price_pridiction/data/Training Data.txt'  , delimiter=',' , max_rows=1, dtype=str)
'''
print(data)
print(names)
'''
x_train = data[:,0:4]
y_train = data[:,4]
'''
print(x_train)
print(y_train)
'''

#
# # visualizing Data
#

'''   Every parameter vs the price   '''
def visualizing_parameters(x,y):
    fig , ax = plt.subplots(1,4, figsize=(12,3), sharey=True)

    for i in range(4):
        ax[i].scatter(x[:,i],y)
        ax[i].set_xlabel(names[i])

    ax[0].set_ylabel("price")
    fig.suptitle("Showing how each parameter affect on the price")
    plt.tight_layout()
    plt.show()

#
# # Normalizing the dataset
#
'''   As data for different parameters is in different range that can cause problem while training the model for that 
      we will normalize the data.
'''

def zscore_normalization(x):
    mean = np.mean(x,axis=0)
    sigma = np.std(x,axis=0)
    x_normalize = (x - mean) / sigma

    '''print(mean,sigma,x_normalize)'''
    return x_normalize, mean, sigma


'''   Visualizing data after normalization   '''

# x_norm, _ , _ = zscore_normalization(x_train)
# print(x_norm)

'''   visuallizing data before and after normalization '''

# x_norm, _ , _ = zscore_normalization(x_train)
# visualizing_parameters(x_train,y_train)
# visualizing_parameters(x_norm,y_train)



#
# # Cost Function
#

def cost_function(x,y,w,b):
    m = x.shape[0]
    sum =0.0
    for i in range(m):
        f_wb = np.dot(x[i],w) + b
        sum += (f_wb - y[i]) ** 2
    cost = sum / (2*m)
    return cost

'''
w=np.array([10,10,10,10])
cost = cost_function(x_train,y_train,w,10)
print(cost)
cost = cost_function(x_norm,y_train,w,10)
print(cost)
'''

#
# # Gradient Decent
#

'''   finging gradient   '''
def gradient(x,y,w,b):
    m,n = x.shape

    dj_dw = np.zeros((n,))
    dj_db = 0
    for i in range(m):
        err = (np.dot(x[i],w) + b ) - y[i] 
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err *x[i,j]
        dj_db += err
        
    dj_dw = dj_dw /m
    dj_db = dj_db /m

    return dj_dw , dj_db

'''   finding gradient decent   '''
def gradient_decent(x,y,w_in,b_in,alpha,num_itrators):
    J_history = []
    cost_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_itrators):
        dj_dw , dj_db = gradient(x,y,w,b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        J_history.append(cost_function(x,y,w,b))
        if len(J_history) > 1:
            if abs(J_history[-2] - J_history[-1]) < 0.003:
                break
        
        if i% math.ceil(num_itrators / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
        
    return w,b,J_history

'''   Performing Some testings   '''
'''
w=np.array([0,0,0,0])
b=0
cost = cost_function(x_train,y_train,w,b)
print(cost)

alpha = 0.5
iter = 10000
x_norm, _ , _ = zscore_normalization(x_train)

nw,nb,j= gradient_decent(x_norm,y_train,w,b,alpha,iter)
cost = cost_function(x_norm,y_train,nw,nb)
print(cost)
print(f"w,b found : {nw,nb}")

tail = j[100:]
length = len(j[100:])


plt.plot(100 + np.arange(length),tail)
plt.xlabel("iterations")
plt.ylabel("Cost")
plt.title("Cost function convergence")
plt.show()
'''

#
# # Making pridiction
#

def compute_pridiction(x,w,b):
    m = len(x)

    for i in range(m):
        f_wb = np.dot(x,w) + b
    
    return f_wb

#
# # Some initial settings
#

w=np.array([0,0,0,0])
b=0
cost = cost_function(x_train,y_train,w,b)
print(cost)

alpha = 0.5
iter = 10000
x_norm, mean , sigma = zscore_normalization(x_train)

nw,nb,j = gradient_decent(x_norm,y_train,w,b,alpha,iter)
print(f"w,b found : {nw,nb}")

'''   Pridiction vs the actual values   '''
y_hat = np.dot(x_norm,nw) + nb

# plt.scatter(y_train,y_hat)
plt.scatter(range(len(y_train)), y_train, label='Actual Price')
plt.scatter(range(len(y_hat)), y_hat, marker='x', color="red", label="Predicted Price")
plt.title("Preciction vs the actual values")
plt.legend()
plt.show()



inp = np.array([30000,3,130,88.6])
inp_norm = ( inp - mean ) / sigma



predition = compute_pridiction(inp_norm,nw,nb)
print(f"pridiction = {predition}")