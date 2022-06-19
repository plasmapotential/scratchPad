# polynomialRegression.py
# Date:         20181006
# Description:  Basic Polynomial Regression 
# Engineer:     Tom Looby

import matplotlib.pyplot as plt
import numpy as np
import pandas

#Polynomial Degree for Regression (1=linear, 2=quadratic, etc)
degree = 2
#Number of Training Sets
ntrain = 299
#HP row for missing values
hprow = 3
#Flags
feature_flag = 0
norm_flag = 0

#Root Directory where we are working
root_dir = '/home/workhorse/school/grad/machine_learning/projects/project1/'
data_file = root_dir + 'auto-mpg.data'
#Import Raw Data, Divide into train (75%) and test (25%)
raw_data = pandas.read_csv(data_file, header=None, delim_whitespace=True, na_values=["?"], usecols=[0,1,2,3,4,5,6,7])

#Remove specific feature when feature_flag == 1
if feature_flag ==1:
   print("Using these Columns: ")
   print(list(raw_data))
   raw_data.rename(columns={0:0,2:1,3:2,4:3,5:4,6:5,7:6}, inplace=True)

#Replace missing values with column mean in HP column
raw_data = raw_data.fillna(raw_data[hprow].mean())

#Normalize if norm_flag is set to 1
if norm_flag==1:
   #Get mins and maxes column-wise
   maxval = []
   minval = []
   for i in range(raw_data.shape[1]):
      maxval.append(raw_data[i].max(axis=0))
      minval.append(raw_data[i].min(axis=0))
   #Normalize between 0 and 1
   for j in range(1,raw_data.shape[1]):
      for i in range(raw_data.shape[0]):
         if minval[j] == maxval[j]:
           raw_data[i,j] = 1
         else:
            raw_data.ix[i,j] = (raw_data.ix[i,j] - minval[j])/(maxval[j] - minval[j])

# Number of Features
D = raw_data.shape[1]
#Number of Datapoints
N = len(raw_data)
#Features / inputs
X_mat = np.zeros((N, (D-1)*(degree)+1))
#Targets
targets = raw_data[0]

#Build X array based upon polynomial degree
colnum = 0
for i in range(D-1):
   for j in range(degree):
      X_mat[:,colnum] = raw_data[i+1]**(j+1)
      colnum += 1
X_mat[:,(D-1)*(degree)] = 1

#Training Data
X_tr = X_mat[0:ntrain]
target_tr = targets[0:ntrain]
#Test Data
X_te = X_mat[ntrain+1::]
N_test = float(N - ntrain)

#W = (X^T X)^-1 (X^T) (target)
W = np.matmul( np.matmul( np.linalg.inv( np.matmul(X_tr.T,X_tr) ), X_tr.T ) ,target_tr)

#Test the model with Trainset
train_pred = np.matmul(X_tr[0:ntrain], W)
train_error = (1.0/(float(N)))*np.square(np.subtract(train_pred, targets[0:ntrain])).mean()
#Now Test the model with Testset
test_pred = np.matmul(X_te[0:98], W)
test_error = (1.0/(N_test))*np.square(np.subtract(test_pred, targets[ntrain+1::])).mean()

# Results and Plots
print("Polynomial Degree: {:d}".format(degree))
print("Train Error: {:f}".format(train_error))
print("Test Error: {:f}".format(test_error))

# Plot Predictions vs Targets with Matplotlib
plt.figure(1)
plt.title("Validation Set Predictions vs Targets.  Polynomial Degree = {:d}".format(degree))
plt.plot(targets[ntrain+1::].index - ntrain, targets[ntrain+1::], '-.', linewidth=1, label='Target Datapoints')
plt.plot(test_pred, linewidth=2, label='Prediction')
plt.ylabel('MPG')
plt.xlabel('Datapoint')
plt.legend(loc='upper right')
plt.show()
