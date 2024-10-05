import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from Conventional_SVM import *
from Multi_Kernel_SVM import *

#====================================DEFINE DATASET AND VARIABLES===================================#
train = 'Malware Dataset/Adjusted Dataset/Adjsample_train'
test = 'Malware Dataset/Adjusted Dataset/Adjsample_test'  

feature_df = pd.read_csv(train)[pd.read_csv(train).columns.drop('label').tolist()]
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']

X_train = np.asarray(feature_df) 
y_train = np.asarray(pd.read_csv(train)['label'])
X_test = np.asarray(pd.read_csv(test)[feature_df.columns]) 
y_test = np.asarray(pd.read_csv(test)['label']) 

#================================CONVENTIONAL SUPPORT VECTOR MACHINE================================#
'''
print('Conventional Support Vector Machine:\n')
conventional_SVM = Conventional_SVM(X_train, y_train, X_test, y_test, feature_df)
conventional_SVM.test_kernels(kernel_types)

# Define kernels to get optimal parameters
common_param_grid = {'svc__C': [0.1, 1, 10, 100], 'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1], 'svc__degree': [2, 3, 4, 5]}

pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])

# Loop through each kernel type and output optimal parameters
print('\n\n\nBest Parameters for each Kernel:')
for kernel in kernel_types:
    param_grid = common_param_grid.copy()
    param_grid['svc__kernel'] = [kernel]
    conventional_SVM.optimal_parameters(pipeline, param_grid, 5, 'accuracy')
'''

#================================MULTI-KERNEL SUPPORT VECTOR MACHINE================================#
print('\n\n\nMultiple Kernel Support Vector Machine:\n')
mksvm = Multi_Kernel_SVM(X_train, y_train, X_test, y_test)
mksvm.compute_kernel_matricies()
mksvm.combine_kernels([0.2, 0.4, 0.5, 0.3])
mksvm.fit_combined_kernels()
mksvm.predict_combined_kernels()
mksvm.get_accuracy()