import pandas as pd
import numpy as np
from numpy import load
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from Conventional_SVM import *
from Multi_Kernel_SVM import *

#====================================DEFINE DATASET AND VARIABLES===================================#
'''
train = 'Malware Dataset/Adjusted Dataset/Adjsample_train'
test = 'Malware Dataset/Adjusted Dataset/Adjsample_test' 

feature_df = pd.read_csv(train)[pd.read_csv(train).columns.drop('label').tolist()]
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']

X_train = np.asarray(feature_df) 
y_train = np.asarray(pd.read_csv(train)['label'])
X_test = np.asarray(pd.read_csv(test)[feature_df.columns]) 
y_test = np.asarray(pd.read_csv(test)['label']) 
'''

data = pd.read_csv('Malware Dataset/archive/ClaMP_Integrated-5184.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_numeric = X.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numeric)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_numeric.columns)

X_non_numeric = X.select_dtypes(exclude=['float64', 'int64'])
X_scaled_full = pd.concat([X_scaled_df, X_non_numeric.reset_index(drop=True)], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#================================CONVENTIONAL SUPPORT VECTOR MACHINE================================#
print('Conventional Support Vector Machine:\n')
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
conventional_SVM = Conventional_SVM(X_train, y_train, X_test, y_test)
conventional_SVM.test_kernels(kernel_types)

# Define kernels to get optimal parameters
'''
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
# '''
print('\n\n\nMultiple Kernel Support Vector Machine:\n')
mksvm = Multi_Kernel_SVM(X_train, y_train, X_test, y_test)
mksvm.compute_kernel_matricies()
mksvm.combine_kernels([0.2, 0.4, 0.5, 0.3])
mksvm.fit_combined_kernels()
mksvm.predict_combined_kernels()
mksvm.get_accuracy()
# '''

#====================TRANSFER LEARNING FEATURE EXTRACTION FUNCTION====================#
from Transfer_Learning import *

print('\n\n\nTransfer Learning Multiple Kernel Support Vector Machine:\n')

num_datasets = 5

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.5, random_state=42)
X_test_splits = np.array_split(X_test, num_datasets)
y_test_splits = np.array_split(y_test, num_datasets)
test_sets = [(X_test_splits[i], y_test_splits[i]) for i in range(num_datasets)]

clf = TransferLearningSVM(X_train, y_train, test_sets)
clf.tlmksvm()
    