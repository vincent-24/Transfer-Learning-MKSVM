import pandas as pd
import numpy as np
from numpy import load
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from Conventional_SVM import *
from Multi_Kernel_SVM import *

# Initialize variables
dataset = ['RT_IOT', 'android']

X_RT_IOT = None
y_RT_IOT = None

X_android = None
y_android = None

# Standardize dataset features
for n in dataset:
    data = pd.read_csv(f'Malware Dataset/UCI_datasets/{n}.csv')

    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)  # Scale all features in X

    # Create a DataFrame for the scaled features
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    globals()[f'X_{n}'] = X_scaled_df
    globals()[f'y_{n}'] = y.values

# Conventional SVM
print('Conventional Support Vector Machine:\n')
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
conventional_SVM = Conventional_SVM(X_RT_IOT, y_RT_IOT, X_android, y_android)
conventional_SVM.test_kernels(kernel_types)

# Multi-Kernel SVM
print('\n\n\nMultiple Kernel Support Vector Machine:\n')
mksvm = Multi_Kernel_SVM(X_RT_IOT2022, y_RT_IOT2022, X_android, y_android)
mksvm.compute_kernel_matricies()
mksvm.combine_kernels([0.2, 0.4, 0.5, 0.3])
mksvm.fit_combined_kernels()
mksvm.predict_combined_kernels()
mksvm.get_accuracy()

# Transfer Learning MKSVM
from Transfer_Learning import *

print('\n\n\nTransfer Learning Multiple Kernel Support Vector Machine:\n')

test_sets = [
    (X_android, y_android)
]

clf = TransferLearningSVM(X_RT_IOT2022, y_RT_IOT2022, test_sets)
clf.tlmksvm()