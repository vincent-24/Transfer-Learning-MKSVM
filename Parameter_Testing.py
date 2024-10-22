import pandas as pd
import numpy as np
from numpy import load
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from Conventional_SVM import *
from Multi_Kernel_SVM import *

#====================================DEFINE DATASET AND VARIABLES===================================#
# Initialize variables
dataset = ['NMAP_FIN_SCAN', 'NMAP_OS_DETECTION', 'NMAP_TCP_scan', 'NMAP_UDP_SCAN', 'NMAP_XMAS_TREE_SCAN']

X_NMAP_FIN_SCAN = None
y_NMAP_FIN_SCAN = None

X_NMAP_OS_DETECTION = None
y_NMAP_OS_DETECTION = None

X_NMAP_TCP_scan = None
y_NMAP_TCP_scan = None

X_NMAP_UDP_SCAN = None
y_NMAP_UDP_SCAN = None

X_NMAP_XMAS_TREE_SCAN = None
y_NMAP_XMAS_TREE_SCAN = None

# Standardize dataset features
for n in dataset:
    data = pd.read_csv(f'Malware Dataset/UCI_NMAP_datasets/{n}.csv')

    X = data.iloc[:, :-1]  
    y = data.iloc[:, -1]   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    globals()[f'X_{n}'] = X_scaled_df
    globals()[f'y_{n}'] = y.values

# Conventional SVM
print('Conventional Support Vector Machine:\n')
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
conventional_SVM = Conventional_SVM(X_NMAP_TCP_scan, y_NMAP_TCP_scan, X_NMAP_OS_DETECTION, y_NMAP_OS_DETECTION)
conventional_SVM.test_kernels(kernel_types)

# Multi-Kernel SVM
print('\n\n\nMultiple Kernel Support Vector Machine:\n')
mksvm = Multi_Kernel_SVM(X_NMAP_TCP_scan, y_NMAP_TCP_scan, X_NMAP_OS_DETECTION, y_NMAP_OS_DETECTION)
mksvm.compute_kernel_matricies()
mksvm.combine_kernels([0.2, 0.4, 0.5, 0.3])
mksvm.fit_combined_kernels()
mksvm.predict_combined_kernels()
mksvm.get_accuracy()

# Transfer Learning MKSVM
from Transfer_Learning import *

print('\n\n\nTransfer Learning Multiple Kernel Support Vector Machine:\n')

test_sets = [
    (X_NMAP_OS_DETECTION, y_NMAP_OS_DETECTION),
    (X_NMAP_UDP_SCAN, y_NMAP_UDP_SCAN),
    (X_NMAP_XMAS_TREE_SCAN, y_NMAP_XMAS_TREE_SCAN)
]

clf = TransferLearningSVM(X_NMAP_TCP_scan, y_NMAP_TCP_scan, test_sets)
clf.tlmksvm()