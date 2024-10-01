import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from MKLpy.algorithms import EasyMKL
from Conventional_SVM import *

# Load datasets
adjsample_train = 'Malware Dataset/Adjusted Dataset/Adjsample_train'
adjsample_test = 'Malware Dataset/Adjusted Dataset/Adjsample_test'   

# SVM accuracy with default parameters
print('\nConventional Support Vector Machines\n')
conventional_SVM = Conventional_SVM(adjsample_train, adjsample_test)
conventional_SVM.define_features('label')
conventional_SVM.test_kernels(kernels = ['linear', 'poly', 'rbf', 'sigmoid'])

# Define kernels to get optimal parameters
kernel_types = ['linear', 'poly', 'rbf', 'sigmoid']
common_param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('svc', SVC())
])

# Loop through each kernel type and output optimal parameters
for kernel in kernel_types:
    param_grid = common_param_grid.copy()
    param_grid['svc__kernel'] = [kernel]
    
    # Degree only applies to 'poly' kernel
    if kernel == 'poly':
        param_grid['svc__degree'] = [2, 3, 4, 5]
    
    # Call the function with the current kernel's param_grid
    conventional_SVM.optimal_parameters(pipeline, param_grid, 5, 'accuracy')

# Multi-Kernel Learning 
print('\n\n\nMultiple Kernel Support Vector Machine\n')

feature_df = pd.read_csv(adjsample_train)[[
        'Payload_ratio', 'flow duration', 'max_Length_of_IP_packets', 'std_Length_of_IP_packets', 
        'var_Length_of_IP_packets', 'max_Length_of_TCP_payload', 'std_Length_of_TCP_payload', 
        'var_Length_of_TCP_payload', 'std_Length_of_TCP_packet_header', 'var_Length_of_TCP_packet_header', 
        'max_Length_of_TCP_segment(packet)', 'std_Length_of_TCP_segment(packet)', 'var_Length_of_TCP_segment(packet)', 
        'max_Time_difference_between_packets_per_session', 'std_Time_difference_between_packets_per_session', 
        'max_Interval_of_arrival_time_of_forward_traffic', 'max_Interval_of_arrival_time_of_backward_traffic', 
        'std_Interval_of_arrival_time_of_backward_traffic', 'std_backward_pkt_length', 'IPratio', 'Domain'
    ]]

# Example data (X_train, y_train, X_test, y_test)
X_train = np.asarray(feature_df) 
y_train = np.asarray(pd.read_csv(adjsample_train)['label'])
X_test = np.asarray(pd.read_csv(adjsample_test)[feature_df.columns]) 
y_test = np.asarray(pd.read_csv(adjsample_test)['label']) 

# Step 1: Compute individual kernel matrices for training and test data
K_linear_train = linear_kernel(X_train)  
K_rbf_train = rbf_kernel(X_train, gamma=0.5) 
K_poly_train = polynomial_kernel(X_train, degree=3)  
K_sigmoid_train = sigmoid_kernel(X_train, gamma=0.1, coef0=0)

# For test data, compute the kernels between X_test and X_train
K_linear_test = linear_kernel(X_test, X_train)  
K_rbf_test = rbf_kernel(X_test, X_train, gamma=0.5)  
K_poly_test = polynomial_kernel(X_test, X_train, degree=3)  
K_sigmoid_test = sigmoid_kernel(X_test, X_train, gamma=0.1, coef0=0)

# Step 2: Combine kernels manually (weighted sum) for both training and test data
beta_linear, beta_rbf, beta_poly, beta_sigmoid = 0.2, 0.4, 0.5, 0.3  # weights

K_combined_train = (
        beta_linear * K_linear_train + 
        beta_rbf * K_rbf_train + 
        beta_poly * K_poly_train + 
        beta_sigmoid * K_sigmoid_train
    )
K_combined_test = (
        beta_linear * K_linear_test + 
        beta_rbf * K_rbf_test + 
        beta_poly * K_poly_test + 
        beta_sigmoid * K_sigmoid_test
    )

# Step 3: Fit the SVM model using the combined kernel on training data
svc = SVC(kernel='precomputed')
svc.fit(K_combined_train, y_train)

# Step 4: Predict using the combined kernel on test data
y_pred = svc.predict(K_combined_test)

# Step 5: Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Multi-kernel SVM Accuracy: {accuracy * 100:.2f}%')