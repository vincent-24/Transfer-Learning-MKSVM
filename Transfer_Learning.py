import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Multi_Kernel_SVM import *

'''
class TransferLearningSVM:
    def __init__(self, X_train, y_train, test_sets):
        self.X_train = X_train
        self.y_train = y_train
        self.test_sets = test_sets
    
    def __normalize_kernel(self, K):
        norm_factor = np.linalg.norm(K, ord='fro')
        if norm_factor == 0:
            return K
        return K / norm_factor

    def __compute_normalized_kernels(self, gamma_rbf=0.5, degree_poly=3, gamma_sigmoid=0.1, coef0_sigmoid=0):
        K_linear_train = self.__normalize_kernel(linear_kernel(self.X_train, self.X_train))
        K_rbf_train = self.__normalize_kernel(rbf_kernel(self.X_train, self.X_train, gamma=gamma_rbf))
        K_poly_train = self.__normalize_kernel(polynomial_kernel(self.X_train, self.X_train, degree=degree_poly))
        K_sigmoid_train = self.__normalize_kernel(sigmoid_kernel(self.X_train, self.X_train, gamma=gamma_sigmoid, coef0=coef0_sigmoid))

        K_linear_test = self.__normalize_kernel(linear_kernel(self.X_test, self.X_train))
        K_rbf_test = self.__normalize_kernel(rbf_kernel(self.X_test, self.X_train, gamma=gamma_rbf))
        K_poly_test = self.__normalize_kernel(polynomial_kernel(self.X_test, self.X_train, degree=degree_poly))
        K_sigmoid_test = self.__normalize_kernel(sigmoid_kernel(self.X_test, self.X_train, gamma=gamma_sigmoid, coef0=coef0_sigmoid))

        return (K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train,
                K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test)
    
    def __combine_kernels(self, weights, K_linear, K_rbf, K_poly, K_sigmoid):
        return (weights[0] * K_linear) + (weights[1] * K_rbf) + (weights[2] * K_poly) + (weights[3] * K_sigmoid)
    
    def __train_svm_with_combined_kernels(self, K_train):
        svm = SVC(kernel='precomputed')
        svm.fit(K_train, self.y_train)
        return svm
    
    def __gradient_based_weight_optimization(self, kernel_weights, learning_rate, max_iters):
        best_weights = kernel_weights
        best_accuracy = 0.0

        for i in range(max_iters):
            (K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train,
            K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test) = self.__compute_normalized_kernels()

            K_train_combined = self.__combine_kernels(kernel_weights, K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train)
            K_test_combined = self.__combine_kernels(kernel_weights, K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test)

            svm = self.__train_svm_with_combined_kernels(K_train_combined)

            y_pred = svm.predict(K_test_combined)
            current_accuracy = accuracy_score(self.y_test, y_pred)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weights = kernel_weights.copy()

            gradients = np.random.randn(4)

            kernel_weights += learning_rate * gradients
            kernel_weights = np.clip(kernel_weights, 0, 1)
            kernel_weights /= np.sum(kernel_weights)

            if i % 10 == 0:
                print(f"Iteration {i}: Accuracy = {current_accuracy:.3f}, Weights = {kernel_weights}")

        return best_weights, best_accuracy
    
    def __transfer_learning_iteration_with_gradient_optimization(self, prev_weights, alpha, epoch):
        best_weights, best_accuracy = self.__gradient_based_weight_optimization(prev_weights, alpha, epoch)

        print(f"Best Accuracy: {best_accuracy:.3f}")
        print(f"Optimized Kernel Weights: {best_weights}")

        return best_accuracy, best_weights
    
    def tlmksvm(self):
        kernel_weights = np.array([0.25, 0.25, 0.25, 0.25])
        alpha = 0.01
        epoch = 100

        for i, (self.X_test, self.y_test) in enumerate(self.test_sets, start=1):
            print(f"\nTransfer Learning Iteration {i}")
            accuracy, kernel_weights = self.__transfer_learning_iteration_with_gradient_optimization(kernel_weights, alpha, epoch)
'''

class TransferLearningSVM:
    def __init__(self, X_train, y_train, test_sets):
        self.X_train = X_train
        self.y_train = y_train
        self.test_sets = test_sets
        self.scaler = StandardScaler()

    def __normalize_kernel(self, K):
        norm_factor = np.linalg.norm(K, ord='fro')
        return K if norm_factor == 0 else K / norm_factor

    def __compute_normalized_kernels(self, X_test, gamma_rbf=0.5, degree_poly=3, gamma_sigmoid=0.1, coef0_sigmoid=0):
        K_linear_train = self.__normalize_kernel(linear_kernel(self.X_train, self.X_train))
        K_rbf_train = self.__normalize_kernel(rbf_kernel(self.X_train, self.X_train, gamma=gamma_rbf))
        K_poly_train = self.__normalize_kernel(polynomial_kernel(self.X_train, self.X_train, degree=degree_poly))
        K_sigmoid_train = self.__normalize_kernel(sigmoid_kernel(self.X_train, self.X_train, gamma=gamma_sigmoid, coef0=coef0_sigmoid))

        K_linear_test = self.__normalize_kernel(linear_kernel(X_test, self.X_train))
        K_rbf_test = self.__normalize_kernel(rbf_kernel(X_test, self.X_train, gamma=gamma_rbf))
        K_poly_test = self.__normalize_kernel(polynomial_kernel(X_test, self.X_train, degree=degree_poly))
        K_sigmoid_test = self.__normalize_kernel(sigmoid_kernel(X_test, self.X_train, gamma=gamma_sigmoid, coef0=coef0_sigmoid))

        return (K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train,
                K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test)

    def __combine_kernels(self, weights, K_linear, K_rbf, K_poly, K_sigmoid):
        return (weights[0] * K_linear) + (weights[1] * K_rbf) + (weights[2] * K_poly) + (weights[3] * K_sigmoid)

    def __train_svm_with_combined_kernels(self, K_train):
        svm = SVC(kernel='precomputed')
        svm.fit(K_train, self.y_train)
        return svm

    def __gradient_based_weight_optimization(self, kernel_weights, learning_rate, max_iters, X_test, y_test):
        best_weights = kernel_weights
        best_accuracy = 0.0

        for i in range(max_iters):
            (K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train,
             K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test) = self.__compute_normalized_kernels(X_test)

            K_train_combined = self.__combine_kernels(kernel_weights, K_linear_train, K_rbf_train, K_poly_train, K_sigmoid_train)
            K_test_combined = self.__combine_kernels(kernel_weights, K_linear_test, K_rbf_test, K_poly_test, K_sigmoid_test)

            svm = self.__train_svm_with_combined_kernels(K_train_combined)

            y_pred = svm.predict(K_test_combined)
            current_accuracy = accuracy_score(y_test, y_pred)

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_weights = kernel_weights.copy()

            # Update kernel weights with a small random adjustment
            gradients = np.random.normal(0, 0.01, size=kernel_weights.shape)
            kernel_weights += learning_rate * gradients
            kernel_weights = np.clip(kernel_weights, 0, 1)
            kernel_weights /= np.sum(kernel_weights)

            if i % 10 == 0:
                print(f"Iteration {i}: Accuracy = {current_accuracy:.3f}, Weights = {kernel_weights}")

        return best_accuracy, best_weights

    def tlmksvm(self):
        kernel_weights = np.array([0.25, 0.25, 0.25, 0.25])
        learning_rate = 0.005
        epoch = 100

        for i, (X_test, y_test) in enumerate(self.test_sets, start=1):
            print(f"\nTransfer Learning Iteration {i}")
            accuracy, kernel_weights = self.__gradient_based_weight_optimization(kernel_weights, learning_rate, epoch, X_test, y_test)
            print(f"Iteration {i} Final Accuracy: {accuracy}, Optimized Kernel Weights: {kernel_weights}")
