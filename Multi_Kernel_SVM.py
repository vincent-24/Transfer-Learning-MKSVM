from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel

class Multi_Kernel_SVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def compute_kernel_matricies(self):
       # For train data, compute individual kernel matrices for training and test data
        self.K_linear_train = linear_kernel(self.X_train)  
        self.K_rbf_train = rbf_kernel(self.X_train, gamma=0.5) 
        self.K_poly_train = polynomial_kernel(self.X_train, degree=3)  
        self.K_sigmoid_train = sigmoid_kernel(self.X_train, gamma=0.1, coef0=0)

        # For test data, compute the kernels between X_test and X_train
        self.K_linear_test = linear_kernel(self.X_test, self.X_train)  
        self.K_rbf_test = rbf_kernel(self.X_test, self.X_train, gamma=0.5)  
        self.K_poly_test = polynomial_kernel(self.X_test, self.X_train, degree=3)  
        self.K_sigmoid_test = sigmoid_kernel(self.X_test, self.X_train, gamma=0.1, coef0=0)

    def combine_kernels(self, weights):
        self.K_combined_train = (
            weights[0] * self.K_linear_train + 
            weights[1] * self.K_rbf_train + 
            weights[2] * self.K_poly_train + 
            weights[3] * self.K_sigmoid_train
        )
        self.K_combined_test = (
            weights[0] * self.K_linear_test + 
            weights[1] * self.K_rbf_test + 
            weights[2] * self.K_poly_test + 
            weights[3] * self.K_sigmoid_test
        )

    def fit_combined_kernels(self):
        self.svc = SVC(kernel='precomputed')
        self.svc.fit(self.K_combined_train, self.y_train)

    def predict_combined_kernels(self):
        self.y_pred = self.svc.predict(self.K_combined_test)

    def get_accuracy(self):
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'Multi-kernel SVM Accuracy: {self.accuracy * 100:.2f}%')

    