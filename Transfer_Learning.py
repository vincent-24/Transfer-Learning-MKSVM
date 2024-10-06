import numpy as np
from sklearn.metrics import accuracy_score
from Multi_Kernel_SVM import Multi_Kernel_SVM

class TransferLearningSVM:
    def __init__(self, source_X, source_y, target_X, target_y):
        self.source_X = source_X
        self.source_y = source_y
        self.target_X = target_X
        self.target_y = target_y

    def train_on_source(self):
        # Initialize Multi-Kernel SVM
        self.source_mksvm = Multi_Kernel_SVM(self.source_X, self.source_y, self.source_X, self.source_y)
        self.source_mksvm.compute_kernel_matricies()
        # Combine kernels with equal weights initially
        self.source_mksvm.combine_kernels([0.25, 0.25, 0.25, 0.25])
        self.source_mksvm.fit_combined_kernels()

    def fine_tune_on_target(self):
        # Recompute kernel matrices between the target dataset only
        self.target_mksvm = Multi_Kernel_SVM(self.target_X, self.target_y, self.target_X, self.target_y)
        
        # Compute the kernel matrices for the target dataset (train and test both use target data)
        self.target_mksvm.compute_kernel_matricies()
        
        # Combine kernels with equal weights for fine-tuning
        self.target_mksvm.combine_kernels([0.25, 0.25, 0.25, 0.25])
        
        # Fine-tune using the target training data kernel matrix
        self.target_mksvm.fit_combined_kernels()
        
        # Predict using the target test kernel matrix
        self.target_mksvm.predict_combined_kernels()
        
        # Now calculate accuracy using the correct target labels and predictions
        accuracy = accuracy_score(self.target_y, self.target_mksvm.y_pred)
        print(f"Fine-tuning Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    def save_kernels(self, path):
        # Save the kernel matrices if needed
        np.savez(path, K_combined_train=self.source_mksvm.K_combined_train, K_combined_test=self.source_mksvm.K_combined_test)

    def load_kernels(self, path):
        # Load pre-trained kernels if needed
        data = np.load(path)
        self.source_mksvm.K_combined_train = data['K_combined_train']
        self.source_mksvm.K_combined_test = data['K_combined_test']
