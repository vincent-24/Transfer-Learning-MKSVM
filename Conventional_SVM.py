from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class Conventional_SVM:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def __evaluate_svm(self, kernel, C=2, gamma='auto', degree=3):
        if self.X_train.shape[0] == 0 or self.X_test.shape[0] == 0:
            print(f"No data to train or test on with {kernel} kernel.")
            return
        clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)
        clf.fit(self.X_train, self.y_train)
        y_predict = clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_predict)
        print(f'{kernel.capitalize()} Kernel Accuracy: {accuracy * 100:.2f}%')

    def test_kernels(self, kernels):
        for kernel in kernels:
            self.__evaluate_svm(kernel)

    def optimal_parameters(self, pipeline, param_grid, cv, scoring):
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring)
        grid_search.fit(self.X_train, self.y_train)

        best_params = grid_search.best_params_
        best_kernel = best_params['svc__kernel']
        best_C = best_params['svc__C']
        best_gamma = best_params.get('svc__gamma', None)
        best_degree = best_params.get('svc__degree', None)

        # Just for formatting
        param_str = f"C: {best_C}"
        if best_kernel in ['rbf', 'sigmoid', 'poly']:
            param_str += f", Gamma: {best_gamma}"
        if best_kernel == 'poly':
            param_str += f", Degree: {best_degree}"

        # Print best params
        print(f'\nBest Parameters for {best_kernel.capitalize()} Kernel: {{{param_str}}}')
        print(f'Best Cross-Validation Accuracy: {grid_search.best_score_:.3f}')
        
        # Evaluate the model with the best parameters found
        self.__evaluate_svm(best_kernel, C=best_C, gamma=best_gamma if best_kernel != 'linear' else 'auto', degree=best_degree if best_kernel == 'poly' else 3)