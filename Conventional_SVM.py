import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class Conventional_SVM:
    def __init__(self, train, test):
        self.train = pd.read_csv(train)
        self.test = pd.read_csv(test)
        self.feature_df = self.train[[
                'Payload_ratio', 'flow duration', 'max_Length_of_IP_packets', 'std_Length_of_IP_packets', 
                'var_Length_of_IP_packets', 'max_Length_of_TCP_payload', 'std_Length_of_TCP_payload', 
                'var_Length_of_TCP_payload', 'std_Length_of_TCP_packet_header', 'var_Length_of_TCP_packet_header', 
                'max_Length_of_TCP_segment(packet)', 'std_Length_of_TCP_segment(packet)', 'var_Length_of_TCP_segment(packet)', 
                'max_Time_difference_between_packets_per_session', 'std_Time_difference_between_packets_per_session', 
                'max_Interval_of_arrival_time_of_forward_traffic', 'max_Interval_of_arrival_time_of_backward_traffic', 
                'std_Interval_of_arrival_time_of_backward_traffic', 'std_backward_pkt_length', 'IPratio', 'Domain'
            ]]

    def define_features(self, label):
        self.X_train = np.asarray(self.feature_df)
        self.y_train = np.asarray(self.train[label])

        self.X_test = np.asarray(self.test[self.feature_df.columns])
        self.y_test = np.asarray(self.test[label])

    def __evaluate_svm(self, kernel, C=2, gamma='auto', degree=3):
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

        print(f'\nBest Parameters: {grid_search.best_params_}')
        print(f'Best Cross-Validation Accuracy: {grid_search.best_score_}')

        # Train with the best parameters
        best_params = grid_search.best_params_
        best_kernel = best_params['svc__kernel']
        best_C = best_params['svc__C']
        best_gamma = best_params['svc__gamma']
        best_degree = best_params.get('svc__degree', 3)  # Default degree to 3 if not present

        # Evaluate the model with the best parameters found
        self.__evaluate_svm(best_kernel, C=best_C, gamma=best_gamma, degree=best_degree)