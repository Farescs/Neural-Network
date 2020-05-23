import numpy as np
import matplotlib.pyplot as plt

class pre_processing:

    # Normalize the data so that it has a zero mean and a standard diviation of 1.
    def normalize(self, data_to_normalize):
        normalized_set = data_to_normalize

        mean = np.mean(normalized_set, 0)
        # Remove the mean of each feature.
        normalized_set -= np.mean(normalized_set, 0)

        stnd = np.std(normalized_set)
        # Divide by the standard diviation of each feature.
        normalized_set /= np.std(normalized_set)

        # Return the normalized data.
        return normalized_set, mean, stnd


    # Pricipal Component Analysis method
    def PCA(self, data, number_of_features=2):

        # Get the covarience matrix of the data.
        covarience_matrix = np.dot(data.T, data) / data.shape[0]

        # Find the eigenvalues and eigenvectors of the covarience matrix. eigh is used instead of eig because
        # it's slightly more preceise than eig, which helps with choosing the correct eigenvectors.
        eig_vals, eig_vecs = np.linalg.eigh(covarience_matrix)

        # Sort the eigenvalues in decreasing order.
        key = np.argsort(eig_vals)[::-1][:number_of_features]

        # Choose eigenvalues and eigenvectors.
        eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]

        # Finish rescaling the data with eigenvectors.
        U = np.dot(data, eig_vecs)

        # Return the rescaled matrix U.
        return U


    # Whitening
    def whitening(self):

        return 0

