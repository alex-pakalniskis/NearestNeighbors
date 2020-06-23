class NearestNeighbors:
    """
    Unsupervised nearest neighbors algorithm
    """

    def fit(self, X):
        """
        Fit nearest neighbors model to an input data set

        Args:
            X: Input data
        
        Returns:
            None
        
        Usage:
            >>> model = NearestNeighbors()
            >>> model.fit(some_data)
        """
        
        pass

    def predict(self, y, k=3):
        """
        Predict nearest neighbors to a user-supplied input

        Args:
            y: User input data
            k: Number of neighbors to return

        Returns:
            k neighbors from original data set

        Usage:
            >>> model = NearestNeighbors()
            >>> model.fit(some_data)
            >>> user_input = "Value to find nearby neighbors"
            >>> print(model.predict(user_input, k=5))

        """

        pass

if __name__ == "__main__":
    pass