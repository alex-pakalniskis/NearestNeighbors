from scipy.spatial import distance
import pandas as pd

class NearestNeighbors:
    """
    Unsupervised nearest neighbors algorithm
    """

    def fit(self, X):
        """
        Fit nearest neighbors model to an input data set

        Args:
            X: pandas DataFrame, first column contains labels
        
        Returns:
            None
        
        Usage:
            >>> model = NearestNeighbors()
            >>> model.fit(some_data)
        """
        
        self.X = X
        self.X_no_labels = self.X.drop(self.X.columns[0], axis=1)
        self.ids = []
        self.distances = []

    def predict(self, y, k=3):
        """
        Predict nearest neighbors to a user-supplied input

        Args:
            y: User input data
            k: Number of neighbors to return

        Returns:
            k nearest neighbors and euclidian distances to y

        Usage:
            >>> model = NearestNeighbors()
            >>> model.fit(some_data)
            >>> user_input = "Value to find nearby neighbors"
            >>> print(model.predict(user_input, k=5))

        """

        assert len(y) == len(self.X.columns) - 1

        for i in range(len(self.X)):
            self.distances.append(distance.euclidean(y, 
                                                     self.X_no_labels.iloc[i]))
            self.ids.append(self.X[self.X.columns[0]].iloc[i])

        self.distance_df = pd.DataFrame([self.ids, self.distances], 
                                        index=["ids","distances"]).T
        return self.distance_df.sort_values("distances")[:k]



if __name__ == "__main__":
    pass