import sys
sys.path.append('./')
from NearestNeighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Load cannabis strain dataset scraped from Weedmaps
STRAIN_DATA_URL = "https://raw.githubusercontent.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/master/data/CLEAN_WMS_2020_05_24.csv"
strain_df = pd.read_csv(STRAIN_DATA_URL)
strain_df.drop(columns="description", inplace=True)

# Instantiate a nearest neighbors model
model = NearestNeighbors()

# Fit the model to the strain data
model.fit(strain_df)

# Create a fake user-input which mimics flavor/effect preferences (same size as single row of strain data set)
N=57
K=6
arr = np.array([1] * K + [0] * (N-K))
np.random.seed(420)
np.random.shuffle(arr)

# Return 10 nearest neighbors to fake user-input
predictions = model.predict(arr, 10, "jaccard")


if __name__ == "__main__":
    print(predictions)