from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

# Strain data set
STRAIN_DATA_URL = "https://raw.githubusercontent.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/master/data/CLEAN_WMS_2020_05_24.csv"
df = pd.read_csv(STRAIN_DATA_URL)
df.drop(columns="description", inplace=True)
df_no_names = df.drop(columns="name")

# Simulated user input for strain recommendation system
N=57
K=6
arr = np.array([1] * K + [0] * (N-K))
np.random.seed(420)
np.random.shuffle(arr)

# Invoke model and fit to strain data
model = NearestNeighbors(n_neighbors=10,metric="jaccard")
model.fit(df_no_names)

# Calculate nearest neighbors strains based on user input for flavors and effects (arr)
results = model.kneighbors(arr.reshape(1, -1))

# Output results as a pandas DataFrame
names = []
distances = []
for i in results[1][0]:
    names.append(df.iloc[i]["name"])
for i in results[0][0]:
    distances.append(i)
results_df = pd.DataFrame([names, distances], index=["Name","Distance"]).T
print(results_df)

