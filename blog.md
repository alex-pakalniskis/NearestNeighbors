# Build a Lo-Fi Cannabis Strain Recommendation System with Python

## Introduction
Recommendation systems. Talk about use cases for recommendation systems and some examples companies.  

k-Nearest Neighbors as a lo-fi recommendation model. 

Applied example: Cannabis strain recommendation

## Methods
### Cannabis Strain Data
Data were scraped from the [Weedmaps Cannabis Strain API](https://api-g.weedmaps.com/wm/v1/strains). Talk about scraping process. Code is available [here](https://github.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/tree/master/code) to harvest and clean the data.

### Simulating User Preference Inputs

### k-Nearest Neighbors Algorithm
![knn](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/KnnClassification.svg/200px-KnnClassification.svg.png)

Distance metrics: Euclidiean for numeric and Jaccard for boolean. Include some LaTeX for pretty math formulas.

Euclidean distance for numeric attributes in n-dimensions for points p, q

$$d(\textbf{p,q}) = \sqrt{\sum_{i=1}^{n} (p_{i} - q_{i})^2}$$

Jaccard index for binary attributes in n-dimensions for p, q

$$J(\textbf{p,q}) = \frac{\vert{p\cap{q}}\vert}{{\vert{p\cup{q}}\vert}} = \frac{\vert{p\cap{q}}\vert}{\vert{p}\vert + \vert{q}\vert - \vert{p\cap{q}}\vert}$$

## Results and Discussion
My implementation
Include code chunks

Scikit-learn implementation
Include code chunks

## Conclusion
Brief closing thoughts