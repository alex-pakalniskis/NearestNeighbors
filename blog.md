---
layout: post
title: "Build a Lo-Fi Cannabis Strain Recommendation System with Python"
author: "Alex Pakalniskis"
categories: journal
tags: [cannabis, recommendation, strain, nearest neighbors, euclidean distance, jaccard index, data science, python, pandas, scipy, scikit-learn]
image: cannabis.jpg 

---

The suggestions for your next favorite playlist, bingeworthy show, or late-night e-commerce splurge are largely powered by recommendation systems<sup>[1](https://en.wikipedia.org/wiki/Recommender_system)</sup>. Tech companies like Google, Amazon, Twitter, and Netflix make use of recommendation systems to tailor content and product suggestions to the user<sup>[2](https://aisel.aisnet.org/cgi/viewcontent.cgi?article=1146&context=icis2004)</sup>, i.e. you and me. However, the ethics of recommendation systems remains a contentious topic<sup>[3](https://link.springer.com/chapter/10.1007/978-3-642-13226-1_10),[4](https://www.usenix.org/system/files/conference/soups2014/soups14-paper-zhang.pdf)</sup>. 

Despite on-going debates, innovations in recommendation algorithms remains a top research priority<sup>[5](https://link.springer.com/article/10.1007/s12652-018-0928-7),[6](https://www.sciencedirect.com/science/article/pii/S0167923618301970),[7](https://ieeexplore.ieee.org/abstract/document/8616805)</sup>. While more sophisticated methods are regularly developed, the k-Nearest Neighbors<sup>[8](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)</sup> algorithm remains a foundational machine learning technique which can be applied to drive a basic recommendation system<sup>[9](https://www.sciencedirect.com/science/article/pii/S221083271400026X)</sup>. The remainder of this post will describe how to build a k-Nearest Neighbors-powered Cannabis strain recommendation system using Python 3<sup>[10](https://en.wikipedia.org/wiki/Python_(programming_language))</sup>.

## Methods
### Cannabis Strain Data
![strain wheel](https://resize.mantisadnetwork.com/mantis-ad-network/image/fetch/w_500,q_75,c_limit,f_jpg/http://uploads.medicaljane.com/wp-content/uploads/2016/06/flavorwheel.png)

Medicinal and recreational cannabis legislation are slowly gaining approval across US states. While cannabis remains illegal (and punishable by imprisonment) at the federal-level, many state cannabis industries are booming<sup>[11](https://www.ocregister.com/2020/03/10/california-passes-1-billion-in-cannabis-tax-revenue-two-years-after-launching-legal-market/)</sup>. To meet the growing demand for consumers and retailers, cannabis tech companies such as Weedmaps<sup>[12](https://weedmaps.com/)</sup> and Leafly<sup>[13](https://www.leafly.com/)</sup> provide information services related to cannabis strains and products, and dispensary inventory. While many services are proprietary, much of this cannabis data is publically available through web APIs<sup>[14](https://en.wikipedia.org/wiki/Web_API)</sup>. For education purposes only, cannabis strain data were scraped from the Weedmaps Cannabis Strain API<sup>[15](https://api-g.weedmaps.com/wm/v1/strains)</sup> in early May 2020. The Weedmaps data set contains over 50 flavor and effect attributes (boolean features) for more than 300 cannabis strains. A helper class was written to facilitate the scraping process and another class for processing the API JSON data into a machine learning-friendly CSV format. Code to replicate the scraping and data cleaning process is available on GitHub<sup>[16](https://github.com/Build-Week-Med-Cabinet-2-MP/bw-med-cabinet-2-ml/tree/master/code)</sup>.

### Simulating User Preference Inputs
![user input](https://public-media.interaction-design.org/images/ux-daily/56e2cfdabdb9e.jpg)

User preference for cannabis strain flavors and effects were simulated using the numpy array manipulation library<sup>[17](https://numpy.org/)</sup>. A random seed was set for reproducibility. See `examples` for the full code implementation<sup>[18](https://github.com/alex-pakalniskis/NearestNeighbors/tree/master/examples)</sup>.

### k-Nearest Neighbors Algorithm

The k-Nearest Neighbor algorithm<sup>[18](http://scholarpedia.org/article/K-nearest_neighbor)</sup> utilizes an analyst-supplied distance metrics to calculate the nearest (or most similar) data points to a given input. Distance is calculated in an n-dimensional hyperspace formed by the data attributes being analyzed. Furthermore, selection of a suitable distance metric often depends on the type of data under inspection. For this cannabis strain example, the k-Nearest Neighbors model will be used to return the k-most similar (nearest) strains to the simulated user preference inputs for flavor and effects.

A common choice for Nearest Neighbors metric is Euclidean distance<sup>[19](https://en.wikipedia.org/wiki/Euclidean_distance)</sup>, most suited for numeric attributes. Mathematically, the euclidean distance in n-dimensions for points p, q with i features is as follows:

$$d(\textbf{p,q}) = \sqrt{\sum_{i=1}^{n} (p_{i} - q_{i})^2}$$

Each of the `i` flavor and effect attributes would be compared for p and q. The cannabis strain data is not optimally suited for euclidean distance as strain attributes are stored as booleans. A more appropriate distance metric to model similary between strains would be the Jaccard index<sup>[20](https://en.wikipedia.org/wiki/Jaccard_index)</sup>. Mathematically, the Jaccard index for p, q is represented as:

$$J(\textbf{p,q}) = \frac{\vert{p\cap{q}}\vert}{\vert{p}\vert + \vert{q}\vert - \vert{p\cap{q}}\vert}$$ 

In our applied example, the metric calculates the intersection of strain p and strain q attributes divided by the union of all strain p and q attributes. My class implementation uses scipy<sup>[21](https://docs.scipy.org/doc/scipy/reference/spatial.distance.html)</sup> scientific computing library to calculate either distance metrics. 

## Results and Discussion
### My Implementation
Robust, heavily tested implementations of k-Nearest Neighbors can be found in the scikit-learn machine learning library<sup>[22](https://scikit-learn.org/stable/)</sup>, building this model from scratch was enlightening. A simple class was built with `fit` and `predict` methods loosely which loosely mirror those of scikit-learn model implementation. Shown below is an example usage run on Ubuntu 18.04 returning the 10 nearest cannabis strains to a simulated user input:

```bash
foo@bar:~$ git clone https://github.com/alex-pakalniskis/NearestNeighbors.git
foo@bar:~$ cd NearestNeighbors
foo@bar:~$ pipenv install
foo@bar:~$ pipenv shell
foo@bar:~$ python3 examples/example-usage.py
                   ids distances
221    Peyote Critical       0.7
281  Strawberry Banana  0.818182
166           Larry OG  0.818182
297  Super Silver Haze  0.818182
89           Diablo OG  0.818182
90          Diamond OG  0.818182
28           Apollo 13  0.818182
292      Sundae Driver  0.818182
290       Sugar Cookie  0.818182
172         Lemon Drop  0.818182
```

### Scikit-Learn Implementation

The polar opposite of a solo coding project, the scikit-learn machine learning library is the collaborative effort of thousands of dedicated open-source contributors from around the world. The `NearestNeighbors` implementation<sup>[23](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors)</sup> in scikit-learn eschews the typical `fit/predict` paradigm, instead opting for a `fit/kneighbors` pattern. Shown below are cannabis strain recommendation results from simulated user preference input identical to those used with my implementation. 

```bash
foo@bar:~$ python3 examples/scikitlearn-implementation.py
              Name  Distance
0  Peyote Critical       0.7
1             FPOG  0.818182
2    Exodus Cheese  0.818182
3          Afgooey  0.818182
4         Larry OG  0.818182
5    Grease Monkey  0.818182
6      Jet Fuel OG  0.818182
7        Harlequin  0.818182
8      Dutch Treat  0.818182
9       Jilly Bean  0.818182
```

## Discussion and Conclusion
While the cannabis strain nearest to the simulated user input was the same for both `NearestNeighbors` implementations, most of the other 9 strain recommendations varied considerably. Larry OG was the only other common strain between the two recommendations. These results suggest that the two implementations likely differ in some critical ways. The parameter-rich scikit-learn class likely employs default values which my implementation lacks. Additionally, there may be a large number of strains which lie 0.81818182 away from the user input. Future projects may look into comparing different boolean metrics or more nuanced recommendation scheme to optimize k-Nearest Neighbors-powered cannabis strain suggestions. Likewise, assigning some hierarchical relationships between attributes (Lemon as a subset of Citrus) may assist with improving relevance of suggestions.

For questions or comments related to this blog, feel free to contact me at `alexpakalniskis3@gmail.com`.