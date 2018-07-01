# smile
tablesaw  (https://github.com/jtablesaw/tablesaw) / smile (https://github.com/haifengl/smile) integration code

This code is tested with Tablesaw version 0.20, as of July 1, 2018

This project provides integration between Tablsaw, a Java-based dataframe and visualization package, and Smile, a Java-based machine learning library. Currently, the integration provides many common ML algorithms, including

- Linear Regression
- Logistic Regression
- Decision Trees
- KNN Classifiers
- Linear Discriminant Analysis
- Random Forests
- Non-hierarchical clustering (K-means, G-means, and X-means)
- Hierarchical clustering
- Principal components
- Association mining 
- Frequent item sets 

This code is not currently in maven central. To use it, you can clone the repo and build with maven after you've installed tablesaw on your machine. 