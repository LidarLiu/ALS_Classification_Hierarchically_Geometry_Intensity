# Hierarchical classification of urban ALS data by using geometry and intensity information
We proposes a hierarchical classification method by using geometry and intensity information of urban ALS data separately. The method uses supervised learning for stable geometry information and unsupervised learning for fluctuating intensity information.

### Introduction of files
Joint_XXX_unsupervised.ipynb is the the core part of our experiment 

Utils.py include some useful function needed in Joint_XXX_unsupervised.ipynb

### Feature extraction
[**cloudcompare**](https://www.danielgm.net/cc/) is used for point normal estimation.  

[**PCL**](http://pointclouds.org/) is used for groud extraction and fpfh calculation.

Height (h) is derived using fuction 'feature_extraction' in Utils.py
