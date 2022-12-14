# Manipulating Word Embeddings in Vector-Space

This repo is all about vector spaces and how to manipulate word vectors. Learning how word vectors are represented in natural language processing is fundamental to all NLP applications.

We will learn how to create word vectors that capture dependencies between words, then visualize their relationships in two dimensions using PCA. Specifically we will:

- Predict analogies between words.
- Use PCA to reduce the dimensionality of the word embeddings and plot them in two dimensions.
- Compare word embeddings by using a similarity measure (the cosine similarity).
- Understand how these vector space models work.

I achieved the following learning goals:
* Covariance matrices
* Dimensionality reduction
* Principal component analysis
* Cosine similarity
* Euclidean distance
* Co-occurrence matrices
* Vector representations
* Vector space models

# Vector Space Models
A vector space model is a mathematical model for representing text documents as vectors of numerical values. It is used in information retrieval and text mining. Vector space models capture semantic meaning and relationships between words.

# Co-occurrence Matrix
A co-occurrence matrix is a matrix that can be used to measure the similarity of two items based on how often they occur together. <br>
In *word by word* design, the co-occurrence of two different words is the number of times that they appear in your corpus together within a certain word distance k. <br>
For a *word by document* design, you will count the number of times the words from your vocabulary appear in documents corpora that belong to specific categories. You can represent the categories, as a vector v, and also compare categories as follows by doing a simple plot.

<img src="images/plot of comparison of categories in vector.jpg">

# Euclidean Distance
A Euclidean distance is a similarity metric used in machine learning and is often used to measure the similarity between two vectors. In a vector space model, the Euclidean distance between two vectors is the length of the vector difference between them.<br>
For two points A(A1, A2) and B(B1, B2), the euclidean distance is:

$$d(B, A) = \sqrt{(B_1 - A_1)^2 + (B_2 - A_2)^2}$$

In general, for *n-dimensional vectors*, the *Euclidean distance* is given as:<br>

<img src="images/euclidean distance for n-dimention vector space.PNG"><br>

From algebra, this formula is known as the *norm* of the difference between the vectors that you are comparing.

<img src="images/Euclidean distance as norm of difference between 2 vectors from algebra.PNG">

# Cosine Similarity
Cosine similarity is a measure of similarity between two vectors that measures the cosine of the angle between them. It is a popular metric for measuring similarity between two vectors.

Cosine similarity is a more effective metric than Euclidean distance for measuring similarity between vectors. This is because cosine similarity is less sensitive to the magnitude of the vectors, and is therefore more robust to changes in vector size. Additionally, cosine similarity is more effective at capturing the similarity between vectors that are close together in terms of angle (i.e. more similar).<br>
However, in general, cosine similarity is more effective when working with high-dimensional data, while Euclidean distance is more effective when working with low-dimensional data.<br>
*for eg*:  If you have two documents of very different sizes, then taking the Euclidean distance is not ideal. The cosine similarity used the angle between the documents and is thus not dependent on the size of the corpuses.<br>
One of the issues with euclidean distance is that it is not always accurate and sometimes we are not looking for that type of similarity metric. *For example*, when comparing large documents to smaller ones with euclidean distance one could get an inaccurate result. Look at the diagram below:

<img src="images/Euclidean distance vs Cosine similarity.PNG">
<img src="images/norm and dot product used in cosine similarity.PNG">
<img src="images/cosine similarity.jpg">

Hence, given two vectors, v and w the cosine similarity, cos(β) is defined as:

$$ cosine\  similarity =\cos(\beta) = \frac{v \cdot w}{\left||v\right|| \left||w\right||}$$

where ||v|| and ||w|| are the Euclidean norms of vectors v and w, and v · w is the dot product of vectors v and w. <br>Cosine similarity gives values between 0 and 1.

<img src="images/cosine similarity between similar and dissimilar vector.JPG">

# Manipulating Words in Vector Spaces
You can use word vector representations to manipulate words in vector spaces to extract patterns and identify certain structures. There are different techniques used to discover the underlying patterns in a set of data and the relationships between words and to identify the underlying structure of a document set.<br>
*for eg*: finding the closest or similar word to a given word, and given vector.<br>
<a href="">Practice Manipulating Word Embeddings Here</a>

# Visualization and PCA
A way to reduce the very high dimensions of the vectors to two dimensions while preserving as much variance as possible is to use a technique called *Principal Component Analysis (PCA)*. PCA is a statistical procedure that finds the directions (components) that maximize the variance in a dataset. In other words, it identifies the underlying structure in the data. Once you have identified these components, you can represent the data in a lower-dimensional space.

Hence, PCA is a a *dimensionality reduction* technique to reduce the dimension of data while preserving relationships among vectors so that it can be visualized and represented more easily using the plot.
* It's very helpful for visualizing your data to check if your representation is capturing relationships among words. *For example*, if you are using a 2D representation, are similar words close together in the space?
* The benefits of dimensionality reduction are that it can make data easier to work with, and it can help improve the performance of machine learning algorithms.

<img src="images/visualization of word vector.JPG">
<img src="images/result of pca ploting vector in 2d.JPG">

# PCA Algorithm
<img src="images/pca working.JPG"/>

**Eigenvectors**: the directions along which the data varies the most. They are resulting vectors, also known as the *uncorrelated features* of your data.<br>
**Eigenvalues**: the amount of variance in the data along those direction or the amount of information retained by each new feature. You can think of it as the variance in the eigenvector. 

<img src="images/pca algorithm.JPG"/>

The steps for computing PCA are:<br>
1) Choose the number of components, k, that you want to keep.
2) Calculate the covariance matrix of your data, X
3) Calculate the eigenvectors and eigenvalues of the covariance matrix
4) Sort the eigenvectors by descending order of the eigenvalues
5) Choose the first k eigenvectors
6) Transform the data into a lower-dimensional space using these k eigenvectors

## Conclusion
I learned how to create word vectors that capture dependencies between words, then visualize their relationships in two dimensions using PCA.<br>
I had the opportunity to apply all of the aforementioned concepts and skills into practice. It was a fantastic learning experience. I appreciate and am grateful for the chance.


Thank you, and happy learning!

---