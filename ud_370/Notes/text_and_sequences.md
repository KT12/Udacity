### Deep Models for Text and Sequences

#### Train Text Embedding Model
For text classification rare words tend to be the most important in text, i.e. 'retinopathy.'  Only appears in 0.001% of documents but poses a problem for machine learning.

#### Semantic Ambiguity
Another problem is using two different words to refer to the same thing. 'Cat' and 'kitty' refer to the same thing but the algorithm needs to learn they are related.  Then they can share parameters.

#### Unsupervised Learning
Turn to unsupervised learning (text without labels) to help solve the problem.  Try to predict a word's context, which will bring similar words closer.  Map words to small vectors which will be close to each other when they have similar meanings.

#### Embeddings
Embedding helps solve the sparsity problem.  Model can generalize from patterns.

#### Word2vec
Simple model which learns embeddings.  Maps each word in a sentence to an embedding.  Then uses embeddings to try to predict the context using logistic classifier.

#### t-sne
Can visuzalize data to see if related words remain close in distance by doing nearest-neighbor lookup.  Or reduce dimensionality by using t-sne (PCA removes too much information) and project in 2-D.

[t-sne paper](http://jmlr.csail.mit.edu/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

#### Word2vec details
Better to compare embeddings by cosine distance than L2 because the length of the vector is not important to classification.

How to predict words?  Take embedding and run through linear model.  Computing softmax entire corpus is expensive.  Instead, use sampled softmax by chossing a handful of random non-target words.  Faster, but no cost in performance.

#### Analgoies
Word2vec allows semantic and syntactic analogies to be expressed in math.

##### Additional resources outside of the lecture
[C-BOW paper](https://arxiv.org/pdf/1411.2738v3.pdf)
[Word Embedding Presentation](https://www.youtube.com/watch?v=D-ekE-Wlcds)
[Word2vec using Gensim](https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-python/)
[Blog post on Skip Grams](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

Notes taken by [@KT12](https://github.com/KT12)