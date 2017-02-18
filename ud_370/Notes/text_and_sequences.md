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
[Word2vec using Gensim](https://codesachin.wordpress.com/2015/10/09/generating-a-word2vec-model-from-a-block-of-text-using-gensim-Writes input `X` into python/)
[Blog post on Skip Grams](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

#### Sequences of Various Length
Need good model trained on a lot of text.
Question, how do you deal with sequences of varying length?

#### RNN's
Covnet uses shared parameters over space to extract patterns from an image.  Recurrent networks do this over time in stead of space.  If the sequence is reasonably stationary, can use the same classifier at each point in time.  However, also want to take into account the past.  Can use the state of the previous classifier as a summary of what happened before.  Instead of using thousands of nodes, have input at each step and recurrent connection from the past.

#### Backprop through time
Need to back prop through time to beginning of the sequence (or what is affordable).  However, SGD does not like correlated updates.  This causes instability in the gradients, going to infinity or 0.

#### Exploding and vanishing gradients
For gradients that go to infiniti, a simple hack is gradient clipping.

$$$ \Delta W \leftarrow \Delta W \left(\frac{\Delta_\max}{\max(|\Delta W|,\Delta_\max)}\right) $$$

Gradients that go to 0 are a problem because it means the distant past is not used in classification.  Need to address this problem in a different way.

#### LSTM
LSTM = long short-term memory.  LSTM's replace a neural net with a 'cell.'

#### Memory Cell
Memory is `M` matrix of values
`X` is input which may be written to `M`, depending on gate
`Y` is output which may be read from `M`, depending on gate
Another gate determines whether `M` is retained

#### LTSM Cell
Now imagine each gate is not binary, but $\in [0.0, 1.0]$

and is differentiable and continuous (i.e. sigmoid function).  Then we are able to back propragate through it.

#### LTSM Cell 2
Gates are each controlled by logistic regression.  There is an additional $tanh \in [-1.0, 1.0]$
before the output gate.  Every step is continuous and differentiable.  Optimization is much easier as gradient vanishing doesn't occur.  LSTM helps the recurrent network can remember what it needs and discard what it doesn't.

#### Regularization
L2 reg works.
Dropout regularization works, but only on the input and outputs.  Do not use it on the recurrent connections.

### Beam Search
Applications of RNN?  Can predict consecutive words or letters.  Naive prediction would choose the next most likely element.  Instead, can sample multiple times and choose the sequence with the greatest possibility.  Prevents making one bad decision and being stuck with it.  However, the solution space is very large, so better to prune and only keep the most likely candidates.

Notes taken by [@KT12](https://github.com/KT12)