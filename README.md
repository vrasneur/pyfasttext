# pyfasttext
Yet another Python binding for [fastText](https://github.com/facebookresearch/fastText).

The binding supports Python 2.7 and Python 3. It requires Cython.
Numpy is also a dependency, but is optional.

`pyfasttext` has been tested successfully on Linux and Mac OS X.

Table of Contents
=================

   * [pyfasttext](#pyfasttext)
      * [Installation](#installation)
         * [Simplest way to install pyfasttext: use pip](#simplest-way-to-install-pyfasttext-use-pip)
         * [Cloning](#cloning)
         * [Requirements for Python 2.7](#requirements-for-python-27)
         * [Building and installing](#building-and-installing)
            * [Building and installing without Numpy](#building-and-installing-without-numpy)
      * [Usage](#usage)
         * [How to load the library?](#how-to-load-the-library)
         * [How to load an existing model?](#how-to-load-an-existing-model)
         * [Word representation learning](#word-representation-learning)
            * [Training using Skipgram](#training-using-skipgram)
            * [Training using CBoW](#training-using-cbow)
         * [Word vectors](#word-vectors)
            * [Word vectors access](#word-vectors-access)
               * [Vector for a given word](#vector-for-a-given-word)
                  * [Numpy ndarray](#numpy-ndarray)
               * [Get the number of words in the model](#get-the-number-of-words-in-the-model)
               * [Get all the word vectors in a model](#get-all-the-word-vectors-in-a-model)
                  * [Numpy ndarray](#numpy-ndarray-1)
            * [Misc operations with word vectors](#misc-operations-with-word-vectors)
               * [Word similarity](#word-similarity)
               * [Most similar words](#most-similar-words)
               * [Analogies](#analogies)
         * [Text classification](#text-classification)
            * [Supervised learning](#supervised-learning)
            * [Get all the labels](#get-all-the-labels)
            * [Get the number of labels](#get-the-number-of-labels)
            * [Prediction](#prediction)
               * [Labels and probabilities](#labels-and-probabilities)
                  * [Normalized probabilities](#normalized-probabilities)
               * [Labels only](#labels-only)
            * [Quantization](#quantization)
         * [Misc utilities](#misc-utilities)
            * [Show the model (hyper)parameters](#show-the-model-hyperparameters)
            * [Extract labels or classes from a dataset](#extract-labels-or-classes-from-a-dataset)
               * [Extract labels](#extract-labels)
               * [Extract classes](#extract-classes)
         * [Exceptions](#exceptions)

## Installation

To compile `pyfasttext`, make sure you have a compiler with C++11 support.

### Simplest way to install pyfasttext: use pip

Just type this line:

```bash
pip install pyfasttext
```

### Cloning

`pyfasttext` uses git [submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules).  
So, you need to add the `--recursive` option when you clone the repository.

```bash
git clone --recursive https://github.com/vrasneur/pyfasttext.git
cd pyfasttext
```

### Requirements for Python 2.7

Python 2.7 support relies on the [future](http://python-future.org) module: `pyfasttext` needs `bytes` objects, which are not available natively in Python2.  
You can install the `future` module with `pip`.

```bash
pip install future
```

### Building and installing

```bash
python setup.py install
```

#### Building and installing without Numpy

`pyfasttext` can export word vectors as `numpy` `ndarray`s, however this feature can be disabled at compile time.

To compile without `numpy`, pyfasttext has a `USE_NUMPY` environment variable. Set this variable to 0 (or empty), like this:

```bash
USE_NUMPY=0 python setup.py install
```

## Usage

### How to load the library?

```python
>>> from pyfasttext import FastText
```

### How to load an existing model?

```python
>>> model = FastText('/path/to/model.bin')
```

or

```python
>>> model = FastText()
>>> model.load_model('/path/to/model.bin')
```

### Word representation learning

You can use all the options provided by the `fastText` binary (`input`, `output`, `epoch`, `lr`, ...).  
Just use keyword arguments in the training methods of the `FastText` object.

#### Training using Skipgram

```python
>>> model = FastText()
>>> model.skipgram(input='data.txt', output='model', epoch=100, lr=0.7)
```
#### Training using CBoW

```python
>>> model = FastText()
>>> model.cbow(input='data.txt', output='model', epoch=100, lr=0.7)
```

### Word vectors

#### Word vectors access

##### Vector for a given word

By default, a single word vector is returned as a regular Python array of floats.

```python
>>> model['dog']
array('f', [-1.308749794960022, -1.8326224088668823, ...])
```

###### Numpy ndarray

The `get_numpy_vector()` method returns the word vector as a `numpy` `ndarray`.

```python
>>> model.get_numpy_vector('dog')
array([-1.30874979, -1.83262241, ...], dtype=float32)
```

If you want a normalized vector (i.e. the vector divided by its norm), there is an optional boolean parameter named `normalized`.

```python
>>> model.get_numpy_vector('dog', normalized=True)
array([-0.07084749, -0.09920666, ...], dtype=float32)
```

##### Get the number of words in the model

```python
>>> model.nwords
500000
```

##### Get all the word vectors in a model

```python
>>> for word in model.words:
...   print(word, model[word])
```

###### Numpy ndarray

If you want all the word vectors as a big `numpy` `ndarray`, you can use the `numpy_normalized_vectors` member. Note that all these vectors are *normalized*.

```python
>>> model.nwords
500000
>>> model.numpy_normalized_vectors
array([[-0.07549749, -0.09407753, ...],
       [ 0.00635979, -0.17272158, ...],
       ..., 
       [-0.01009259,  0.14604086, ...],
       [ 0.12467574, -0.0609326 , ...]], dtype=float32)
>>> model.numpy_normalized_vectors.shape
(500000, 100) # (number of words, dimension)
```

#### Misc operations with word vectors

##### Word similarity

```python
>>> model.similarity('dog', 'cat')
0.75596606254577637
```

##### Most similar words

```python
>>> model.nearest_neighbors('dog', k=2)
[('dogs', 0.7843924736976624), ('cat', 75596606254577637)]
```

##### Analogies

The `most_similar()` method works similarly as the one in [gensim](https://radimrehurek.com/gensim/models/keyedvectors.html).

```python
>>> model.most_similar(positive=['woman', 'king'], negative=['man'], k=1)
[('queen', 0.77121970653533936)]
```

### Text classification

#### Supervised learning

```python
>>> model = FastText()
>>> model.supervised(input='/path/to/input.txt', output='/path/to/model', epoch=100, lr=0.7)
```

#### Get all the labels

```python
>>> model.labels
['LABEL1', 'LABEL2', ...]
```

#### Get the number of labels

```python
>>> model.nlabels
100
```

#### Prediction

To obtain the `k` most likely labels from test sentences, there are multiple `predict_*()` methods.  
If you want to obtain all the possible labels, use `None` for `k`.

##### Labels and probabilities

If you have a list of strings (or an iterable object), use this:

```python
>>> model.predict_proba(['first sentence', 'second sentence'], k=2)
[[('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)], [('LABEL2', 1.0), ('LABEL3', 1.953126549381068e-08)]]
```
If your test data is stored inside a file, use this:

```python
>>> model.predict_proba_file('/path/to/test.txt', k=2)
[[('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)], [('LABEL2', 1.0), ('LABEL3', 1.953126549381068e-08)]]
```

If you want to test a single string, use this:

```python
>>> model.predict_proba_single('first sentence', k=2)
[('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)]
```

###### Normalized probabilities

For performance reasons, fastText probabilities often do not sum up to 1.0.

If you want normalized probabilities (where the sum is closer to 1.0 than the original probabilities), you can use the `normalized=True` parameter in all the methods that output probabilities (`predict_proba()`, `predict_proba_file()` and `predict_proba_single()`).

```python
>>> sum(proba for label, proba in model.predict_proba_single('this is a sentence that needs to be classified', k=None))
0.9785203068801335
>>> sum(proba for label, proba in model.predict_proba_single('this is a sentence that needs to be classified', k=None, normalized=True))
0.9999999999999898
```

##### Labels only

If you have a list of strings (or an iterable object), use this:

```python
>>> model.predict(['first sentence', 'second sentence'], k=2)
[['LABEL1', 'LABEL3'], ['LABEL2', 'LABEL3']]
```
If your test data is stored inside a file, use this:

```python
>>> model.predict_file('/path/to/test.txt', k=2)
[['LABEL1', 'LABEL3'], ['LABEL2', 'LABEL3']]
```

If you want to test a single string, use this:

```python
>>> model.predict_single('first sentence', k=2)
['LABEL1', 'LABEL3']
```

#### Quantization

Use keyword arguments in the `quantize()` method.

```python
>>> model.quantize(input='/path/to/input.txt', output='/path/to/model')
```

You can load quantized models using the `FastText` constructor or the `load_model()` method.

### Misc utilities

#### Show the model (hyper)parameters

```python
>>> model.args
{'bucket': 11000000,
 'cutoff': 0,
 'dim': 100,
 'dsub': 2,
 'epoch': 100,
...
}
```

#### Extract labels or classes from a dataset

You can use the `FastText` object to extract labels or classes from a dataset.
The label prefix (which is `__label__` by default) is set using the `label` parameter in the constructor.

If you load an existing model, the label prefix will be the one defined in the model.

```python
>>> model = FastText(label='__my_prefix__')
```

##### Extract labels

There can be multiple labels per line.

```python
>>> model.extract_labels('/path/to/dataset1.txt')
[['LABEL2', 'LABEL5'], ['LABEL1'], ...]
```

##### Extract classes

There can be only one class per line.

```python
>>> model.extract_classes('/path/to/dataset2.txt')
['LABEL3', 'LABEL1', 'LABEL2', ...]
```

### Exceptions

The `fastText` source code directly calls exit() when something wrong happens (e.g. a model file does not exist, ...).

Instead of exiting, `pyfasttext` raises a Python exception (`RuntimeError`).

```python
>>> import pyfasttext
>>> model = pyfasttext.FastText('/path/to/non-existing_model.bin')
Model file cannot be opened for loading!
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "src/pyfasttext.pyx", line 124, in pyfasttext.FastText.__cinit__ (src/pyfasttext.cpp:1800)
  File "src/pyfasttext.pyx", line 348, in pyfasttext.FastText.load_model (src/pyfasttext.cpp:5947)
RuntimeError: fastext tried to exit: 1
```
