# pyfasttext
Yet another Python binding for [fastText](https://github.com/facebookresearch/fastText).

The binding supports Python 2.6, 2.7 and Python 3. It requires [Cython](http://cython.org/) and [cysignals](http://cysignals.readthedocs.io/en/latest/).  
[Numpy](http://www.numpy.org/) is also a dependency, but is optional.

`pyfasttext` has been tested successfully on Linux and Mac OS X.

Table of Contents
=================

   * [pyfasttext](#pyfasttext)
   * [Table of Contents](#table-of-contents)
      * [Installation](#installation)
         * [Simplest way to install pyfasttext: use pip](#simplest-way-to-install-pyfasttext-use-pip)
            * [Possible compilation error](#possible-compilation-error)
         * [Cloning](#cloning)
         * [Requirements for Python 2.7](#requirements-for-python-27)
         * [Building and installing manually](#building-and-installing-manually)
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
               * [Words for a given vector](#words-for-a-given-vector)
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
               * [Is a model quantized?](#is-a-model-quantized)
         * [Subwords](#subwords)
            * [Get the subwords](#get-the-subwords)
            * [Get the subword vectors](#get-the-subword-vectors)
         * [Sentence and text vectors](#sentence-and-text-vectors)
            * [Unsupervised models](#unsupervised-models)
            * [Supervised models](#supervised-models)
         * [Misc utilities](#misc-utilities)
            * [Show the module version](#show-the-module-version)
            * [Show fastText version](#show-fasttext-version)
            * [Show the model (hyper)parameters](#show-the-model-hyperparameters)
            * [Show the model version number](#show-the-model-version-number)
            * [Extract labels or classes from a dataset](#extract-labels-or-classes-from-a-dataset)
               * [Extract labels](#extract-labels)
               * [Extract classes](#extract-classes)
         * [Exceptions](#exceptions)
         * [Interruptible operations](#interruptible-operations)

## Installation

To compile `pyfasttext`, make sure you have a compiler with C++11 support.

### Simplest way to install pyfasttext: use pip

Just type this line:

```bash
pip install pyfasttext
```

#### Possible compilation error

If you have a compilation error, you can try to install `cysignals` manually:

```bash
pip install cysignals
```

Then, retry to install `pyfasttext` with the already mentioned `pip` command.

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

### Building and installing manually

First, install all the requirements:

```bash
pip install -r requirements.txt
```

Then, build and install with `setup.py`:

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

The `model.get_numpy_vector(word)` method returns the word vector as a `numpy` `ndarray`.

```python
>>> model.get_numpy_vector('dog')
array([-1.30874979, -1.83262241, ...], dtype=float32)
```

If you want a normalized vector (*i.e.* the vector divided by its norm), there is an optional boolean parameter named `normalized`.

```python
>>> model.get_numpy_vector('dog', normalized=True)
array([-0.07084749, -0.09920666, ...], dtype=float32)
```
##### Words for a given vector

The inverse operation of `model[word]` or `model.get_numpy_vector(word)` is `model.words_for_vector(vector, k)`.  
It returns a list of the `k` words closest to the provided vector. The default value for `k` is 1.

```python
>>> king = model.get_numpy_vector('king')
>>> man = model.get_numpy_vector('man')
>>> woman = model.get_numpy_vector('woman')
>>> model.words_for_vector(king + woman - man, k=1)
[('queen', 0.77121970653533936)]
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

The `model.most_similar()` method works similarly as the one in [gensim](https://radimrehurek.com/gensim/models/keyedvectors.html).

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

To obtain the `k` most likely labels from test sentences, there are multiple `model.predict_*()` methods.  
The default value for `k` is 1. If you want to obtain all the possible labels, use `None` for `k`.

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

If you want normalized probabilities (where the sum is closer to 1.0 than the original probabilities), you can use the `normalized=True` parameter in all the methods that output probabilities (`model.predict_proba()`, `model.predict_proba_file()` and `model.predict_proba_single()`).

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

Use keyword arguments in the `model.quantize()` method.

```python
>>> model.quantize(input='/path/to/input.txt', output='/path/to/model')
```

You can load quantized models using the `FastText` constructor or the `model.load_model()` method.

##### Is a model quantized?

If you want to know if a model has been quantized before, use the `model.quantized` attribute.

```python
>>> model = FastText('/path/to/model.bin')
>>> model.quantized
False
>>> model = FastText('/path/to/model.ftz')
>>> model.quantized
True
```

### Subwords

fastText can use subwords (*i.e.* character ngrams) when doing unsupervised or supervised learning.

You can access the subwords, and their associated vectors, using `pyfasttext`.

#### Get the subwords

fastText's word embeddings can be augmented with subword-level information. It is possible to retrieve the subwords and their associated vectors from a model using `pyfasttext`.

To retrieve all the subwords for a given word, use the `model.get_all_subwords(word)` method.

```python
>>> model.args.get('minn'), model.args.get('maxn')
(2, 4)
>>> model.get_all_subwords('hello') # word + subwords from 2 to 4 characters
['hello', '<h', '<he', '<hel', 'he', 'hel', 'hell', 'el', 'ell', 'ello', 'll', 'llo', 'llo>', 'lo', 'lo>', 'o>']
```

For fastText, `<` means "beginning of a word" and `>` means "end of a word".

As you can see, fastText includes the full word. You can omit it using the `omit_word=True` keyword argument.

```python
>>> model.get_all_subwords('hello', omit_word=True)
['<h', '<he', '<hel', 'he', 'hel', 'hell', 'el', 'ell', 'ello', 'll', 'llo', 'llo>', 'lo', 'lo>', 'o>']
```

When a model is quantized, fastText may *prune* some subwords.
If you want to see only the subwords that are really used when computing a word vector, you should use the `model.get_subwords(word)` method.

```python
>>> model.quantized
True
>>> model.get_subwords('beautiful')
['eau', 'aut', 'ful', 'ul']
>>> model.get_subwords('hello')
['hello'] # fastText will not use any subwords when computing the word vector, only the full word
```

#### Get the subword vectors

To get the individual vectors given the subwords, use the `model.get_numpy_subword_vectors(word)` method.

```python
>>> model.get_numpy_subword_vectors('beautiful') # 4 vectors, so 4 rows
array([[ 0.49022141,  0.13586822,  ..., -0.14065443,  0.89617103], # subword "eau"
       [-0.42594951,  0.06260503,  ..., -0.18182631,  0.34219387], # subword "aut"
       [ 0.49958718,  2.93831301,  ..., -1.97498322, -1.16815805], # subword "ful"
       [-0.4368791 , -1.92924356,  ...,  1.62921488, 1.90240896]], dtype=float32) # subword "ul"
```

In fastText, the final word vector is the average of these individual vectors.

```python
>>> import numpy as np
>>> vec1 = model.get_numpy_vector('beautiful')
>>> vecs2 = model.get_numpy_subword_vectors('beautiful')
>>> np.allclose(vec1, np.average(vecs2, axis=0))
True
```

### Sentence and text vectors

To compute the vector of a sequence of words (*i.e.* a sentence), fastText uses two different methods:
* one for unsupervised models
* another one for supervised models

When fastText computes a word vector, recall that it uses the average of the following vectors: the word itself and its subwords.

#### Unsupervised models

For unsupervised models, the representation of a sentence for fastText is the average of the normalized word vectors.

To get the resulting vector as a regular Python array, use the `model.get_sentence_vector(line)` method.  
To get the resulting vector as a `numpy` `ndarray`, use the `model.get_numpy_sentence_vector(line)` method.

```python
>>> vec = model.get_numpy_sentence_vector('beautiful cats')
>>> vec1 = model.get_numpy_vector('beautiful', normalized=True)
>>> vec2 = model.get_numpy_vector('cats', normalized=True)
>>> np.allclose(vec, np.average([vec1, vec2], axis=0)
True
```

#### Supervised models

For supervised models, fastText uses the regular word vectors, as well as vectors computed using word ngrams (*i.e.* shorter sequences of words from the sentence). When computing the average, these vectors are not normalized.

To get the resulting vector as a regular Python array, use the `model.get_text_vector(line)` method.  
To get the resulting vector as a `numpy` `ndarray`, use the `model.get_numpy_text_vector(line)` method.

```python
>>> model.get_numpy_sentence_vector('beautiful cats') # for an unsupervised model
array([-0.20266785,  0.3407566 ,  ...,  0.03044436,  0.39055538], dtype=float32)
>>> model.get_numpy_text_vector('beautiful cats') # for a supervised model
array([-0.20840774,  0.4289546 ,  ..., -0.00457615,  0.52417743], dtype=float32)
```

### Misc utilities

#### Show the module version

```python
>>> import pyfasttext
>>> pyfasttext.__version__
'0.4.3'
```

#### Show fastText version

As there is no version number in fastText, we use the latest fastText commit hash (from `HEAD`) as a substitute.

```python
>>> import pyfasttext
>>> pyfasttext.__fasttext_version__
'431c9e2a9b5149369cc60fb9f5beba58dcf8ca17'
```

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

#### Show the model version number

fastText uses a versioning scheme for its generated models. You can retrieve the model version number using the `model.version` attribute.

| version number | description |
| :---: | :--- |
| -1  | for really old models with no version number |
| 11  | first version number added by fastText |
| 12  | for models generated after fastText added support for subwords in supervised learning |

```python
>>> model.version
12
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

The `fastText` source code directly calls exit() when something wrong happens (*e.g.* a model file does not exist, ...).

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

### Interruptible operations

`pyfasttext` uses `cysignals` to make all the computationally intensive operations (*e.g.* training) interruptible.

To easily interrupt such an operation, just type `Ctrl-C` in your Python shell.

```python
>>> model.skipgram(input='/path/to/input.txt', output='/path/to/mymodel')
Read 12M words
Number of words:  60237
Number of labels: 0
... # type Ctrl-C during training
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "src/pyfasttext.pyx", line 680, in pyfasttext.FastText.skipgram (src/pyfasttext.cpp:11125)
  File "src/pyfasttext.pyx", line 674, in pyfasttext.FastText.train (src/pyfasttext.cpp:11009)
  File "src/pyfasttext.pyx", line 668, in pyfasttext.FastText.train (src/pyfasttext.cpp:10926)
  File "src/cysignals/signals.pyx", line 94, in cysignals.signals.sig_raise_exception (build/src/cysignals/signals.c:1328)
KeyboardInterrupt
>>> # you can have your shell back!
```
