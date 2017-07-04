# pyfasttext
Yet another Python binding for [fastText](https://github.com/facebookresearch/fastText)

The binding only supports Python 3 and requires Cython.

## Installation

```bash
git clone --recursive https://github.com/vrasneur/pyfasttext.git
python setup.py install
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

You can use all the options provided by the fastText binary (`input`, `output`, `epoch`, `lr`, ...).  
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

#### Vector for a given word

```python
>>> model['dog']
array('f', [-0.4947430193424225, 8.133808296406642e-05, ...])
```

### Get all the word vectors in a model

```python
>>> for word in model.words:
...   print(word, model[word])
```

#### Word similarity

```python
>>> model.similarity('dog', 'cat')
0.75596606254577637
```

### Most similar words

```python
>>> model.nn('dog', 2)
[('dogs', 0.7843924736976624), ('cat', 75596606254577637)]
```

#### Analogies

```python
>>> model.most_similar(positive=['woman', 'king'], negative=['man'], 1)
0.77121970653533936
```

### Text classification

#### Supervised learning

```python
>>> model = FastText()
>>> model.supervised(input='data.txt', output='model', epoch=100, lr=0.7)
```

#### Get all the labels

```python
>>> model.labels
['LABEL1', 'LABEL2', ...]
```

