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

### Word representation learning

#### Training using Skipgram

```python
>>> model = FastText()
>>> model.skipgram(input='data.txt', output='model')
```
#### Training using CBoW

```python
>>> model = FastText()
>>> model.cbow(input='data.txt', output='model')
```

### Supervised learning

```python
>>> model = FastText()
>>> model.cbow(input='data.txt', output='model')
```

### Loading an existing model

```python
>>> model = FastText('/path/to/model.bin')
```

or

```python
>>> model = FastText()
>>> model.load_model('/path/to/model.bin')
```
