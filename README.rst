pyfasttext
==========

Yet another Python binding for
`fastText <https://github.com/facebookresearch/fastText>`__.

The binding only supports Python 3 and requires Cython.

Installation
------------

.. code:: bash

    git clone --recursive https://github.com/vrasneur/pyfasttext.git
    python setup.py install

Usage
-----

How to load the library?
~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> from pyfasttext import FastText

How to load an existing model?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> model = FastText('/path/to/model.bin')

or

.. code:: python

    >>> model = FastText()
    >>> model.load_model('/path/to/model.bin')

Word representation learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| You can use all the options provided by the fastText binary
  (``input``, ``output``, ``epoch``, ``lr``, ...).
| Just use keyword arguments in the training methods of the ``FastText``
  object.

Training using Skipgram
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> model = FastText()
    >>> model.skipgram(input='data.txt', output='model', epoch=100, lr=0.7)

Training using CBoW
^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> model = FastText()
    >>> model.cbow(input='data.txt', output='model', epoch=100, lr=0.7)

Vector for a given word
^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> model['dog']
    array('f', [-0.4947430193424225, 8.133808296406642e-05, ...])

Get all the word vectors in a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> for word in model.words:
    ...   print(word, model[word])

Word similarity
^^^^^^^^^^^^^^^

.. code:: python

    >>> model.similarity('dog', 'cat')
    0.75596606254577637

Most similar words
~~~~~~~~~~~~~~~~~~

.. code:: python

    >>> model.nn('dog', k=2)
    [('dogs', 0.7843924736976624), ('cat', 75596606254577637)]

Analogies
^^^^^^^^^

The ``most_similar()`` method works similarly as the one in
`gensim <https://radimrehurek.com/gensim/models/keyedvectors.html>`__.

.. code:: python

    >>> model.most_similar(positive=['woman', 'king'], negative=['man'], k=1)
    [('queen', 0.77121970653533936)]

Text classification
~~~~~~~~~~~~~~~~~~~

Supervised learning
^^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> model = FastText()
    >>> model.supervised(input='data.txt', output='model', epoch=100, lr=0.7)

Get all the labels
^^^^^^^^^^^^^^^^^^

.. code:: python

    >>> model.labels
    ['LABEL1', 'LABEL2', ...]

Prediction
^^^^^^^^^^

To obtain the *k* most likely label from test sentences, there are
multiple *predict*\ \*()\_ methods.

If you have a list of strings (or an iterable object), use this:

.. code:: python

    >>> model.predict_proba(['first sentence', 'second sentence'], k=2)
    [[('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)], [('LABEL2', 1.0), ('LABEL3', 1.953126549381068e-08)]]

If your test data is stored inside a file, use this:

.. code:: python

    >>> model.predict_file('/path/to/test.txt', k=2)
    [[('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)], [('LABEL2', 1.0), ('LABEL3', 1.953126549381068e-08)]]

If you want to test a single string, use this:

.. code:: python

    >>> model.predict_line('first sentence', k=2)
    [('LABEL1', 0.99609375), ('LABEL3', 1.953126549381068e-08)]

Misc utilities
^^^^^^^^^^^^^^

Show the model (hyper)parameters
''''''''''''''''''''''''''''''''

.. code:: python

    >>> model.args
    {'bucket': 11000000,
     'cutoff': 0,
     'dim': 100,
     'dsub': 2,
     'epoch': 100,
    ...
    }

Extract labels from a dataset
'''''''''''''''''''''''''''''

There can be multiple labels per line.

.. code:: python

    >>> model.extract_labels('/path/to/dataset1.txt')
    [['LABEL2', 'LABEL5'], ['LABEL1'], ...]

Extract classes from a dataset
''''''''''''''''''''''''''''''

There can be only one class per line.

.. code:: python

    >>> model.extract_classes('/path/to/dataset2.txt')
    ['LABEL3', 'LABEL1', 'LABEL2', ...]
