from cython.operator cimport dereference as deref

from libc.stdint cimport int32_t, int64_t
from libc.stdio cimport EOF
from libc.stdlib cimport malloc, free
from libc.math cimport exp, log
from libc.string cimport strdup, memcpy

from builtins import bytes

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared, unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.set cimport set
from libcpp.utility cimport pair
from libcpp.queue cimport priority_queue

import array

IF USE_NUMPY:
  import numpy as np
  cimport numpy as np

  np.import_array()

cdef extern from "<iostream>" namespace "std" nogil:
  cdef cppclass istream:
    int peek()

  cdef cppclass ifstream(istream):
    ifstream() except +
    ifstream(const string&) except +

cdef extern from "<sstream>" namespace "std" nogil:
  cdef cppclass istringstream(istream):
    istringstream() except +
    istringstream(const string&) except +
    void str(const string&)

ctypedef float real

cdef extern from "fastText/src/vector.h" namespace "fasttext" nogil:
  cdef cppclass Vector:
    Vector(int64_t)
    real *data_
    void zero()
    int64_t size()
    real& operator[](int64_t)
    void mul(real)
    real norm()
    void addVector(const Vector &, real)

cdef extern from "fastText/src/matrix.h" namespace "fasttext" nogil:
  cdef cppclass Matrix:
    Matrix()
    Matrix(int64_t, int64_t)
    real *data_
    void zero()
    void addRow(const Vector&, int64_t, real)
    real dotRow(const Vector&, int64_t)

cdef extern from "fastText/src/args.h" namespace "fasttext" nogil:
  cdef enum loss_name:
    pass
  cdef enum model_name:
    pass
  cdef cppclass Args:
    Args()
    string label
    void parseArgs(int, char **) except +

cdef extern from "fastText/src/dictionary.h" namespace "fasttext" nogil:
  cdef cppclass Dictionary:
    int32_t nlabels()
    string getLabel(int32_t)
    int32_t nwords()
    string getWord(int32_t)

cdef extern from "fastText/src/fasttext.h" namespace "fasttext" nogil:
  cdef cppclass CFastText "fasttext::FastText":
    FastText() except +
    int getDimension()
    void getVector(Vector&, const string&)
    void loadModel(const string&) except +
    void train(shared_ptr[Args]) except +
    void quantize(shared_ptr[Args]) except +
    void test(istream&, int32_t)
    void predict(istream&, int32_t, vector[pair[real, string]]&)

cdef extern from "compat.h" namespace "pyfasttext" nogil:
  unique_ptr[T] make_unique[T](...)

cdef extern from "utils.h" namespace "pyfasttext" nogil:
  cdef cppclass CStringArrayDeleter:
    CStringArrayDeleter()
    CStringArrayDeleter(char **, size_t)

cdef extern from "fasttext_access.h" namespace "pyfasttext" nogil:
  cdef cppclass ArgValue:
    size_t index()
  cdef shared_ptr[Dictionary] &get_fasttext_dict(CFastText&)
  cdef shared_ptr[Args] &get_fasttext_args(CFastText&)
  cdef map[string, ArgValue] get_args_map(shared_ptr[Args]&)
  string convert_loss_name(loss_name)
  string convert_model_name(model_name)

cdef extern from "variant/v1.2.0/variant.hpp" namespace "mpark" nogil:
  T get[T](ArgValue&)

cdef char **to_cstring_array(list_str, encoding) except NULL:
  cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
  if ret == NULL:
    return ret

  for i in range(len(list_str)):
    temp = strdup(bytes(list_str[i], encoding))
    ret[i] = temp

  return ret

cdef class FastText:
  cdef:
    CFastText ft
    str encoding
    str label
    bool loaded
    unique_ptr[Matrix] word_vectors

  def __cinit__(self, model_fname=None, label='__label__', encoding='utf8'):
    self.label = label
    self.encoding = encoding
    self.loaded = False

    if model_fname is not None:
      self.load_model(model_fname)

  cdef check_loaded(self):
    if not self.loaded:
      raise RuntimeError('model not loaded!')

  def extract_labels(self, fname, encoding=None):
    labels = []

    if self.label is None:
      return labels

    if encoding is None:
      encoding = self.encoding

    with open(fname, 'r', encoding=encoding) as f:
      for line in f:
        labels.append([label.replace(self.label, '') for label in line.split()
	               if label.startswith(self.label)])
      
    return labels

  def extract_classes(self, fname, encoding=None):
    labels = self.extract_labels(fname, encoding=encoding)
    return [label[0] if label else None
            for label in labels]

  @property
  def encoding(self):
    return self.encoding

  # label is the label *prefix* in fastText source code
  @property
  def label(self):
    return self.label

  @property
  def labels(self):
    labels = []

    if not self.loaded:
      return labels

    dict = get_fasttext_dict(self.ft)
    nlabels = deref(dict).nlabels()

    for i in range(nlabels):
      label = deref(dict).getLabel(i).decode(self.encoding)
      if self.label is not None:
        label = label.replace(self.label, '')
      labels.append(label)

    return labels

  @property
  def nlabels(self):
    nlabels = 0

    if not self.loaded:
      return nlabels

    dict = get_fasttext_dict(self.ft)
    nlabels = deref(dict).nlabels()

    return nlabels

  @property
  def args(self):
    ret = {}

    if not self.loaded:
      ret['label'] = self.label
      return ret

    cdef size_t index = 0
    args = get_fasttext_args(self.ft)
    args_map = get_args_map(args)
    for item in args_map:
      key = item.first.decode(self.encoding)
      index = item.second.index()
      if index == 0:
        ret[key] = get[bool](item.second)
      elif index == 1:
        ret[key] = get[int](item.second)
      elif index == 2:
        ret[key] = get[size_t](item.second)
      elif index == 3:
        ret[key] = get[double](item.second)
      elif index == 4:
        ret[key] = get[string](item.second).decode(self.encoding)
      elif index == 5:
        ret[key] = convert_loss_name(get[loss_name](item.second))
      elif index == 6:
        ret[key] = convert_model_name(get[model_name](item.second))

    return ret

  @property
  def words(self):
    words = []

    if not self.loaded:
      return words

    dict = get_fasttext_dict(self.ft)
    nwords = deref(dict).nwords()

    for i in range(nwords):
      word = deref(dict).getWord(i).decode(self.encoding)
      words.append(word)

    return words

  @property
  def nwords(self):
    nwords = 0

    if not self.loaded:
      return nwords

    dict = get_fasttext_dict(self.ft)
    nwords = deref(dict).nwords()

    return nwords

  def __getitem__(self, key):
    if not self.loaded:
      return None

    cdef:
      int dim = self.ft.getDimension()
      unique_ptr[Vector] vec = make_unique[Vector](dim)

    key = bytes(key, self.encoding)
    arr = array.array('f')
    deref(vec).zero()
    
    self.ft.getVector(deref(vec), key)
    for i in range(deref(vec).size()):
      arr.append(deref(vec)[i])

    return arr

  IF USE_NUMPY:
    def get_numpy_vector(self, key, normalized=False):
      if not self.loaded:
        return None

      cdef:
        int dim = self.ft.getDimension()
        unique_ptr[Vector] vec = make_unique[Vector](dim)
        np.npy_intp shape[1]

      key = bytes(key, self.encoding)
      deref(vec).zero()
      
      self.ft.getVector(deref(vec), key)
      if normalized:
        norm = deref(vec).norm()
        if norm > 0:
          deref(vec).mul(1.0 / norm)
      shape[0] = <np.npy_intp>(deref(vec).size())
      arr = np.PyArray_SimpleNew(1, shape, np.NPY_FLOAT32)
      memcpy(np.PyArray_DATA(arr), <void *>(deref(vec).data_), deref(vec).size() * sizeof(real))

      return arr

  cdef precompute_word_vectors(self):
    if self.word_vectors:
      return

    cdef:
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      string word

    dict = get_fasttext_dict(self.ft)
    self.word_vectors = make_unique[Matrix](deref(dict).nwords(), self.ft.getDimension())
    deref(self.word_vectors).zero()
    for i in range(deref(dict).nwords()):
      word = deref(dict).getWord(i)
      self.ft.getVector(deref(vec), word)
      norm = deref(vec).norm()
      if norm > 0:
        deref(self.word_vectors).addRow(deref(vec), i, 1.0 / norm)

  def uncache_word_vectors(self):
    self.word_vectors.reset()

  IF USE_NUMPY:
    @property
    def numpy_normalized_vectors(self):
      if not self.loaded:
        return None

      self.precompute_word_vectors()

      cdef:
        int dim = self.ft.getDimension()
        np.npy_intp shape[1]

      dict = get_fasttext_dict(self.ft)
      shape[0] = <np.npy_intp>(deref(dict).nwords() * dim)
      arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32, <void *>deref(self.word_vectors).data_)
      return arr.reshape(deref(dict).nwords(), dim).copy()

  cdef find_nearest_neighbors(self, const Vector &query_vec, int32_t k,
                              const set[string] &ban_set, encoding):
    self.precompute_word_vectors()

    cdef:
      priority_queue[pair[real, string]] heap
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      string word
      int32_t i = 0

    ret = []

    query_norm = query_vec.norm()
    if abs(query_norm) < 1e-8:
      query_norm = 1.0
    dict = get_fasttext_dict(self.ft)
    for idx in range(deref(dict).nwords()):
      word = deref(dict).getWord(idx)
      dp = deref(self.word_vectors).dotRow(query_vec, idx)
      heap.push(pair[real, string](dp / query_norm, word))

    while i < k and heap.size() > 0:
      it = ban_set.find(heap.top().second)
      if it == ban_set.end():
        ret.append((heap.top().second.decode(encoding), heap.top().first))
        i += 1

      heap.pop()

    return ret

  def nearest_neighbors(self, word, k=10, encoding=None):
    self.check_loaded()

    if encoding is None:
      encoding = self.encoding

    word = bytes(word, encoding)
    cdef:
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      set[string] ban_set
    ban_set.insert(word)
    self.ft.getVector(deref(vec), word)
    return self.find_nearest_neighbors(deref(vec), k, ban_set, encoding)

  def similarity(self, word1, word2, encoding=None):
    self.check_loaded()

    if encoding is None:
      encoding = self.encoding

    word1 = bytes(word1, encoding)
    word2 = bytes(word2, encoding)
    cdef:
      unique_ptr[Vector] vec1 = make_unique[Vector](self.ft.getDimension())
      unique_ptr[Vector] vec2 = make_unique[Vector](self.ft.getDimension())
      real dp = 0.0

    self.ft.getVector(deref(vec1), word1)
    self.ft.getVector(deref(vec2), word2)
    for i in range(self.ft.getDimension()):
      dp += deref(vec1)[i] * deref(vec2)[i]

    norm1 = deref(vec1).norm()
    norm2 = deref(vec2).norm()

    if norm1 > 0 and norm2 > 0:
      return dp / (norm1 * norm2)
    return 0.0

  def most_similar(self, positive=[], negative=[], k=10, encoding=None):
    self.check_loaded()

    if encoding is None:
      encoding = self.encoding

    self.precompute_word_vectors()

    cdef:
      unique_ptr[Vector] buffer = make_unique[Vector](self.ft.getDimension())
      unique_ptr[Vector] query = make_unique[Vector](self.ft.getDimension())
      set[string] ban_set

    deref(query).zero()
    for word in positive:
      word = bytes(word, encoding)
      ban_set.insert(word)
      self.ft.getVector(deref(buffer), word)
      deref(query).addVector(deref(buffer), 1.0)
    for word in negative:
      word = bytes(word, encoding)
      ban_set.insert(word)
      self.ft.getVector(deref(buffer), word)
      deref(query).addVector(deref(buffer), -1.0)
    return self.find_nearest_neighbors(deref(query), k, ban_set, encoding)

  cdef update_label(self, encoding):
    args = get_fasttext_args(self.ft)
    if not deref(args).label.empty():
        self.label = str(deref(args).label.decode(encoding))

  def load_model(self, fname, encoding=None):
    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    self.ft.loadModel(fname)
    self.update_label(encoding)
    self.loaded = True

  def train(self, command, encoding=None, **kwargs):
    if encoding is None:
      encoding = self.encoding

    args = ['fastText', command]
    for key, val in kwargs.items():
      args.append('-' + key)
      args.append(str(val))

    cdef:
      shared_ptr[Args] s_args = make_shared[Args]()
      char **c_args = to_cstring_array(args, encoding)
      unique_ptr[CStringArrayDeleter] deleter = make_unique[CStringArrayDeleter](c_args, <size_t>len(args))

    deref(s_args).parseArgs(len(args), c_args)
    if command == 'quantize':
      self.ft.quantize(s_args)
    else:
      self.ft.train(s_args)

    self.update_label(encoding)
    self.loaded = True

  def skipgram(self, encoding=None, **kwargs):
    self.train('skipgram', encoding=encoding, **kwargs)

  def cbow(self, encoding=None, **kwargs):
    self.train('cbow', encoding=encoding, **kwargs)

  def supervised(self, encoding=None, **kwargs):
    self.train('supervised', encoding=encoding, **kwargs)

  def quantize(self, encoding=None, **kwargs):
    self.train('quantize', encoding=encoding, **kwargs)

  def test(self, fname, k=1, encoding=None):
    if k is None:
      k = self.nlabels

    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    cdef unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)

    self.ft.test(deref(ifs), k)

  cdef convert_c_predictions_proba(self, vector[pair[real, string]] &c_predictions,
                                   bool normalized, str encoding):
    log_sum = 0.0
    if normalized and not c_predictions.empty():
      # fasttext probabilities are never zero, and are in descending order
      first = c_predictions[0].first
      for c_pred in c_predictions:
        log_sum += exp(c_pred.first - first)
      log_sum = first + log(log_sum)

    preds = []
    for c_pred in c_predictions:
      proba = c_pred.first
      if normalized:
         proba -= log_sum
      proba = exp(proba)
      label = c_pred.second.decode(encoding)
      if self.label is not None:
        label = label.replace(self.label, '')

      preds.append((label, proba))

    return preds

  cdef convert_c_predictions(self, vector[pair[real, string]] &c_predictions,
                             str encoding):
    preds = []
    for c_pred in c_predictions:
      label = c_pred.second.decode(encoding)
      if self.label is not None:
        label = label.replace(self.label, '')

      preds.append(label)

    return preds

  cdef predict_aux(self, lines, k, bool proba, bool normalized, str encoding):
    self.check_loaded()

    if k is None:
      k = self.nlabels

    if encoding is None:
      encoding = self.encoding

    cdef:
      unique_ptr[istringstream] iss = make_unique[istringstream]()
      vector[pair[real, string]] c_predictions

    predictions = []
    for line in lines:
      line = bytes(line, encoding)
      deref(iss).str(line)
      self.ft.predict(deref(iss), k, c_predictions)

      if proba:
        predictions.append(self.convert_c_predictions_proba(c_predictions, normalized, encoding))
      else:
        predictions.append(self.convert_c_predictions(c_predictions, encoding))

    return predictions

  def predict_proba(self, lines, k=1, normalized=False, encoding=None):
    return self.predict_aux(lines, k=k, proba=True, normalized=normalized, encoding=encoding)

  def predict(self, lines, k=1, encoding=None):
    return self.predict_aux(lines, k=k, proba=False, normalized=False, encoding=encoding)

  def predict_proba_single(self, line, k=1, normalized=False, encoding=None):
    return self.predict_proba([line], k=k, normalized=normalized, encoding=encoding)[0]

  def predict_single(self, line, k=1, encoding=None):
    return self.predict([line], k=k, encoding=encoding)[0]

  cdef predict_aux_file(self, fname, k, bool proba, bool normalized, str encoding):
    self.check_loaded()

    if k is None:
      k = self.nlabels

    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    cdef:
      unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)
      vector[pair[real, string]] c_predictions

    predictions = []
    while deref(ifs).peek() != EOF:
      self.ft.predict(deref(ifs), k, c_predictions)

      if proba:
        predictions.append(self.convert_c_predictions_proba(c_predictions, normalized, encoding))
      else:
        predictions.append(self.convert_c_predictions(c_predictions, encoding))
    return predictions

  def predict_proba_file(self, fname, k=1, normalized=False, encoding=None):
    return self.predict_aux_file(fname, k=k, proba=True, normalized=normalized, encoding=encoding)

  def predict_file(self, fname, k=1, encoding=None):
    return self.predict_aux_file(fname, k=k, proba=False, normalized=False, encoding=encoding)
