# pyfasttext
# Yet another Python binding for fastText
# 
# Copyright (c) 2017 Vincent Rasneur <vrasneur@free.fr>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; version 3 only.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

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

IF USE_CYSIGNALS:
  from cysignals.signals cimport sig_on, sig_off, sig_check
ELSE:
  sig_on = sig_off = sig_check = lambda: None

import array

IF USE_NUMPY:
  import numpy as np
  cimport numpy as np

  np.import_array()

cdef extern from "<random>" namespace "std" nogil:
  cdef cppclass minstd_rand:
    pass

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
    int64_t size() const
    real& operator[](int64_t) const
    void mul(real)
    real norm() const
    void addVector(const Vector &, real)

cdef extern from "fastText/src/matrix.h" namespace "fasttext" nogil:
  cdef cppclass Matrix:
    Matrix()
    Matrix(int64_t, int64_t)
    int64_t m_
    int64_t n_
    real *data_
    void zero()
    void addRow(const Vector&, int64_t, real)
    real dotRow(const Vector&, int64_t) const

cdef extern from "fastText/src/args.h" namespace "fasttext" nogil:
  cdef enum loss_name:
    pass
  cdef enum model_name:
    pass
  cdef cppclass Args:
    Args()
    string label
    void parseArgs(vector[string]&) except +

cdef extern from "fastText/src/model.h" namespace "fasttext" nogil:
  cdef cppclass Model:
    minstd_rand rng

cdef extern from "fastText/src/dictionary.h" namespace "fasttext" nogil:
  cdef cppclass Dictionary:
    int32_t nlabels() const
    string getLabel(int32_t) const
    int32_t nwords() const
    int32_t getId(const string &) const
    string getWord(int32_t) const
    vector[int32_t] getSubwords(const string &) const
    void getSubwords(const string &, vector[int32_t] &, vector[string] &) const
    int32_t getLine(istream&, vector[int32_t]&, vector[int32_t]&, minstd_rand&) const

cdef extern from "fastText/src/fasttext.h" namespace "fasttext" nogil:
  cdef cppclass CFastText "fasttext::FastText":
    FastText() except +
    shared_ptr[const Dictionary] getDictionary() const
    int getDimension() const
    void getVector(Vector&, const string&) const
    void loadModel(const string&) except +
    void train(shared_ptr[Args]) except +
    void quantize(shared_ptr[Args]) except +
    void test(istream&, int32_t)
    void predict(istream&, int32_t, vector[pair[real, string]]&) const

cdef extern from "compat.h" namespace "pyfasttext" nogil:
  unique_ptr[T] make_unique[T](...)

cdef extern from "fasttext_access.h" namespace "pyfasttext" nogil:
  cdef cppclass ArgValue:
    size_t which()
  bool check_model(CFastText&, const string&) except +
  void load_older_model(CFastText&, const string&) except +
  shared_ptr[const Args] get_fasttext_args(const CFastText&)
  shared_ptr[Model] get_fasttext_model(CFastText&)
  void set_fasttext_max_tokenCount(CFastText&)
  bool add_input_vector(const CFastText&, Vector &, int32_t)
  int32_t get_model_version(const CFastText&)
  bool is_model_quantized(const CFastText&)
  bool is_dict_pruned(const CFastText&)
  bool is_word_pruned(const CFastText&, int32_t)
  map[string, ArgValue] get_args_map(const shared_ptr[const Args]&)
  string convert_loss_name(const loss_name)
  string convert_model_name(const model_name)

cdef extern from "variant/include/mapbox/variant.hpp" namespace "mapbox::util" nogil:
  T get[T](const ArgValue&)

__version__ = VERSION
__fasttext_version__ = FASTTEXT_VERSION

cdef vec_to_array(const Vector &vec):
  arr = array.array('f')
  for i in range(vec.size()):
    arr.append(vec[i])

  return arr

IF USE_NUMPY:
  cdef vec_to_numpy_array(const Vector &vec):
    cdef np.npy_intp shape[1]

    shape[0] = <np.npy_intp>(vec.size())
    arr = np.PyArray_SimpleNew(1, shape, np.NPY_FLOAT32)
    memcpy(np.PyArray_DATA(arr), <void *>(vec.data_), vec.size() * sizeof(real))

    return arr

  cdef mat_to_numpy_array(const Matrix &mat):
    cdef np.npy_intp shape[1]

    shape[0] = <np.npy_intp>(mat.m_ * mat.n_)
    arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_FLOAT32, <void *>mat.data_)
    return arr.reshape(mat.m_, mat.n_).copy()

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

  def extract_labels(self, fname):
    labels = []

    if self.label is None:
      return labels

    with open(fname, 'r', encoding=self.encoding) as f:
      for line in f:
        labels.append([label.replace(self.label, '') for label in line.split()
	               if label.startswith(self.label)])
      
    return labels

  def extract_classes(self, fname):
    labels = self.extract_labels(fname)
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

    dict = self.ft.getDictionary()
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

    dict = self.ft.getDictionary()
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
      index = item.second.which()
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
        ret[key] = convert_loss_name(get[loss_name](item.second)).decode(self.encoding)
      elif index == 6:
        ret[key] = convert_model_name(get[model_name](item.second)).decode(self.encoding)

    return ret

  @property
  def version(self):
    if not self.loaded:
      return -1

    return get_model_version(self.ft)

  @property
  def quantized(self):
    if not self.loaded:
      return False

    return is_model_quantized(self.ft)

  @property
  def pruned(self):
    if not self.loaded:
      return False

    return is_dict_pruned(self.ft)

  @property
  def words(self):
    words = []

    if not self.loaded:
      return words

    dict = self.ft.getDictionary()
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

    dict = self.ft.getDictionary()
    nwords = deref(dict).nwords()

    return nwords

  def __getitem__(self, key):
    if not self.loaded:
      return None

    cdef:
      int dim = self.ft.getDimension()
      unique_ptr[Vector] vec = make_unique[Vector](dim)

    key = bytes(key, self.encoding)
    deref(vec).zero()
    self.ft.getVector(deref(vec), key)

    return vec_to_array(deref(vec))

  IF USE_NUMPY:
    def get_numpy_vector(self, key, normalized=False):
      if not self.loaded:
        return None

      cdef:
        int dim = self.ft.getDimension()
        unique_ptr[Vector] vec = make_unique[Vector](dim)

      key = bytes(key, self.encoding)
      deref(vec).zero()
      
      self.ft.getVector(deref(vec), key)
      if normalized:
        norm = deref(vec).norm()
        if norm > 0:
          deref(vec).mul(1.0 / norm)

      return vec_to_numpy_array(deref(vec))

  def get_subwords(self, word, omit_word=False, omit_pruned=True):
    if not self.loaded:
      return []

    cdef:
      vector[int32_t] ngrams
      vector[string] substrings
      vector[string] temp

    word = bytes(word, self.encoding)
    dict = self.ft.getDictionary()
    deref(dict).getSubwords(word, ngrams, substrings)

    if omit_pruned:
      if not ngrams.empty():
        if not omit_word:
          id = deref(dict).getId(word)
          if id != -1:
            temp.push_back(substrings[0])

        for idx in xrange(ngrams.size() - 1):
          if not is_word_pruned(self.ft, ngrams[idx + 1]):
            temp.push_back(substrings[idx])
      substrings = temp
    elif omit_word and not substrings.empty():
      substrings.erase(substrings.begin())

    return [substr.decode(self.encoding) for substr in substrings]

  def get_all_subwords(self, word, omit_word=False):
    return self.get_subwords(word, omit_word=omit_word, omit_pruned=False)

  IF USE_NUMPY:
    def get_numpy_subword_vectors(self, word):
      if not self.loaded:
        return None

      cdef:
        int dim = self.ft.getDimension()
        unique_ptr[Vector] vec = make_unique[Vector](dim)
        vector[int32_t] ngrams
        np.npy_intp shape[2]
        char *ptr

      word = bytes(word, self.encoding)
      dict = self.ft.getDictionary()
      ngrams = deref(dict).getSubwords(word);
      if ngrams.empty():
        return None

      shape[0] = <np.npy_intp>ngrams.size()
      shape[1] = <np.npy_intp>dim
      arr = np.PyArray_SimpleNew(2, shape, np.NPY_FLOAT32)
      for idx in xrange(ngrams.size()):
        deref(vec).zero()
        filled = add_input_vector(self.ft, deref(vec), ngrams[idx])
        if not filled:
          return None

        ptr = <char*>(np.PyArray_DATA(arr))
        ptr += idx * np.PyArray_STRIDE(arr, 0)
        memcpy(ptr, <void *>(deref(vec).data_), deref(vec).size() * sizeof(real))

      return arr

  cdef get_sentence_vector_aux(self, line, Vector &svec, sep=None):
    if not self.loaded:
      return None

    cdef:
      int dim = self.ft.getDimension()
      unique_ptr[Vector] vec = make_unique[Vector](dim)

    svec.zero()
    count = 0
    for word in line.split(sep):
      word = bytes(word, self.encoding)
      deref(vec).zero()
      self.ft.getVector(deref(vec), word)
      norm = deref(vec).norm()
      if norm > 0:
        deref(vec).mul(1.0 / norm)
        svec.addVector(deref(vec), 1.0)
        count += 1

    if count > 0:
      svec.mul(1.0 / count)

  def get_sentence_vector(self, line, sep=None):
    cdef:
      int dim = self.ft.getDimension()
      unique_ptr[Vector] svec = make_unique[Vector](dim)

    self.get_sentence_vector_aux(line, deref(svec), sep=sep)

    return vec_to_array(deref(svec))

  IF USE_NUMPY:
    def get_numpy_sentence_vector(self, line, sep=None):
      cdef:
        int dim = self.ft.getDimension()
        unique_ptr[Vector] svec = make_unique[Vector](dim)

      self.get_sentence_vector_aux(line, deref(svec), sep=sep)

      return vec_to_numpy_array(deref(svec))

  cdef get_text_vector_aux(self, line, Vector &vec):
    if not self.loaded:
      return None

    cdef:
      unique_ptr[istringstream] iss = make_unique[istringstream]()
      vector[int32_t] words
      vector[int32_t] labels

    vec.zero()
    dict = self.ft.getDictionary()
    line = bytes(line, self.encoding)
    deref(iss).str(line)
    model = get_fasttext_model(self.ft)
    # handle word + char ngrams (subwords) + word ngrams
    deref(dict).getLine(deref(iss), words, labels, deref(model).rng)
    for word in words:
      filled = add_input_vector(self.ft, vec, word)
      if not filled:
        return None

    if not words.empty():
       vec.mul(1.0 / words.size())

  def get_text_vector(self, line):
    cdef:
      int dim = self.ft.getDimension()
      unique_ptr[Vector] vec = make_unique[Vector](dim)

    self.get_text_vector_aux(line, deref(vec))

    return vec_to_array(deref(vec))

  IF USE_NUMPY:
    def get_numpy_text_vector(self, line):
      cdef:
        int dim = self.ft.getDimension()
        unique_ptr[Vector] vec = make_unique[Vector](dim)

      self.get_text_vector_aux(line, deref(vec))

      return vec_to_numpy_array(deref(vec))

  cdef precompute_word_vectors(self):
    if self.word_vectors:
      return

    cdef:
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      string word

    dict = self.ft.getDictionary()
    self.word_vectors = make_unique[Matrix](deref(dict).nwords(), self.ft.getDimension())
    deref(self.word_vectors).zero()
    for i in range(deref(dict).nwords()):
      sig_check()
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

      return mat_to_numpy_array(deref(self.word_vectors))

  cdef find_nearest_neighbors(self, const Vector &query_vec, int32_t k,
                              const set[string] &ban_set):
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
    dict = self.ft.getDictionary()
    for idx in range(deref(dict).nwords()):
      sig_check()
      word = deref(dict).getWord(idx)
      dp = deref(self.word_vectors).dotRow(query_vec, idx)
      heap.push(pair[real, string](dp / query_norm, word))

    while i < k and heap.size() > 0:
      sig_check()
      it = ban_set.find(heap.top().second)
      if it == ban_set.end():
        ret.append((heap.top().second.decode(self.encoding), heap.top().first))
        i += 1

      heap.pop()

    return ret

  def words_for_vector(self, v, k=1):
    self.check_loaded()

    cdef:
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      set[string] ban_set

    if len(v) != deref(vec).size():
      raise ValueError('Input vector length ({}) is not equal to the model dimension ({})!'.format(len(v), deref(vec).size()))

    deref(vec).zero()
    for i in range(deref(vec).size()):
      deref(vec)[i] = v[i]
    return self.find_nearest_neighbors(deref(vec), k, ban_set)

  def nearest_neighbors(self, word, k=10):
    self.check_loaded()

    word = bytes(word, self.encoding)
    cdef:
      unique_ptr[Vector] vec = make_unique[Vector](self.ft.getDimension())
      set[string] ban_set
    ban_set.insert(word)
    self.ft.getVector(deref(vec), word)
    return self.find_nearest_neighbors(deref(vec), k, ban_set)

  def similarity(self, word1, word2):
    self.check_loaded()

    word1 = bytes(word1, self.encoding)
    word2 = bytes(word2, self.encoding)
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

  def most_similar(self, positive=[], negative=[], k=10):
    self.check_loaded()

    self.precompute_word_vectors()

    cdef:
      unique_ptr[Vector] buffer = make_unique[Vector](self.ft.getDimension())
      unique_ptr[Vector] query = make_unique[Vector](self.ft.getDimension())
      set[string] ban_set

    deref(query).zero()
    for word in positive:
      word = bytes(word, self.encoding)
      ban_set.insert(word)
      self.ft.getVector(deref(buffer), word)
      deref(query).addVector(deref(buffer), 1.0)
    for word in negative:
      word = bytes(word, self.encoding)
      ban_set.insert(word)
      self.ft.getVector(deref(buffer), word)
      deref(query).addVector(deref(buffer), -1.0)
    return self.find_nearest_neighbors(deref(query), k, ban_set)

  cdef update_label(self):
    args = get_fasttext_args(self.ft)
    if not deref(args).label.empty():
        self.label = str(deref(args).label.decode(self.encoding))

  def load_model(self, fname):
    fname = bytes(fname, self.encoding)
    if check_model(self.ft, fname):
      self.ft.loadModel(fname)
    else:
      load_older_model(self.ft, fname)
    self.update_label()
    self.loaded = True

  def train(self, command, **kwargs):
    cdef vector[string] args
    args.push_back(bytes('fastText', self.encoding))
    args.push_back(bytes(command, self.encoding))

    for key, val in kwargs.items():
      args.push_back(bytes('-' + key, self.encoding))
      args.push_back(bytes(str(val), self.encoding))

    cdef shared_ptr[Args] s_args = make_shared[Args]()

    deref(s_args).parseArgs(args)
    try:
      if command == 'quantize':
        sig_on()
        self.ft.quantize(s_args)
        sig_off()
      else:
        sig_on()
        self.ft.train(s_args)
        sig_off()
    except:
      # make the other threads finish
      set_fasttext_max_tokenCount(self.ft)
      raise

    self.update_label()
    self.loaded = True

  def skipgram(self, **kwargs):
    self.train('skipgram', **kwargs)

  def cbow(self, **kwargs):
    self.train('cbow', **kwargs)

  def supervised(self, **kwargs):
    self.train('supervised', **kwargs)

  def quantize(self, **kwargs):
    self.train('quantize', **kwargs)

  def test(self, fname, k=1):
    if k is None:
      k = self.nlabels

    fname = bytes(fname, self.encoding)
    cdef unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)

    self.ft.test(deref(ifs), k)

  cdef convert_c_predictions_proba(self, vector[pair[real, string]] &c_predictions,
                                   bool normalized):
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
      label = c_pred.second.decode(self.encoding)
      if self.label is not None:
        label = label.replace(self.label, '')

      preds.append((label, proba))

    return preds

  cdef convert_c_predictions(self, vector[pair[real, string]] &c_predictions):
    preds = []
    for c_pred in c_predictions:
      label = c_pred.second.decode(self.encoding)
      if self.label is not None:
        label = label.replace(self.label, '')

      preds.append(label)

    return preds

  cdef predict_aux(self, lines, k, bool proba, bool normalized):
    self.check_loaded()

    if k is None:
      k = self.nlabels

    cdef:
      unique_ptr[istringstream] iss = make_unique[istringstream]()
      vector[pair[real, string]] c_predictions

    predictions = []
    for line in lines:
      line = bytes(line, self.encoding)
      deref(iss).str(line)
      sig_on()
      self.ft.predict(deref(iss), k, c_predictions)
      sig_off()

      if proba:
        predictions.append(self.convert_c_predictions_proba(c_predictions, normalized))
      else:
        predictions.append(self.convert_c_predictions(c_predictions))

    return predictions

  def predict_proba(self, lines, k=1, normalized=False):
    return self.predict_aux(lines, k=k, proba=True, normalized=normalized)

  def predict(self, lines, k=1):
    return self.predict_aux(lines, k=k, proba=False, normalized=False)

  def predict_proba_single(self, line, k=1, normalized=False):
    return self.predict_proba([line], k=k, normalized=normalized)[0]

  def predict_single(self, line, k=1):
    return self.predict([line], k=k)[0]

  cdef predict_aux_file(self, fname, k, bool proba, bool normalized):
    self.check_loaded()

    if k is None:
      k = self.nlabels

    fname = bytes(fname, self.encoding)
    cdef:
      unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)
      vector[pair[real, string]] c_predictions

    predictions = []
    while deref(ifs).peek() != EOF:
      self.ft.predict(deref(ifs), k, c_predictions)

      if proba:
        predictions.append(self.convert_c_predictions_proba(c_predictions, normalized))
      else:
        predictions.append(self.convert_c_predictions(c_predictions))
    return predictions

  def predict_proba_file(self, fname, k=1, normalized=False):
    return self.predict_aux_file(fname, k=k, proba=True, normalized=normalized)

  def predict_file(self, fname, k=1):
    return self.predict_aux_file(fname, k=k, proba=False, normalized=False)
