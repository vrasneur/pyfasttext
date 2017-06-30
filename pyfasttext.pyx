from cython.operator cimport dereference as deref

from libc.stdint cimport int32_t
from libc.stdio cimport EOF
from libc.stdlib cimport malloc, free
from libc.math cimport exp
from libc.string cimport strdup

from builtins import bytes

from libcpp cimport bool
from libcpp.memory cimport shared_ptr, make_shared, unique_ptr, make_unique
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.utility cimport pair

cdef extern from "<iostream>" namespace "std":
  cdef cppclass istream:
    int peek()

  cdef cppclass ifstream(istream):
    ifstream() except +
    ifstream(const string&) except +

cdef extern from "<sstream>" namespace "std":
  cdef cppclass istringstream(istream):
    istringstream() except +
    istringstream(const string&) except +
    void str(const string&)

cdef extern from "fastText/src/args.h" namespace "fasttext":
  cdef cppclass Args:
    Args()
    void parseArgs(int, char **) except +

cdef extern from "fastText/src/dictionary.h" namespace "fasttext":
  cdef cppclass Dictionary:
    int32_t nlabels()
    string getLabel(int32_t)

cdef extern from "fastText/src/fasttext.h" namespace "fasttext":
  cdef cppclass CFastText "fasttext::FastText":
    FastText() except +
    void loadModel(const string&) except +
    void train(shared_ptr[Args]) except +
    void test(istream&, int32_t)
    void predict(istream&, int32_t, vector[pair[float, string]]&)

cdef extern from "fasttext_access.h":
  cdef shared_ptr[Dictionary] &get_fasttext_dict(CFastText&)

cdef char **to_cstring_array(list_str, encoding):
  cdef char **ret = <char **>malloc(len(list_str) * sizeof(char *))
  for i in range(len(list_str)):
    temp = strdup(bytes(list_str[i], encoding))
    ret[i] = temp

  return ret

cdef free_cstring_array(char **arr, length):
  for i in range(length):
    free(arr[i])

  free(arr)

cdef class FastText:
  cdef:
    CFastText ft
    str encoding
    str prefix
    bool loaded

  def __cinit__(self, prefix='__label__', encoding='utf8'):
    self.prefix = prefix
    self.encoding = encoding
    self.loaded = False

  cdef check_loaded(self):
    if not self.loaded:
      raise RuntimeError('model not loaded!')

  @property
  def encoding(self):
    return self.encoding

  @property
  def labels(self):
    labels = []

    if not self.loaded:
      return labels

    dict = get_fasttext_dict(self.ft)
    nlabels = deref(dict).nlabels()

    for i in range(nlabels):
      label = deref(dict).getLabel(i).decode(self.encoding)
      if self.prefix is not None:
        label = label.replace(self.prefix, '')
      labels.append(label)

    return labels

  def load_model(self, fname, encoding=None):
    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    self.ft.loadModel(fname)
    self.loaded = True

  def train(self, command, encoding=None, **kwargs):
    if encoding is None:
      encoding = self.encoding

    args = ['fastText', command]
    for key, val in kwargs.items():
      args.append('-' + key)
      args.append(str(val))

    cdef:
      char **c_args = to_cstring_array(args, encoding)
      shared_ptr[Args] s_args = make_shared[Args]()

    deref(s_args).parseArgs(len(args), c_args)
    self.ft.train(s_args)

    free_cstring_array(c_args, len(args))
    self.loaded = True

  def skipgram(self, encoding=None, **kwargs):
    self.train('skipgram', encoding=encoding, **kwargs)

  def cbow(self, encoding=None, **kwargs):
    self.train('cbow', encoding=encoding, **kwargs)

  def supervised(self, encoding=None, **kwargs):
    self.train('supervised', encoding=encoding, **kwargs)

  def test(self, fname, k=1, encoding=None):
    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    cdef:
      unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)
      int32_t c_k = k

    self.ft.test(deref(ifs), c_k)

  cdef convert_c_predictions(self, vector[pair[float, string]] &c_predictions, str encoding):
    preds = []
    for c_pred in c_predictions:
      proba = exp(c_pred.first)
      label = c_pred.second.decode(encoding)
      if self.prefix is not None:
        label = label.replace(self.prefix, '')

      preds.append((proba, label))

    return preds

  def predict_file(self, fname, k=1, encoding=None):
    self.check_loaded()

    if encoding is None:
      encoding = self.encoding

    fname = bytes(fname, encoding)
    cdef:
      unique_ptr[ifstream] ifs = make_unique[ifstream](<string>fname)
      int32_t c_k = k
      vector[pair[float, string]] c_predictions

    predictions = []
    while deref(ifs).peek() != EOF:
      self.ft.predict(deref(ifs), c_k, c_predictions)

      predictions.append(self.convert_c_predictions(c_predictions, encoding))
    return predictions

  def predict_lines(self, lines, k=1, encoding=None):
    self.check_loaded()

    if encoding is None:
      encoding = self.encoding

    cdef:
      unique_ptr[istringstream] iss = make_unique[istringstream]()
      int32_t c_k = k
      vector[pair[float, string]] c_predictions

    predictions = []
    for line in lines:
      line = bytes(line, encoding)
      deref(iss).str(line)
      self.ft.predict(deref(iss), c_k, c_predictions)
      predictions.append(self.convert_c_predictions(c_predictions, encoding))

    return predictions

  def predict_line(self, line, k=1, encoding=None):
    return self.predict_lines([line], k=k, encoding=encoding)[0]
