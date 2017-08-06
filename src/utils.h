#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>

namespace pyfasttext
{

struct CStringArrayDeleter
{
  char **arr_ = nullptr;
  size_t len_ = 0;
  
  CStringArrayDeleter(char **arr, size_t len)
  : arr_(arr), len_(len)
  {}

  CStringArrayDeleter(const CStringArrayDeleter &other) = delete;
  CStringArrayDeleter &operator=(const CStringArrayDeleter &other) = delete;
  CStringArrayDeleter() = default;

  ~CStringArrayDeleter()
  {
    if(arr_ != nullptr) {
      for(size_t idx = 0; idx < len_; idx++) {
	free(arr_[idx]);
      }

      free(arr_);
    }
  }
};

}

#endif
