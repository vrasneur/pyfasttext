#ifndef COMPAT_H
#define COMPAT_H

#include <memory>

namespace pyfasttext
{

// std::make_unique() is only available in C++14
template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args)
{
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}

#endif
