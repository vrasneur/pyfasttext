#ifndef EXIT_H
#define EXIT_H

#include <stdexcept>
#include <string>

#define exit(status) custom_exit(status)

inline void custom_exit(int status)
{
  throw std::runtime_error("fastext tried to exit: " + std::to_string(status));
}

#endif
