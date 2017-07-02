#ifndef CUSTOM_EXIT_H
#define CUSTOM_EXIT_H

// make sure the exit() prototype is included before overriding its definition
#include <cstdlib>

#include <stdexcept>
#include <string>

#define exit(status) custom_exit(status)

inline void custom_exit(int status)
{
  throw std::runtime_error("fastext tried to exit: " + std::to_string(status));
}

#endif
