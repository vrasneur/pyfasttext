#ifndef CUSTOM_EXIT_H
#define CUSTOM_EXIT_H

// make sure the exit() prototype is included before overriding its definition
#include <cstdlib>

#include <stdexcept>
#include <sstream>

#define exit(status) custom_exit(status)

inline void custom_exit(int status)
{
  std::stringstream ss;
  ss << "fastext tried to exit: " << status;
  
  throw std::runtime_error(ss.str());
}

#endif
