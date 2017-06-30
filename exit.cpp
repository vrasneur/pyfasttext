#include <cstdlib>

#include <stdexcept>
#include <string>

#include "exit.h"

asm(".symver exit, exit@GLIBC_2.2.5");

void __wrap_exit(int status)
{
  throw std::runtime_error("fastext tried to exit: " + std::to_string(status));
}
