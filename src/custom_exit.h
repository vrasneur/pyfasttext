/* pyfasttext
 * Yet another Python binding for fastText
 * 
 * Copyright (c) 2017 Vincent Rasneur <vrasneur@free.fr>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; version 3 only.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 */  

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
