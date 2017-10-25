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

#ifndef PRIVATE_ACCESS_H
#define PRIVATE_ACCESS_H

// generic macros for private member/method access
// from http://bloglitb.blogspot.fr/2010/07/access-to-private-members-thats-easy.html
// the trick is: it is legal to pass the address of a private member/method
// in an explicit instanciation of a template

#define ACCESS_INIT						\
  template<typename Tag>					\
  struct result {						\
    typedef typename Tag::type type;				\
    static type ptr;						\
  };								\
								\
  template<typename Tag>					\
  typename result<Tag>::type result<Tag>::ptr;			\
								\
  template<typename Tag, typename Tag::type p>			\
    struct rob : result<Tag> {					\
    struct filler {						\
      filler() { result<Tag>::ptr = p; }			\
    };								\
    static filler filler_obj;					\
  };								\
								\
  template<typename Tag, typename Tag::type p>			\
    typename rob<Tag, p>::filler rob<Tag, p>::filler_obj

#define ALLOW_MEMBER_ACCESS(CLASS, TYPE, MEMBER)       \
  struct Only_##MEMBER { typedef TYPE CLASS::*type; }; \
  template class rob<Only_##MEMBER, &CLASS::MEMBER>

#define ALLOW_METHOD_ACCESS(CLASS, RET_TYPE, MEMBER, ...)		\
  struct Only_##MEMBER { typedef RET_TYPE(CLASS::*type)(__VA_ARGS__); }; \
  template class rob<Only_##MEMBER, &CLASS::MEMBER>

#define ALLOW_CONST_METHOD_ACCESS(CLASS, RET_TYPE, MEMBER, ...)		\
  struct Only_##MEMBER { typedef RET_TYPE(CLASS::*type)(__VA_ARGS__) const; }; \
  template class rob<Only_##MEMBER, &CLASS::MEMBER>

#define ACCESS(OBJECT, MEMBER) \
  ((OBJECT).*result<Only_##MEMBER>::ptr)

#define INIT_ACCESS(obj, member)		\
  decltype(ACCESS(obj, member)) &member = ACCESS(obj, member)

#endif
