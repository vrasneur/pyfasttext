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

#define ALLOW_METHOD_ACCESS(CLASS, RET_TYPE, PARAMS_TYPE, MEMBER)	\
  struct Only_##MEMBER { typedef RET_TYPE(CLASS::*type)(PARAMS_TYPE); }; \
  template class rob<Only_##MEMBER, &CLASS::MEMBER>

#define ALLOW_CONST_METHOD_ACCESS(CLASS, RET_TYPE, PARAMS_TYPE, MEMBER)	\
  struct Only_##MEMBER { typedef RET_TYPE(CLASS::*type)(PARAMS_TYPE) const; }; \
  template class rob<Only_##MEMBER, &CLASS::MEMBER>

#define ACCESS(OBJECT, MEMBER) \
  ((OBJECT).*result<Only_##MEMBER>::ptr)

#define INIT_ACCESS(obj, member)		\
  decltype(ACCESS(obj, member)) &member = ACCESS(obj, member)

#endif
