#ifndef PRIVATE_ACCESS_H
#define PRIVATE_ACCESS_H

// generic macros for private member access
// from https://stackoverflow.com/questions/424104/can-i-access-private-members-from-outside-the-class-without-using-friends

#define CONCATE_(X, Y) X##Y
#define CONCATE(X, Y) CONCATE_(X, Y)

#define ALLOW_ACCESS(CLASS, TYPE, MEMBER) \
  template<typename Only, TYPE CLASS::*Member>				\
    struct CONCATE(MEMBER, __LINE__) { friend TYPE (CLASS::*Access(Only*)) { return Member; } }; \
  template<typename> struct Only_##MEMBER;				\
  template<> struct Only_##MEMBER<CLASS> { friend TYPE (CLASS::*Access(Only_##MEMBER<CLASS>*)); }; \
  template struct CONCATE(MEMBER, __LINE__)<Only_##MEMBER<CLASS>, &CLASS::MEMBER>

#define ACCESS(OBJECT, MEMBER) \
  (OBJECT).*Access((Only_##MEMBER<std::remove_reference<decltype(OBJECT)>::type>*)nullptr)

#endif
