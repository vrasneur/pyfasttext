#include "fasttext_access.h"

#include "private_access.h"

using namespace fasttext;

ALLOW_ACCESS(FastText, std::shared_ptr<Dictionary>, dict_);

std::shared_ptr<Dictionary> &get_fasttext_dict(FastText &ft)
{
  return ACCESS(ft, dict_);
}
