#include "fasttext_access.h"

#include "private_access.h"

using namespace fasttext;

namespace pyfasttext
{

ALLOW_ACCESS(FastText, std::shared_ptr<Dictionary>, dict_);
ALLOW_ACCESS(FastText, std::shared_ptr<Args>, args_);

std::shared_ptr<Dictionary> &get_fasttext_dict(FastText &ft)
{
  return ACCESS(ft, dict_);
}

std::shared_ptr<Args> &get_fasttext_args(FastText &ft)
{
  return ACCESS(ft, args_);
}

std::map<std::string, ArgValue> get_args_map(const std::shared_ptr<Args> &args)
{
  std::map<std::string, ArgValue> vals;

#define ITEM(name) vals.emplace(#name, args->name);
#include "args.itm"
#undef ITEM
  
  return vals;
}

std::string convert_loss_name(const fasttext::loss_name loss)
{
  switch(loss) {
  case fasttext::loss_name::hs:
    return "hs";
  case fasttext::loss_name::ns:
    return "ns";
  case fasttext::loss_name::softmax:
    return "softmax";
  default:
    return "unknown";
  }
}

std::string convert_model_name(const fasttext::model_name model)
{
  switch(model) {
  case fasttext::model_name::cbow:
    return "cbow";
  case fasttext::model_name::sg:
    return "skipgram";
  case fasttext::model_name::sup:
    return "supervised";
  default:
    return "unknown";
  }
}

}
