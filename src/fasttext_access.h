#ifndef FASTTEXT_ACCESS_H
#define FASTTEXT_ACCESS_H

#include <memory>
#include <map>

#include "fastText/src/fasttext.h"

#include "variant/include/mapbox/variant.hpp"

namespace pyfasttext
{

bool check_model(fasttext::FastText &ft, const std::string &fname);

void load_older_model(fasttext::FastText &ft, const std::string &fname);

std::shared_ptr<fasttext::Args>& get_fasttext_args(fasttext::FastText &ft);

std::shared_ptr<fasttext::Dictionary>& get_fasttext_dict(fasttext::FastText &ft);

using ArgValue = mapbox::util::variant<bool, int, size_t, double, std::string, fasttext::loss_name, fasttext::model_name>;

std::map<std::string, ArgValue> get_args_map(const std::shared_ptr<fasttext::Args> &args);

std::string convert_loss_name(const fasttext::loss_name loss);

std::string convert_model_name(const fasttext::model_name model);

}

#endif
