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

std::shared_ptr<const fasttext::Args> get_fasttext_args(const fasttext::FastText &ft);

std::shared_ptr<fasttext::Model> get_fasttext_model(fasttext::FastText &ft);

void set_fasttext_max_tokenCount(fasttext::FastText &ft);

bool add_input_vector(const fasttext::FastText &ft, fasttext::Vector &vec, int32_t id);

bool add_input_vector(const fasttext::FastText &ft, fasttext::Vector &vec, const std::string &ngram);

int32_t get_model_version(const fasttext::FastText &ft);

bool is_model_quantized(const fasttext::FastText &ft);

bool is_dict_pruned(const fasttext::FastText &ft);

bool is_word_pruned(const fasttext::FastText &ft, int32_t h);

using ArgValue = mapbox::util::variant<bool, int, size_t, double, std::string, fasttext::loss_name, fasttext::model_name>;

std::map<std::string, ArgValue> get_args_map(const std::shared_ptr<const fasttext::Args> &args);

std::string convert_loss_name(const fasttext::loss_name loss);

std::string convert_model_name(const fasttext::model_name model);

}

#endif
