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
