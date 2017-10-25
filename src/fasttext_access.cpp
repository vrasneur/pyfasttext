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

#include <iostream>

#include "fasttext_access.h"

#include "private_access.h"

using namespace fasttext;

namespace pyfasttext
{

ACCESS_INIT;

ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<Args>, args_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<Dictionary>, dict_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<Matrix>, input_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<Matrix>, output_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<QMatrix>, qinput_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<QMatrix>, qoutput_);
ALLOW_MEMBER_ACCESS(FastText, std::shared_ptr<Model>, model_);
ALLOW_MEMBER_ACCESS(FastText, std::atomic<int64_t>, tokenCount);
ALLOW_MEMBER_ACCESS(FastText, bool, quant_);
ALLOW_MEMBER_ACCESS(FastText, int32_t, version);
ALLOW_METHOD_ACCESS(FastText, bool, checkModel, std::istream&);

ALLOW_MEMBER_ACCESS(Dictionary, std::vector<int32_t>, word2int_);
ALLOW_MEMBER_ACCESS(Dictionary, std::vector<entry>, words_);
ALLOW_MEMBER_ACCESS(Dictionary, int32_t, size_);
ALLOW_MEMBER_ACCESS(Dictionary, int32_t, nwords_);
ALLOW_MEMBER_ACCESS(Dictionary, int32_t, nlabels_);
ALLOW_MEMBER_ACCESS(Dictionary, int64_t, ntokens_);
ALLOW_MEMBER_ACCESS(Dictionary, int64_t, pruneidx_size_);
typedef std::unordered_map<int32_t, int32_t> pruneidx_type;
ALLOW_MEMBER_ACCESS(Dictionary, pruneidx_type, pruneidx_);
ALLOW_CONST_METHOD_ACCESS(Dictionary, int32_t, find, const std::string&);
ALLOW_CONST_METHOD_ACCESS(Dictionary, void, pushHash, std::vector<int32_t>&, int32_t);
ALLOW_METHOD_ACCESS(Dictionary, void, initTableDiscard, );
ALLOW_METHOD_ACCESS(Dictionary, void, initNgrams, );

bool check_model(FastText &ft, const std::string &fname)
{
  std::ifstream ifs(fname, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return ACCESS(ft, checkModel)(ifs);
}

static void load_older_dict(std::shared_ptr<Dictionary> &dict, std::istream &ifs)
{
  INIT_ACCESS(*dict, words_);
  INIT_ACCESS(*dict, word2int_);
  INIT_ACCESS(*dict, size_);
  INIT_ACCESS(*dict, nwords_);
  INIT_ACCESS(*dict, nlabels_);
  INIT_ACCESS(*dict, ntokens_);

  words_.clear();
  std::fill(word2int_.begin(), word2int_.end(), -1);
  ifs.read((char*) &size_, sizeof(int32_t));
  ifs.read((char*) &nwords_, sizeof(int32_t));
  ifs.read((char*) &nlabels_, sizeof(int32_t));
  ifs.read((char*) &ntokens_, sizeof(int64_t));

  for (int32_t i = 0; i < size_; i++) {
    char c;
    entry e;
    while ((c = ifs.get()) != 0) {
      e.word.push_back(c);
    }
    ifs.read((char*) &e.count, sizeof(int64_t));
    ifs.read((char*) &e.type, sizeof(entry_type));
    words_.push_back(e);
    word2int_[ACCESS(*dict, find)(e.word)] = i;
  }
  
  ACCESS(*dict, pruneidx_size_) = 0;
  ACCESS(*dict, pruneidx_).clear();

  ACCESS(*dict, initTableDiscard)();
  ACCESS(*dict, initNgrams)();
}

// load older FastText models
// i.e. models that were generated before quantization support
void load_older_model(FastText &ft, const std::string &fname)
{
  std::ifstream ifs(fname, std::ifstream::binary);
  if (!ifs.is_open()) {
    std::cerr << "Model file cannot be opened for loading!" << std::endl;
    exit(EXIT_FAILURE);
  }
  
  INIT_ACCESS(ft, args_) = std::make_shared<Args>();
  INIT_ACCESS(ft, dict_) = std::make_shared<Dictionary>(ACCESS(ft, args_));
  INIT_ACCESS(ft, input_) = std::make_shared<Matrix>();
  INIT_ACCESS(ft, output_) = std::make_shared<Matrix>();
  INIT_ACCESS(ft, qinput_) = std::make_shared<QMatrix>();
  INIT_ACCESS(ft, qoutput_) = std::make_shared<QMatrix>();
  INIT_ACCESS(ft, quant_) = false;

  ACCESS(ft, version) = -1;

  args_->load(ifs);
  if(args_->model == model_name::sup) {
    // backward compatibility: old supervised models do not use char ngrams.
    args_->maxn = 0;
  }

  load_older_dict(dict_, ifs);
  input_->load(ifs);
  output_->load(ifs);

  INIT_ACCESS(ft, model_) = std::make_shared<Model>(input_, output_, args_, 0);
  
  model_->quant_ = quant_;
  model_->setQuantizePointer(qinput_, qoutput_, args_->qout);

  if (args_->model == model_name::sup) {
    model_->setTargetCounts(dict_->getCounts(entry_type::label));
  } else {
    model_->setTargetCounts(dict_->getCounts(entry_type::word));
  }
}

std::shared_ptr<const Args> get_fasttext_args(const FastText &ft)
{
  return ACCESS(ft, args_);
}

std::shared_ptr<fasttext::Model> get_fasttext_model(fasttext::FastText &ft)
{
  return ACCESS(ft, model_);
}

void set_fasttext_max_tokenCount(FastText &ft)
{
  const auto dict = ft.getDictionary();
  const auto args = get_fasttext_args(ft);
  
  ACCESS(ft, tokenCount) = args->epoch * dict->ntokens();
}

bool add_input_vector(const FastText &ft, Vector &vec, int32_t id)
{
  bool filled = false;
  
  if(id >= 0) {
    if(ACCESS(ft, quant_)) {
      vec.addRow(*ACCESS(ft, qinput_), id);
    }
    else {
      vec.addRow(*ACCESS(ft, input_), id);
    }

    filled = true;
  }
  
  return filled;
}

bool add_input_vector(const FastText &ft, Vector &vec, const std::string &ngram)
{
  const auto dict = ft.getDictionary();
  const int32_t id = dict->getId(ngram);

  return add_input_vector(ft, vec, id);
}

int32_t get_model_version(const FastText &ft)
{
  return ACCESS(ft, version);
}

bool is_model_quantized(const FastText &ft)
{
  return ACCESS(ft, quant_);
}

bool is_dict_pruned(const FastText &ft)
{
  const auto dict = ft.getDictionary();

  return (ACCESS(*dict, pruneidx_size_) != -1);
}

bool is_word_pruned(const FastText &ft, int32_t h)
{
  const auto dict = ft.getDictionary();

  // invalid hash?
  if(h == -1) {
    return true;
  }
  
  // dictionary not pruned at all?
  if(!is_dict_pruned(ft)) {
    return false;
  }

  // pruned dictionary, but ngram is still available?
  std::vector<int32_t> ngrams;
  ACCESS(*dict, pushHash)(ngrams, h - ACCESS(*dict, nwords_));
  
  return ngrams.empty();
}

std::map<std::string, ArgValue> get_args_map(const std::shared_ptr<const Args> &args)
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
