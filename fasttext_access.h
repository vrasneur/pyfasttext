#ifndef FASTTEXT_ACCESS_H
#define FASTTEXT_ACCESS_H

#include <memory>

#include <fastText/src/fasttext.h>

std::shared_ptr<fasttext::Dictionary>& get_fasttext_dict(fasttext::FastText &ft);

#endif
