CACHE_DIR := $(HOME)/.cache/pyfasttext
CACHE_ENWIK8_TEXT := $(CACHE_DIR)/enwik8.txt
CACHE_LID_FTZ := $(CACHE_DIR)/lid.176.ftz

all: test_enwik8 test_lid
	@echo "All tests are OK!"

test_enwik8: $(CACHE_ENWIK8_TEXT)
	TEST_ARG="$<" python -m doctest test_enwik8.txt

test_lid: $(CACHE_LID_FTZ)
	TEST_ARG="$<" python -m doctest test_lid.txt

$(CACHE_DIR):
	mkdir -p $@

$(CACHE_ENWIK8_TEXT): $(CACHE_DIR)
	wget 'http://mattmahoney.net/dc/enwik8.zip'
	unzip enwik8.zip
	perl ../src/fastText/wikifil.pl enwik8 > $@

$(CACHE_LID_FTZ): $(CACHE_DIR)
	wget 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/supervised_models/lid.176.ftz' -O $@
