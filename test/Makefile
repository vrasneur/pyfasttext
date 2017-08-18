CACHE_DIR := $(HOME)/.cache/pyfasttext
CACHE_TEXT := $(CACHE_DIR)/enwik8.txt

all: test

test: $(CACHE_TEXT)
	python -m doctest test.txt "$(CACHE_TEXT)"
	@echo "All tests are OK!"

$(CACHE_DIR):
	mkdir -p $@

$(CACHE_TEXT): $(CACHE_DIR)
	wget 'http://mattmahoney.net/dc/enwik8.zip'
	unzip enwik8.zip
	perl ../src/fastText/wikifil.pl enwik8 > $@
