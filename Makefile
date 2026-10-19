CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -Wpedantic -std=c11

BUILD_DIR = rebuilt
LIB = $(BUILD_DIR)/libpmc_group.so
CONTEXT_SOURCES = context/context_mixer.c
CONTEXT_HEADERS = context/context_mixer.h
BIN_DIR = $(BUILD_DIR)/bin
PROGRAM_SOURCES = $(wildcard programs/*.c)
PROGRAM_HEADERS = $(wildcard programs/*.h)
PROGRAMS = $(basename $(notdir $(PROGRAM_SOURCES)))
PROGRAM_BINS = $(addprefix $(BIN_DIR)/,$(PROGRAMS))

.PHONY: all rebuilt test test-models test-all clean

all: rebuilt

rebuilt: $(LIB) $(PROGRAM_BINS)

test:
	python3 -m unittest discover -s tests -v

test-models:
	python3 -m unittest discover -s experiments/function_classification/tests -v

test-all: test test-models

$(LIB): pmc.c pmc.h $(CONTEXT_SOURCES) $(CONTEXT_HEADERS)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ pmc.c $(CONTEXT_SOURCES)

$(BIN_DIR)/%: programs/%.c $(PROGRAM_HEADERS) pmc.h $(LIB)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -I. -o $@ $< -L$(BUILD_DIR) -lpmc_group \
		-Wl,-rpath,'$$ORIGIN/..'

clean:
	rm -f $(LIB) $(PROGRAM_BINS)
