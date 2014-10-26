CXX = g++

ifeq (macos,macos)
	EXTRA_LIB_FLAGS = -framework Accelerate
endif

EXTRA_LIB_FLAGS = -lpthread
LIB_FLAGS = -larmadillo $(EXTRA_LIB_FLAGS)

OPT = -O2 -std=c++11 -march=native -funroll-loops

CXXFLAGS = $(DEBUG) $(FINAL) $(OPT) $(EXTRA_OPT)

all: run_mf

run_mf: main.cc
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_FLAGS)
	
.PHONY: = clean

clean:
	rm -rf run_mf



