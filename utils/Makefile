
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE := $(shell python -c 'import numpy; print(numpy.get_include())')
BOOST_LIB = /usr/lib/x86_64-linux-gnu
FFTW = ../third_party/fftw-3.3.4/build/

OPT = -O2

specgram:
	g++ --std=c++11 $(OPT) -c specgram.cpp  -D PYTHON \
		-fPIC -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) \
		-I../third_party/ -I$(FFTW)/include
	g++ -std=c++11 $(OPT) -shared -Wl,--export-dynamic specgram.o \
	   	-L$(BOOST_LIB) -lboost_python \
		-L/usr/lib/python$(PYTHON_VERSION)/config \
		-lpython$(PYTHON_VERSION) -Wl,-rpath `pwd` \
		-L$(FFTW)/lib -lfftw3f -Wl,-rpath $(FFTW)/lib \
		-o cspecgram.so

clean:
	rm -f *.o *.so
