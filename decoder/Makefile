
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE := $(shell python -c 'import numpy; print(numpy.get_include())')

all: boost-py kws.so

OPT = -O2

boost-py:
	sudo apt-get install libboost-dev -y
	sudo apt-get install libboost-python-dev -y

BOOST_LIB = /usr/lib/x86_64-linux-gnu
 
kws.so:
	g++ -std=c++11 $(OPT) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) \
		-D PYTHON -fPIC -c kws.cpp
	g++ -std=c++11 $(OPT) -shared -Wl,--export-dynamic kws.o -L$(BOOST_LIB) \
		-lboost_python -L/usr/lib/python$(PYTHON_VERSION)/config \
		-lpython$(PYTHON_VERSION) -o kws.so

clean:
	rm -f *.o *.so
