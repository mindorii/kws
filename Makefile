
FFTW = fftw-3.3.4

all: .deps fftw decode ops utils

.deps:
	sudo apt-get install libsamplerate-dev -y
	sudo pip install -r requirements.txt
	touch .deps

.PHONY: decode ops utils

fftw:
	mkdir third_party && cd third_party && \
	wget http://www.fftw.org/$(FFTW).tar.gz && \
	tar -xzf $(FFTW).tar.gz && \
	rm $(FFTW).tar.gz && \
	cd $(FFTW) && \
	mkdir -p build && \
	./configure --prefix=`pwd`/build \
		--enable-float --enable-shared && \
	make clean && \
	make -j 4 && \
	make install

decode:
	$(MAKE) -C decoder

ops:
	$(MAKE) -C user_ops

utils:
	$(MAKE) -C utils
	
clean:
	$(MAKE) -C utils clean
	$(MAKE) -C user_ops clean
	$(MAKE) -C decoder clean
