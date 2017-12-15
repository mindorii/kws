
TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
WCTC_LIB := $(shell echo `pwd`/warp-ctc/build)
WCTC_INC := $(shell echo `pwd`/warp-ctc/include)
CUDA = /usr/local/cuda/include

NVCC := $(shell echo `which nvcc`)
ifeq ($(NVCC),)
	OPS = cpu_only
else
	OPS = ops
endif

all: .warpctc $(OPS)

cpu_only:
	g++ -std=c++11 -fopenmp -O2 -shared warp_ctc_ops.cc -o warp_ctc.so \
         -I $(TF_INC) -fPIC -L $(WCTC_LIB) \
         -I $(WCTC_INC) -lwarpctc -Wl,-rpath $(WCTC_LIB) \

ops: .warpctc
	nvcc -ccbin /usr/bin/cc -m64 -Xcompiler \
        ,\"-fopenmp\",\"-O2\",\"-g\",\"-fPIC\" \
         -gencode arch=compute_52,code=sm_52 --std=c++11 \
         -I$(CUDA) -I$(WCTC_INC) -c -shared warp_ctc_gpu.cu \
         -o libwarpwrap.so 
	g++ -std=c++11 -fopenmp -O2 -shared warp_ctc_ops.cc -o warp_ctc.so \
         -DNVCC -I $(TF_INC) -fPIC -L $(WCTC_LIB) \
         -I $(WCTC_INC) -lwarpctc -Wl,-rpath $(WCTC_LIB) \
         -I $(CUDA) -L. -lwarpwrap

.warpctc:
	git clone https://github.com/baidu-research/warp-ctc.git
	cd warp-ctc && mkdir build && cd build && cmake ../ \
		&& make
	touch .warpctc

clean:
	rm -f *.so
