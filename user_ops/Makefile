
TF_INC := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
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
		-D_GLIBCXX_USE_CXX11_ABI=0 \
		-I $(TF_INC) -fPIC -L $(WCTC_LIB) \
        -I $(WCTC_INC) -lwarpctc -Wl,-rpath $(WCTC_LIB) \
		-I $(TF_INC)/external/nsync/public/ \
		-L $(TF_LIB) -ltensorflow_framework

ops: .warpctc
	nvcc -ccbin /usr/bin/cc -m64 -Xcompiler \
        ,\"-fopenmp\",\"-O2\",\"-g\",\"-fPIC\" \
         -gencode arch=compute_52,code=sm_52 --std=c++11 \
		 -D_GLIBCXX_USE_CXX11_ABI=0 \
         -I $(CUDA) -I$(WCTC_INC) -c -shared warp_ctc_gpu.cu \
		 -I $(TF_INC)/external/nsync/public/ \
		 -L $(TF_LIB) -ltensorflow_framework \
         -o libwarpwrap.so 
	g++ -std=c++11 -fopenmp -O2 -shared warp_ctc_ops.cc -o warp_ctc.so \
   		 -D_GLIBCXX_USE_CXX11_ABI=0 \
         -DNVCC -I $(TF_INC) -fPIC -L $(WCTC_LIB) \
         -I $(WCTC_INC) -lwarpctc -Wl,-rpath $(WCTC_LIB) \
         -I $(CUDA) -L. -lwarpwrap \
		 -I $(TF_INC)/external/nsync/public/ \
		 -L $(TF_LIB) -ltensorflow_framework

.warpctc:
	git clone https://github.com/baidu-research/warp-ctc.git
	cd warp-ctc && mkdir build && cd build && cmake ../ \
		&& make
	touch .warpctc

clean:
	rm -f *.so
