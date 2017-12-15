#include <vector>

#include <ctc.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("WarpCtc")
    .Input("inputs: float")
    .Input("input_lengths: int32")
    .Input("labels: int32")
    .Input("label_lengths: int32")
    .Output("loss: float")
    .Output("gradient: float");

class WarpCtcCpuOp: public OpKernel {
    public:
        explicit WarpCtcCpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

        void Compute(OpKernelContext* ctx) override {
            const Tensor* inputs_t;
            const Tensor* labels_t;
            const Tensor* input_lengths_t;
            const Tensor* label_lengths_t;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_t));
            OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths_t));
            OP_REQUIRES_OK(ctx, ctx->input("labels", &labels_t));
            OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths_t));

            const TensorShape& inputs_shape = inputs_t->shape();
            const int64 batch_size = inputs_shape.dim_size(1);
            const int64 num_classes = inputs_shape.dim_size(2);

            auto inputs = inputs_t->tensor<float, 3>().data();
            auto input_lengths = input_lengths_t->tensor<int, 1>().data();
            auto labels = labels_t->tensor<int, 1>().data();
            auto label_lengths = label_lengths_t->tensor<int, 1>().data();

            ctcOptions info;
            info.loc = CTC_CPU;
            info.num_threads = 8;
            info.blank_label = 0;

            // Grab the input tensor
            size_t cpu_alloc_bytes;
            get_workspace_size(label_lengths, input_lengths,
                               num_classes, batch_size, info,
                               &cpu_alloc_bytes);
            void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);

            TensorShape loss_shape({batch_size});
            Tensor* loss_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, loss_shape, &loss_tensor));
            auto losses = loss_tensor->template flat<float>().data();
            
            Tensor* grad_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, inputs_shape, &grad_tensor));
            auto grad = grad_tensor->template flat<float>().data();

            compute_ctc_loss(inputs, grad, labels, 
                             label_lengths, input_lengths,
                             num_classes, batch_size,
                             losses, ctc_cpu_workspace, info);
            free(ctc_cpu_workspace);

        }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtc")
                        .Device(DEVICE_CPU),
                        WarpCtcCpuOp);

// For GPU support

#ifdef NVCC
#include "warp_ctc.h"

class WarpCtcGpuOp: public OpKernel {
    public:
        explicit WarpCtcGpuOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

        void Compute(OpKernelContext* ctx) override {
            const Tensor* inputs_t;
            const Tensor* labels_t;
            const Tensor* input_lengths_t;
            const Tensor* label_lengths_t;
            OP_REQUIRES_OK(ctx, ctx->input("inputs", &inputs_t));
            OP_REQUIRES_OK(ctx, ctx->input("input_lengths", &input_lengths_t));
            OP_REQUIRES_OK(ctx, ctx->input("labels", &labels_t));
            OP_REQUIRES_OK(ctx, ctx->input("label_lengths", &label_lengths_t));

            const TensorShape& inputs_shape = inputs_t->shape();
            const int64 batch_size = inputs_shape.dim_size(1);
            const int64 num_classes = inputs_shape.dim_size(2);

            auto inputs = inputs_t->tensor<float, 3>().data();
            auto input_lengths = input_lengths_t->tensor<int, 1>().data();
            auto labels = labels_t->tensor<int, 1>().data();
            auto label_lengths = label_lengths_t->tensor<int, 1>().data();

            TensorShape loss_shape({batch_size});
            Tensor* loss_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(0, loss_shape, &loss_tensor));
            auto losses = loss_tensor->template flat<float>().data();
            
            Tensor* grad_tensor = NULL;
            OP_REQUIRES_OK(ctx, ctx->allocate_output(1, inputs_shape, &grad_tensor));
            auto grad = grad_tensor->template flat<float>().data();
            warp_ctc_gpu(inputs, grad, labels, 
                         label_lengths, input_lengths, 
                         num_classes, batch_size, 
                         losses);
        }
};

REGISTER_KERNEL_BUILDER(Name("WarpCtc")
                        .Device(DEVICE_GPU)
                        .HostMemory("input_lengths")
                        .HostMemory("labels")
                        .HostMemory("label_lengths")
                        .HostMemory("loss"),
                        WarpCtcGpuOp);
#endif
