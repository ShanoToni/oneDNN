/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP
#define GPU_NVIDIA_CUDNN_MATMUL_IMPL_HPP

#include "cudnn.h"

#include "gpu/nvidia/cudnn_matmul_base_impl.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

struct cudnn_matmul_impl_t : cudnn_matmul_base_impl_t {

    status_t init(matmul_pd_t *pd) {

        CHECK(get_cublas_data_type(pd->src_md()->data_type, src_type_));

        CHECK(get_cublas_data_type(pd->weights_md()->data_type, weights_type_));

        isbatched_ = pd->batched();

        memory_desc_wrapper src_d = memory_desc_wrapper(pd->src_md());
        memory_desc_wrapper weights_d = memory_desc_wrapper(pd->weights_md());
        memory_desc_wrapper dst_d = memory_desc_wrapper(pd->dst_md());

        if (!(src_d.is_plain() && weights_d.is_plain() && dst_d.is_plain())) {
            return status::unimplemented;
        }

        with_dst_scale_
                = !pd->attr()->scales_.get(DNNL_ARG_DST).has_default_values();
        with_separate_bias_ = pd->with_bias();
        if ((with_separate_bias_)
                && (pd->weights_md(1)->data_type != pd->dst_md()->data_type)) {
            // When datatype of bias is different from the dst,
            // we need to reorder the output.
            bias_dt_mismatch_ = true;
            reorder_required_ = true;
            CHECK(get_cublas_data_type(
                    pd->weights_md(1)->data_type, dst_type_));
        } else {
            CHECK(get_cublas_data_type(pd->dst_md()->data_type, dst_type_));
        }

        // cuBLAS only supports s8s8f32 configuration.
        // Hence, one final reorder is required if the cfg = s8s8s8
        if (dst_type_ == cudaDataType_t::CUDA_R_8I) {
            reorder_required_ = true;
            dst_type_ = cudaDataType_t::CUDA_R_32F;
        }

        if (with_eltwise(0, pd) || with_eltwise(1, pd)) {
            with_separate_eltwise_ = true;
            CHECK(create_and_set_op_descriptor(pd, act_desc_));
        }

        // Set parameter when post-op sum is specified
        if (with_sum(pd)) { post_op_sum_ = sum_scale(pd); }

        has_runtime_params_ = src_d.has_runtime_dims_or_strides()
                || dst_d.has_runtime_dims_or_strides()
                || weights_d.has_runtime_dims_or_strides();

        if (!has_runtime_params_) {
            // Initialise all gemm parameters if there are no runtime parameters
            init_parameters(src_d, weights_d, dst_d,
                    memory_desc_wrapper(pd->weights_md(1)));
            init_scratchpad(pd);
        }

        return status::success;
    }

    status_t init_gemm_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d) override {

        if (isbatched_) batch_count_ = dst_d.dims()[0];
        const dim_t M = dst_d.dims()[isbatched_ + 1];
        const dim_t N = dst_d.dims()[isbatched_ + 0];
        const dim_t K = src_d.dims()[isbatched_ + 1];

        M_ = (int)M;
        N_ = (int)N;
        K_ = (int)K;

        const auto &dst_strides = &dst_d.blocking_desc().strides[isbatched_];
        const auto &src_strides = &src_d.blocking_desc().strides[isbatched_];
        const auto &weights_strides
                = &weights_d.blocking_desc().strides[isbatched_];

        // A matrix is the weights
        transA_ = weights_strides[1] == 1
                        && weights_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;
        // B matrix is the src
        transB_ = src_strides[1] == 1 && src_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;
        // C matrix is the dst
        transC_ = dst_strides[1] == 1 && dst_d.dims()[isbatched_ + 0] > 1
                ? cublasOperation_t::CUBLAS_OP_N
                : cublasOperation_t::CUBLAS_OP_T;

        lda_ = get_ld(weights_d, transA_);
        ldb_ = get_ld(src_d, transB_);
        ldc_ = get_ld(dst_d, transC_);

        if (isbatched_) {
            // These parameters are required for cublasGemmStridedBatchedEx()
            stride_a_ = get_batch_stride(weights_d);
            stride_b_ = get_batch_stride(src_d);
            stride_c_ = get_batch_stride(dst_d);

            // Enable broadcast semantics.
            if (src_d.dims()[0] > weights_d.dims()[0])
                stride_a_ = 0;
            else if (src_d.dims()[0] < weights_d.dims()[0])
                stride_b_ = 0;
        }

        return status::success;
    }

    status_t init_parameters(const memory_desc_wrapper src_d,
            const memory_desc_wrapper weights_d,
            const memory_desc_wrapper dst_d, const memory_desc_wrapper bias_d) {
        // Matmul supports runtime paramters for dimensions and scales.
        // We need to initialize them in the execute function.
        CHECK(init_gemm_parameters(src_d, weights_d, dst_d));

        if (with_separate_bias_ || reorder_required_ || with_separate_eltwise_
                || with_dst_scale_) {
            // Initialise cuDNN descriptors
            cudnnDataType_t data_types[NUM_IO];
            int ndims = dst_d.ndims() < 4 ? 4 : dst_d.ndims();
            int dims[NUM_IO][DNNL_MAX_NDIMS];
            int strides[NUM_IO][DNNL_MAX_NDIMS];

            convert_dims_matmul(dst_d.dims(), dims[dst], dst_d.ndims());
            CHECK(convert_data_type(dst_d.md_, &data_types[dst], false));
            convert_dims_matmul(
                    dst_d.blocking_desc().strides, strides[dst], dst_d.ndims());
            CHECK(create_and_set_tensor_descriptor(&tensor_descs_[dst],
                    data_types[dst], ndims, dims[dst], strides[dst]));

            if (reorder_required_ && !bias_dt_mismatch_) {
                // If reorder is required, we need to create a scratchpad memory
                // to store the intermediate result
                CHECK(create_and_set_tensor_descriptor(&temp_mem_desc_,
                        cudnnDataType_t::CUDNN_DATA_FLOAT, ndims, dims[dst],
                        strides[dst]));
            }

            if (with_separate_bias_) {
                // Create bias and destination tensor descriptors
                convert_dims_matmul(bias_d.dims(), dims[bias], bias_d.ndims());
                convert_dims_matmul(bias_d.blocking_desc().strides,
                        strides[bias], bias_d.ndims());
                CHECK(convert_data_type(bias_d.md_, &data_types[bias], false));
                CHECK(create_and_set_tensor_descriptor(&tensor_descs_[bias],
                        data_types[bias], ndims, dims[bias], strides[bias]));
                if (bias_dt_mismatch_) {
                    CHECK(create_and_set_tensor_descriptor(&temp_mem_desc_,
                            data_types[bias], ndims, dims[dst], strides[dst]));
                }
            }
        }

        const auto dst_nelems = dst_d.nelems(true);
        reorder_scratch_size_ = dst_nelems * sizeof(float);

        return status::success;
    }

    void init_scratchpad(matmul_pd_t *pd) override {
        if (reorder_scratch_size_ > 0) {
            auto scratchpad = pd->scratchpad_registry().registrar();
            scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
                    reorder_scratch_size_, 1);
        }
    }

    void execute(cublasHandle_t cublas_handle, cudnnHandle_t cudnn_handle,
            void *a, void *b, void *c, void *bias, void *reorder_scratch,
            void *src_scale, void *wei_scale, void *dst_scale) {
        float gemm_beta = 0;
        if (!bias_dt_mismatch_ && !reorder_required_) {
            // Case where no reorder is required, scratchpad points to dst (c)
            reorder_scratch = c;
            temp_mem_desc_ = tensor_descs_[io::dst];
            gemm_beta = post_op_sum_;
        }
        auto flip_op = [](cublasOperation_t op) {
            return (op == cublasOperation_t::CUBLAS_OP_T)
                    ? cublasOperation_t::CUBLAS_OP_N
                    : cublasOperation_t::CUBLAS_OP_T;
        };

        float scale = 1.0f;
        float host_dst_scale = 1.0f;
        if (src_scale) {
            float host_src_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_src_scale,
                    (CUdeviceptr)src_scale, sizeof(float));
            scale *= host_src_scale;
        }
        if (wei_scale) {
            float host_wei_scale = 1.0f;
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_wei_scale,
                    (CUdeviceptr)wei_scale, sizeof(float));
            scale *= host_wei_scale;
        }
        if (dst_scale) {
            CUDA_EXECUTE_FUNC(cuMemcpy, (CUdeviceptr)&host_dst_scale,
                    (CUdeviceptr)dst_scale, sizeof(float));
            // For eltwise post-ops, apply the dst scale afterward
            if (!with_separate_eltwise_) scale /= host_dst_scale;
        }

        if (isbatched_) {
            // Calls cublasGemmStridedBatchedEx()
            if (transC_ == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        flip_op(transB_), flip_op(transA_), N_, M_, K_, &scale,
                        b, src_type_, ldb_, stride_b_, a, weights_type_, lda_,
                        stride_a_, &gemm_beta, reorder_scratch, dst_type_, ldc_,
                        stride_c_, batch_count_, acc_type_, gemm_algo_);

            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmStridedBatchedEx, cublas_handle,
                        transA_, transB_, M_, N_, K_, &scale, a, weights_type_,
                        lda_, stride_a_, b, src_type_, ldb_, stride_b_,
                        &gemm_beta, reorder_scratch, dst_type_, ldc_, stride_c_,
                        batch_count_, acc_type_, gemm_algo_);
            }
        } else {
            // Calls cublasGemmEx()
            if (transC_ == cublasOperation_t::CUBLAS_OP_T) {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle,
                        flip_op(transB_), flip_op(transA_), N_, M_, K_, &scale,
                        b, src_type_, ldb_, a, weights_type_, lda_, &gemm_beta,
                        reorder_scratch, dst_type_, ldc_, acc_type_,
                        gemm_algo_);
            } else {
                CUBLAS_EXECUTE_FUNC(cublasGemmEx, cublas_handle, transA_,
                        transB_, M_, N_, K_, &scale, a, weights_type_, lda_, b,
                        src_type_, ldb_, &gemm_beta, reorder_scratch, dst_type_,
                        ldc_, acc_type_, gemm_algo_);
            }
        }
        handle_post_ops(cudnn_handle, c, bias, reorder_scratch, host_dst_scale);
    }

    ~cudnn_matmul_impl_t() { cleanup(); }

    void cleanup() override {
        if (act_desc_) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyActivationDescriptor, act_desc_);
            act_desc_ = nullptr;
        }
        if ((reorder_required_ && !bias_dt_mismatch_)
                || ((with_separate_bias_ && bias_dt_mismatch_)
                        && temp_mem_desc_)) {
            CUDNN_EXECUTE_FUNC_V(cudnnDestroyTensorDescriptor, temp_mem_desc_);
            temp_mem_desc_ = nullptr;
        }
        for (size_t i = 0; i < NUM_IO; i++) {
            if (tensor_descs_[i]) {
                CUDNN_EXECUTE_FUNC_V(
                        cudnnDestroyTensorDescriptor, tensor_descs_[i]);
                tensor_descs_[i] = nullptr;
            }
        }
    }

private:
    cublasOperation_t transA_;
    cublasOperation_t transB_;
    cublasOperation_t transC_;
    int M_, N_, K_;
    cudaDataType_t src_type_, weights_type_, dst_type_;
    cublasGemmAlgo_t gemm_algo_
            = cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
