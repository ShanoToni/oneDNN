/*******************************************************************************
* Copyright 2019-2024 Intel Corporation
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

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"
#include "gpu/generic/sycl/rnn/rnn_kernels.hpp"
#include "gpu/intel/utils.hpp"
#include "gpu/nvidia/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace rnn_utils;

#define PRINT_VEC(data, size) \
    { \
        void *raw_data = nullptr; \
        data.map_data(&raw_data, nullptr, size * sizeof(float)); \
        for (auto i = 0; i < size; i++) { \
            std::cout << #data << "[" << i \
                      << "] = " << static_cast<float *>(raw_data)[i] << "\n"; \
        } \
        std::cout << "\n\n"; \
        data.unmap_data(raw_data, nullptr); \
    }

#define PRINT_VEC2(data, size) \
    { \
        void *raw_data = nullptr; \
        data->map_data(&raw_data, nullptr, size * sizeof(float)); \
        for (auto i = 0; i < size; i++) { \
            std::cout << #data << "[" << i \
                      << "] = " << static_cast<float *>(raw_data)[i] << "\n"; \
        } \
        std::cout << "\n\n"; \
        data->unmap_data(raw_data, nullptr); \
    }
/*
template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::rnn_elemwise)) {
    memory_desc_t bias_md = types::zero_md();
    memory_desc_t in_out_md = types::zero_md();
    dims_t in_out_dims = {dim_t {1}, dim_t {1}, batch, 16};
    dims_t bias_dims = {dim_t {1}, dim_t {1}, dim_t {1}, 16};

    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());
    printf("\n ========= BEFORE BIAS ==========\n");
    PRINT_VEC2(scratch_gates.get_ptr(), 16)
    stream->wait();

    memory_desc_init_by_tag(
            bias_md, 4, bias_dims, data_type::f32, format_tag::ldgo);
    memory_desc_init_by_tag(
            in_out_md, 4, in_out_dims, data_type::f32, format_tag::ldgo);
    memory_t a(ctx.stream()->engine(), &in_out_md,
            scratch_gates.get_ptr()->clone());
    memory_t b(ctx.stream()->engine(), &bias_md,
            user_data.bias(lay, dir).get_ptr()->clone());
    memory_t c(ctx.stream()->engine(), &in_out_md,
            workspace.states(lay, dir, iter).get_ptr()->clone());
    exec_args_t bias_args;
    bias_args[DNNL_ARG_SRC_0] = {&a, true};
    bias_args[DNNL_ARG_SRC_1] = {&b, true};
    bias_args[DNNL_ARG_DST] = {&c, false};
    exec_ctx_t bias_ctx(ctx, std::move(bias_args));

    auto status = bias_primitive->execute(bias_ctx);

    printf("\n ========= AFTER BIAS ==========\n");
    stream->wait();
    PRINT_VEC2(scratch_gates.get_ptr(), 2 * 16)
    PRINT_VEC2(user_data.bias(lay, dir).get_ptr(), 2 * 16)

    return status;
}
*/
/*
template <prop_kind_t aprop>
bias_sig((_ref_rnn_common_t<aprop>::rnn_bias)) {
    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());
    parallel_for(ctx, bias_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        ws.gates(lay, dir, iter).get())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto bias_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        user_data.bias(lay, dir).get())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        ws.states(lay, dir, iter).get())
                          ->get_out_memory_arg(ctx.stream(), cgh);
        ref_rnn_bias bias_kernel(pd()->sycl_rnn_bias_conf_t_, src_mem_arg,
                bias_mem_arg, dst_mem_arg);

        size_t local_batch = 4;
        size_t local_channel = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_channels = calc_global_range(static_cast<size_t>(dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(
                        ::sycl::range<3>(1, global_batch, global_channels),
                        ::sycl::range<3>(1, local_batch, local_channel)),
                bias_kernel);
    });

    return status::success;
}
*/

/*
template elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
*/
//template bias_sig(ref_rnn_fwd_t::rnn_bias);
//template elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);

//template <prop_kind_t aprop>
//elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise)) {
//    return status::success;
//}

//template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
//template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);

//template <prop_kind_t aprop>
//elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise_u8s8)) {
//    return status::success;
//}

//template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
//template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);

// template elemwise_sig_gru(ref_rnn_fwd_t::gru_elemwise);
// template elemwise_sig_gru(ref_rnn_bwd_t::gru_elemwise);
//
} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
