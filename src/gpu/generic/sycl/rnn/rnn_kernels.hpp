/*******************************************************************************
* Copyright 2023-2024 Intel Corporation
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

#ifndef SRC_GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP
#define SRC_GPU_GENERIC_SYCL_RNN_RNN_KERNELS_HPP

#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

// #define OFF5(i0, D0, i1, D1, i2, D2, i3, D3, i4, D4) \
//     (((((i0) * (D1) + (i1)) * (D2) + (i2)) * (D3) + (i3)) * (D4) + (i4))

#define COPY_SRC_LAYER true

int off_ws_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = COPY_SRC_LAYER ? i0_ + 1 : i0_;
    int i0_size = COPY_SRC_LAYER ? n_layer + 1 : n_layer;
    int i2 = i2_ + 1;
    return OFF5(i0, i0_size, i1, n_dir, i2, n_iter + 1, i3, batch, i4,
            states_ws_ld);
}

struct ref_rnn_copy_init_layer_t{
    ref_rnn_copy_init_layer_t(const sycl_rnn_copy_init_layer_conf_t &conf,
        xpu::sycl::in_memory_arg_t src,
        xpu::sycl::out_memory_arg_t dst): src_{src}, dst_{dst}, conf_{conf}{}


    void operator()(::sycl::nd_item<3> item) const {
    const int t = item.get_global_id(0); // timestep
    const int n = item.get_global_id(1); // batch
    const int c = item.get_global_id(2); // channel

    if (t >= conf_.n_iter || n >= conf_.batch || c >= conf_.slc) return;

    dim_t src_offset = src_data_offset(t, n, c);
    auto src = load_float_value(src_md().data_type(), src_ptr(), src_offset);


    dim_t dst_offset = 0;
    if(conf_.lr){
        dst_offset = dst_data_offset(0, t, n, c);
        store_float_value(src_md().data_type(), src, dst_ptr(), dst_offset);
    }
    if(conf_.rl){
        dst_offset = dst_data_offset(conf_.n_dir - 1, conf_.n_iter - t - 1, n, c);
        store_float_value(src_md().data_type(), src, dst_ptr(), dst_offset);
    }

    // auto dst = load_float_value();
 
    // __global WS_STATE_DATA_T *src = (__global WS_STATE_DATA_T *)src_base
    //         + src_l_off(it, b, c);

    // if (lr) {
    //     dst = dst_base
    //             + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, -1,
    //                     0, it, b, c);
    //     dst[0] = src[0];
    // }
    // if (rl) {
    //     dst = dst_base
    //             + off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, -1,
    //                     n_dir - 1, n_iter - it - 1, b, c);
    //     dst[0] = src[0];
    // }
    }

    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    inline dim_t src_data_offset(dim_t t, dim_t n, dim_t c) const {
        return conf_.src_md.off(t, n, c);
    }

    inline dim_t dst_data_offset(dim_t d, dim_t t, dim_t n, dim_t c) const{
        return off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter, conf_.batch, conf_.states_ws_ld, -1,
                        d, t, n, c);
    }

    // inline dim_t dst_data_offset()


    // int src_l_off(int iter, int batch, int slc) {
    //     return ((iter % conf_.dims[0]) * conf_.block_strides[0] + (iter / conf_.dims[0]) * conf_.strides[0]
    //             + (batch % conf_.dims[1]) * conf_.block_strides[1] + (batch / conf_.dims[1]) * conf_.strides[1]
    //             + (slc % conf_.dims[2]) * conf_.blocks_strides[2] + (slc / conf_.dims[2]) * conf_.strides[2]);
    // }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_init_layer_conf_t conf_;
};

}
}
}
}
}

#endif