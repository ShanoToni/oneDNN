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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "gpu/generic/sycl/sycl_io_helper.hpp"
#include "gpu/generic/sycl/sycl_math_utils.hpp"
#include "gpu/generic/sycl/sycl_primitive_conf.hpp"
#include "xpu/sycl/types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

#ifdef OFF5
#undef OFF5
#endif

#ifndef OFF5
#define OFF5(i0, D0, i1, D1, i2, D2, i3, D3, i4, D4) \
    (((((i0) * (D1) + (i1)) * (D2) + (i2)) * (D3) + (i3)) * (D4) + (i4))
#endif

#define COPY_SRC_LAYER true

inline int off_ws_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = COPY_SRC_LAYER ? i0_ + 1 : i0_;
    int i0_size = COPY_SRC_LAYER ? n_layer + 1 : n_layer;
    int i2 = i2_ + 1;
    ::sycl::ext::oneapi::experimental::printf(
            "OFFSET: \n i0 :%d, i0_size:%d, i1:%d, n_dir:%d, i2:%d, n_iter "
            ":%d, i3:%d, batch:%d, i4:%d, states_ws_ld:%d \n",
            i0, i0_size, i1, n_dir, i2, n_iter, i3, batch, i4, states_ws_ld);
    return OFF5(i0, i0_size, i1, n_dir, i2, n_iter + 1, i3, batch, i4,
            states_ws_ld);
}

//off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
//                conf_.batch, conf_.states_ws_ld, lay, dir, -1, n, c);
#ifdef OFF2
#undef OFF2
#endif

#ifndef OFF2
#define OFF2(i0, D0, i1, D1) ((i0) * (D1) + (i1))
#endif

inline int off_ker_bias(int dhc, int i0, int i1, int n_gates) {
    return OFF2(i0, n_gates, i1, dhc);
}

inline int cell_ws_state(int states_ws_ld, int i, int j) {
    return i * states_ws_ld + j;
}

inline int cell_scratch_mem(
        int scratch_gates_ld, int dhc, int i, int n, int j) {
    return i * scratch_gates_ld + n * dhc + j;
}

inline int off_out_ws_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = false ? i0_ + 1 : i0_;
    int i0_size = false ? n_layer + 1 : n_layer;
    int i2 = i2_ + 1;
    ::sycl::ext::oneapi::experimental::printf(
            "OFFSET: \n i0 :%d, i0_size:%d, i1:%d, n_dir:%d, i2:%d, n_iter "
            ":%d, i3:%d, batch:%d, i4:%d, states_ws_ld:%d \n",
            i0, i0_size, i1, n_dir, i2, n_iter, i3, batch, i4, states_ws_ld);
    return OFF5(
            i0, i0_size, i1, n_dir, i2, n_iter, i3, batch, i4, states_ws_ld);
}

inline int off_ws_c_state(int n_layer, int n_dir, int n_iter, int batch,
        int states_ws_ld, int i0_, int i1, int i2_, int i3, int i4) {
    int i0 = i0_;
    int i0_size = n_layer;
    int i2 = i2_ + 1;
    return OFF5(i0, i0_size, i1, n_dir, i2, n_iter + 1, i3, batch, i4,
            states_ws_ld);
}

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif

struct ref_rnn_copy_init_layer_t {
    ref_rnn_copy_init_layer_t(const sycl_rnn_copy_init_layer_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src,
            xpu::sycl::out_memory_arg_t &dst)
        : src_ {src}, dst_ {dst}, conf_ {conf} {}

    void operator()(::sycl::nd_item<3> item) const {
        const dim_t t = item.get_global_id(0); // timestep
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (t >= conf_.n_iter || n >= conf_.batch || c >= conf_.slc) return;

        dim_t src_offset = src_data_offset(t, n, c);
        auto src
                = load_float_value(src_md().data_type(), src_ptr(), src_offset);

        dim_t dst_offset = 0;
        if (conf_.lr) {
            dst_offset = dst_data_offset(0, t, n, c);
            auto dst_val = load_float_value(
                    src_md().data_type(), dst_ptr(), dst_offset);
            ::sycl::ext::oneapi::experimental::printf(
                    "INIT: \n src_offset: %lu, src_v: %f dst: %f dst_offset: "
                    "%lu \n",
                    src_offset, src, dst_val, dst_offset);
            store_float_value(src_md().data_type(), src, dst_ptr(), dst_offset);
        }
    }

    const xpu::sycl::md_t &src_md() const { return conf_.src_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    inline dim_t src_data_offset(dim_t t, dim_t n, dim_t c) const {
        return conf_.src_md.off(t, n, c);
    }

    inline dim_t dst_data_offset(dim_t d, dim_t t, dim_t n, dim_t c) const {
        return off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, -1, d, t - 1, n, c);
    }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_init_layer_conf_t conf_;
};

struct ref_rnn_copy_init_iter_t {
    ref_rnn_copy_init_iter_t(const sycl_rnn_copy_init_iter_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src_iter,
            const xpu::sycl::in_memory_arg_t &src_iter_c,
            xpu::sycl::out_memory_arg_t &workspace)
        : conf_ {conf}
        , src_iter_ {src_iter}
        , src_iter_c_ {src_iter_c}
        , workspace_ {workspace} {}

    void operator()(::sycl::nd_item<3> item) const {
        const dim_t lay = item.get_global_id(0) / conf_.n_dir; // layer
        const dim_t dir = item.get_global_id(0) % conf_.n_dir; // direction
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (lay >= conf_.n_layer || dir >= conf_.n_dir || n >= conf_.batch
                || c >= conf_.sic)
            return;

        dim_t src_iter_offset = src_iter_data_offset(lay, dir, n, c);
        auto src_iter = src_iter_ptr();
        float src_iter_v = 0.0f;
        if (src_iter) {
            src_iter_v = load_float_value(
                    src_iter_md().data_type(), src_iter_ptr(), src_iter_offset);
            if (conf_.quantize) {
                src_iter_v = conf_.scale * src_iter_v + conf_.shift;
            }
        }
        dim_t dst_iter_offset = dst_iter_data_offset(lay, dir, n, c);
        auto dst_val = load_float_value(
                src_iter_md().data_type(), workspace_ptr(), dst_iter_offset);
        ::sycl::ext::oneapi::experimental::printf(
                "INIT: \n src_offset: %lu, src_v: %f dst: %f dst_offset: "
                "%lu \n",
                src_iter_offset, src_iter_v, dst_val, dst_iter_offset);
        store_float_value(src_iter_md().data_type(), src_iter_v,
                workspace_ptr(), dst_iter_offset);

        if (conf_.with_iter_c) {
            dim_t src_iter_c_offset = src_iter_c_data_offset(lay, dir, n, c);
            auto src_iter_c = src_iter_c_ptr();
            float src_iter_c_v = 0.0f;
            if (src_iter_c) {
                src_iter_c_v = load_float_value(src_iter_c_md().data_type(),
                        src_iter_c_ptr(), src_iter_c_offset);
                if (conf_.quantize) {
                    src_iter_c_v = conf_.scale * src_iter_c_v + conf_.shift;
                }
            }
            int dst_iter_c_offset = dst_iter_c_data_offset(lay, dir, n, c);
            store_float_value(src_iter_c_md().data_type(), src_iter_c_v,
                    workspace_ptr(), dst_iter_c_offset);
        }
    }

    const xpu::sycl::md_t &src_iter_md() const { return conf_.src_iter_md; }
    const xpu::sycl::md_t &src_iter_c_md() const { return conf_.src_iter_c_md; }
    void *src_iter_ptr() const { return src_iter_.get_pointer(); }
    void *src_iter_c_ptr() const { return src_iter_c_.get_pointer(); }
    void *workspace_ptr() const { return workspace_.get_pointer(); }

    inline dim_t src_iter_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return conf_.src_iter_md.off(lay, dir, n, c);
    }

    inline dim_t src_iter_c_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return conf_.src_iter_c_md.off(lay, dir, n, c);
    }

    inline dim_t dst_iter_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, lay-1, dir, conf_.n_iter-1, n, c);
    }

    inline dim_t dst_iter_c_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return off_ws_c_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, lay, dir, 0, n, c);
    }

    sycl_rnn_copy_init_iter_conf_t conf_;
    xpu::sycl::in_memory_arg_t src_iter_;
    xpu::sycl::in_memory_arg_t src_iter_c_;
    xpu::sycl::out_memory_arg_t workspace_;
};

struct ref_rnn_copy_res_layer_t {
    ref_rnn_copy_res_layer_t(const sycl_rnn_copy_res_layer_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src,
            xpu::sycl::out_memory_arg_t &dst)
        : src_ {src}, dst_ {dst}, conf_ {conf} {}

    void operator()(::sycl::nd_item<3> item) const {

        const dim_t t = item.get_global_id(0); // timestep
        const dim_t n = item.get_global_id(1); // batch
        const dim_t c = item.get_global_id(2); // channel

        if (c >= conf_.dhc || n >= conf_.batch || t >= conf_.n_iter) return;

        auto src = src_ptr();
        auto dst = dst_ptr();

        int dir = 0;
        if (conf_.lr) {
            bool dequantize_at_copy = conf_.dequantize
                    && conf_.direction
                            != dnnl_rnn_direction_t::dnnl_bidirectional_sum;
            auto src_offset = src_data_offset(dir, t, n, c);
            auto src_v
                    = load_float_value(dst_md().data_type(), src, src_offset);

            if (dequantize_at_copy) {
                src_v = (src_v - conf_.shift) / conf_.scale;
            }

            auto dst_offset = dst_data_offset(t, n, dir * conf_.dhc + c);
            auto dst_val
                    = load_float_value(dst_md().data_type(), dst, dst_offset);
            ::sycl::ext::oneapi::experimental::printf(
                    "RES: \n src_offset: %lu, src_v: %f dst: %f dst_offset: "
                    "%lu \n",
                    src_offset, src_v, dst_val, dst_offset);

            store_float_value(dst_md().data_type(), src_v, dst, dst_offset);

            dir = 1;
        }
        if (conf_.rl) {
            if (conf_.direction
                    == dnnl_rnn_direction_t::dnnl_bidirectional_sum) {
                auto src_offset
                        = src_data_offset(dir, conf_.n_iter - t - 1, n, c);
                auto dst_offset = dst_data_offset(t, n, c);
                auto src_v = load_float_value(
                        dst_md().data_type(), src, src_offset);
                auto dst_v = load_float_value(
                        dst_md().data_type(), dst, dst_offset);
                float val = src_v + dst_v;
                if (conf_.dequantize) {

                    val = ::sycl::min(::sycl::max(val, 0.f), 255.f);
                    val = (val - 2 * conf_.shift) / conf_.scale;
                }

                store_float_value(dst_md().data_type(), val, dst, dst_offset);

            } else {

                auto dst_offset = dst_data_offset(t, n, dir * conf_.dhc + c);
                auto src_offset
                        = src_data_offset(dir, conf_.n_iter - t - 1, n, c);
                auto src_v = load_float_value(
                        dst_md().data_type(), src, src_offset);
                if (conf_.dequantize) {
                    src_v = (src_v - conf_.shift) / conf_.scale;
                }
                store_float_value(dst_md().data_type(), src_v, dst, dst_offset);
            }
        }
    }

    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }

    inline dim_t dst_data_offset(dim_t t, dim_t n, dim_t c) const {
        return conf_.dst_md.off(t, n, c);
    }

    inline dim_t src_data_offset(dim_t d, dim_t t, dim_t n, dim_t c) const {
        return off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, conf_.n_layer - 1, d, t-1, n, c);
    }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_copy_res_layer_conf_t conf_;
};

struct ref_rnn_copy_res_iter {
    ref_rnn_copy_res_iter(const sycl_rnn_copy_res_iter_conf_t &conf,
            const xpu::sycl::in_memory_arg_t &src_base,
            const xpu::sycl::out_memory_arg_t &dst_base,
            const xpu::sycl::out_memory_arg_t &dst_c_base)
        : src_ {src_base}, dst_ {dst_base}, dst_c_ {dst_c_base}, conf_ {conf} {}
    void operator()(::sycl::nd_item<3> item) const {
        const int lay = item.get_global_id(0) / conf_.n_dir;
        const int dir = item.get_global_id(0) % conf_.n_dir;

        const int n = item.get_global_id(1);
        const int c = item.get_global_id(2);

        auto src = src_ptr();
        auto dst = dst_ptr();

        if (lay >= conf_.n_layer || dir >= conf_.n_dir || n >= conf_.batch
                || c >= conf_.dhc)
            return;

        if (dst && c < conf_.dhc) {
            auto src_offset = src_data_offset(lay, dir, n, c);
            auto dst_offset = dst_data_offset(lay, dir, n, c);

            auto src_v
                    = load_float_value(dst_md().data_type(), src, src_offset);
            ::sycl::ext::oneapi::experimental::printf(
                    "RES: \n src_offset: %lu, src_v: %f dst_offset: "
                    "%lu \n",
                    src_offset, src_v, dst_offset);

            if (conf_.dequantize) {
                src_v = (src_v - conf_.shift) / conf_.scale;
            }

            store_float_value(dst_md().data_type(), src_v, dst, dst_offset);
        }

        if (conf_.with_dst_iter_c) {
            auto dst_c = dst_c_ptr();

            auto dst_c_offset = dst_iter_c_data_offset(lay, dir, n, c);
            auto src_c_offset = src_iter_c_data_offset(lay, dir, n, c);

            auto src_c_v
                    = load_float_value(dst_md().data_type(), src, src_c_offset);
            store_float_value(
                    dst_md().data_type(), src_c_v, dst_c, dst_c_offset);
        }
    }
    inline dim_t dst_data_offset(dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return conf_.dst_md.off(lay, dir, n, c);
    }

    inline dim_t src_data_offset(dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return off_ws_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, lay, dir, conf_.n_iter - 2, n,
                c);
        //off_ws_state(n_layer, n_dir, n_iter, batch, states_ws_ld, lay, dir,
        //        n_iter - 1, b, s)
    }
    inline dim_t src_iter_c_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return conf_.dst_iter_c_md.off(lay, dir, n, c);
    }
    inline dim_t dst_iter_c_data_offset(
            dim_t lay, dim_t dir, dim_t n, dim_t c) const {
        return off_ws_c_state(conf_.n_layer, conf_.n_dir, conf_.n_iter,
                conf_.batch, conf_.states_ws_ld, lay, dir, -1, n, c);
    }

    const xpu::sycl::md_t &dst_md() const { return conf_.dst_md; }
    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *dst_c_ptr() const { return dst_c_.get_pointer(); }

    xpu::sycl::in_memory_arg_t src_;
    xpu::sycl::out_memory_arg_t dst_;
    xpu::sycl::out_memory_arg_t dst_c_;
    sycl_rnn_copy_res_iter_conf_t conf_;
};

struct ref_rnn_bias {
    ref_rnn_bias(const sycl_rnn_bias_conf_t &conf,
            const xpu::sycl::inout_memory_arg_t &src_base,
            const xpu::sycl::in_memory_arg_t &bias,
            const xpu::sycl::out_memory_arg_t &dst_base)
        : src_ {src_base}, bias_ {bias}, dst_ {dst_base}, conf_ {conf} {}
    void operator()(::sycl::nd_item<3> item) const {

        const int b = item.get_global_id(1);
        const int c = item.get_global_id(0);

        if (b >= conf_.batch || c >= conf_.dhc) return;

        auto src = src_ptr();
        auto bias = bias_ptr();
        auto dst = dst_ptr();

        auto src_offset = src_data_offset(b, c);
        auto bias_offset = bias_data_offset(b, c);
        auto dst_offset = dst_data_offset(b, c);

        auto src_val
                = load_float_value(conf_.dst_md.data_type(), src, src_offset);
        auto bias_val
                = load_float_value(conf_.dst_md.data_type(), bias, bias_offset);

        auto g = compute_gates(src_val, bias_val);
        ::sycl::ext::oneapi::experimental::printf(
                "BIAS: \n src_offset: %lu, src_v: %f bias_offset: %lu bias_v: "
                "%f dst_offset: %lu, G: %f  \n",
                src_offset, src_val, bias_offset, bias_val, dst_offset, g);

        store_float_value(conf_.dst_md.data_type(), g, dst, dst_offset);
        store_float_value(conf_.dst_md.data_type(), g, src, src_offset);
    }

    inline dim_t src_data_offset(int b, int c) const {
        return cell_scratch_mem(conf_.gates_ws_ld, conf_.dhc, b, 0, c);
    }

    inline dim_t bias_data_offset(int b, int c) const {
        return off_ker_bias(conf_.dhc, 0, c, 0);
    }

    inline dim_t dst_data_offset(int b, int c) const {
        return cell_ws_state(conf_.states_ws_ld, b, c);
    }

    float compute_gates(float in_val, float bias_val) const {
        switch (conf_.activation_kind) {
            case alg_kind::eltwise_relu:
                return (float)(math::relu_fwd(
                        (float)(in_val + bias_val), conf_.alpha));
            case alg_kind::eltwise_tanh:
                return (float)(math::tanh_fwd((float)(in_val + bias_val)));
            case alg_kind::eltwise_logistic:
                return (float)(math::logistic_fwd((float)(in_val + bias_val)));
            default: return 0;
        }
    }

    void *src_ptr() const { return src_.get_pointer(); }
    void *dst_ptr() const { return dst_.get_pointer(); }
    void *bias_ptr() const { return bias_.get_pointer(); }

    xpu::sycl::inout_memory_arg_t src_;
    xpu::sycl::in_memory_arg_t bias_;
    xpu::sycl::out_memory_arg_t dst_;
    sycl_rnn_bias_conf_t conf_;
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
