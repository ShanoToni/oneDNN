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

// General architecture
//
// for diff states, we have n_states + 1 as we have n_states diff
// to propagate to the previous iteration and 1 states to propagate
// to the previous layer
// index 0 is dh for cell(t-1, l) to consume
// index 1 is dc for cell(t-1, l) to consume
// index 2 is dh for cell(t, l-1) to consume
// this indexing enables to have the same indexing for states in elemwise
// function
// only the cell execution function should be impacted

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"
#include "common/primitive.hpp"
#include "common/primitive_desc.hpp"
#include "gpu/generic/sycl/sycl_gpu_primitive.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_traits.hpp"
#include "common/gemm_utils.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"
#include "gpu/gpu_primitive.hpp"
#include "gpu/intel/gemm/gpu_gemm.hpp"

#include <iostream>
#include <memory>
#include "gpu/generic/sycl/rnn/rnn_kernels.hpp"
#include "gpu/intel/utils.hpp"
#include "gpu/nvidia/engine.hpp"
#include "gpu/nvidia/stream.hpp"
#include "xpu/sycl/memory_storage_helper.hpp"

#define DPRINT(fmt, ...) \
    do { \
        if (get_verbose_dev_mode(verbose_t::debuginfo) >= 2) { \
            printf(fmt, __VA_ARGS__); \
            fflush(nullptr); \
        } \
    } while (0)

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

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

using namespace dnnl::impl::utils;
using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace dnnl::impl::math;
using namespace prop_kind;
using namespace alg_kind;
using namespace rnn_utils;
using namespace dnnl::impl::memory_tracking::names;

#define AOC array_offset_calculator

static status_t init_ocl_conf(const rnn_pd_t *rnn_pd,
        const rnn_utils::conf_t &rnn, const memory_desc_wrapper &src_layer_d,
        const memory_desc_wrapper &src_iter_d,
        const memory_desc_wrapper &src_iter_c_d,
        const memory_desc_wrapper &weights_layer_d,
        const memory_desc_wrapper &weights_iter_d,
        const memory_desc_wrapper &bias_d,
        const memory_desc_wrapper &dst_layer_d,
        const memory_desc_wrapper &dst_iter_d,
        const memory_desc_wrapper &dst_iter_c_d,
        const memory_desc_wrapper &diff_src_layer_d,
        const memory_desc_wrapper &diff_src_iter_d,
        const memory_desc_wrapper &diff_src_iter_c_d,
        const memory_desc_wrapper &diff_weights_layer_d,
        const memory_desc_wrapper &diff_weights_iter_d,
        const memory_desc_wrapper &diff_bias_d,
        const memory_desc_wrapper &diff_dst_layer_d,
        const memory_desc_wrapper &diff_dst_iter_d,
        const memory_desc_wrapper &diff_dst_iter_c_d,
        const memory_desc_wrapper &ws_d, rnn_offsets_t &off) {

    using namespace rnn_utils;

    auto src_dt = rnn.src_data_type;
    auto src_c_dt = src_iter_c_d.data_type();
    auto wei_dt = weights_layer_d.data_type();
    auto bia_dt = rnn.bias_data_type;
    auto acc_dt = rnn.acc_data_type;
    auto aux_dt = rnn.aux_data_type;
    auto diff_dt = rnn.diff_data_type;
    auto input_dt = rnn.input_data_type;
    auto output_dt = rnn.output_data_type;
    auto dst_dt = rnn.dst_data_type;
    auto dst_c_dt = dst_iter_c_d.data_type();

    auto is_fwd = rnn.is_fwd;

    auto with_bias = rnn_pd->with_bias();
    auto with_src_iter = rnn_pd->with_src_iter();
    auto with_src_iter_c = rnn_pd->with_src_iter_c();
    auto with_dst_iter = rnn_pd->with_dst_iter();
    auto with_dst_iter_c = rnn_pd->with_dst_iter_c();
    auto copy_bias = rnn.copy_bias;
    auto is_int8 = rnn.is_int8;
    auto is_training = rnn.is_training;
    auto recompute_gates = rnn.recompute_gates;
    auto copy_src_layer = rnn.copy_src_layer;
    auto copy_diff_dst_layer = rnn.copy_diff_dst_layer;
    auto copy_diff_src_layer = rnn.copy_diff_src_layer;

    if (!rnn.is_fwd) {
        if (!utils::everyone_is(diff_dt, diff_src_layer_d.data_type(),
                    diff_dst_layer_d.data_type()))
            return status::unimplemented;
        if (!utils::one_of(
                    diff_src_iter_d.data_type(), diff_dt, data_type::undef)
                || !utils::one_of(diff_src_iter_c_d.data_type(), diff_dt,
                        data_type::undef)
                || !utils::one_of(
                        diff_dst_iter_d.data_type(), diff_dt, data_type::undef)
                || !utils::one_of(diff_dst_iter_c_d.data_type(), diff_dt,
                        data_type::undef))
            return status::unimplemented;
    }

    off.src_layer = gpu::intel::get_outer_strides(src_layer_d);

    off.src_iter = gpu::intel::get_outer_strides(src_iter_d);

    if (with_src_iter_c) {
        off.src_iter_c = gpu::intel::get_outer_strides(src_iter_c_d);
    }
    off.weights_layer = gpu::intel::get_outer_strides(weights_layer_d);

    off.weights_layer_comp_off
            = weights_layer_d.dims()[0] * weights_layer_d.strides()[0];
    off.weights_iter = gpu::intel::get_outer_strides(weights_iter_d);

    off.weights_iter_comp_off
            = weights_iter_d.dims()[0] * weights_iter_d.strides()[0];
    off.bias = gpu::intel::get_outer_strides(bias_d);

    off.dst_layer = gpu::intel::get_outer_strides(dst_layer_d);

    off.dst_iter = gpu::intel::get_outer_strides(dst_iter_d);

    if (with_dst_iter_c) {
        off.dst_iter_c = gpu::intel::get_outer_strides(dst_iter_c_d);
    }

    if (!is_fwd) {
        off.diff_src_layer = gpu::intel::get_outer_strides(diff_src_layer_d);

        off.diff_src_iter = gpu::intel::get_outer_strides(diff_src_iter_d);

        if (with_src_iter_c) {
            off.diff_src_iter_c
                    = gpu::intel::get_outer_strides(diff_src_iter_c_d);
        }
        off.diff_weights_layer
                = gpu::intel::get_outer_strides(diff_weights_layer_d);

        off.diff_weights_iter
                = gpu::intel::get_outer_strides(diff_weights_iter_d);

        off.diff_bias = gpu::intel::get_outer_strides(diff_bias_d);

        off.diff_dst_layer = gpu::intel::get_outer_strides(diff_dst_layer_d);

        off.diff_dst_iter = gpu::intel::get_outer_strides(diff_dst_iter_d);

        if (with_dst_iter_c) {
            off.diff_dst_iter_c
                    = gpu::intel::get_outer_strides(diff_dst_iter_c_d);
        }
    }

    return status::success;
}

template <prop_kind_t aprop>
inline status_t init_ocl_conf(const rnn_utils::conf_t &rnn,
        const rnn_pd_t *rnn_pd, rnn_offsets_t &off) {

    const memory_desc_wrapper fakedesc = rnn_pd->src_md(0);
    return init_ocl_conf(rnn_pd, rnn, rnn_pd->src_md(0), rnn_pd->src_md(1),
            rnn_pd->src_md(2), rnn_pd->weights_md(0), rnn_pd->weights_md(1),
            rnn_pd->weights_md(2), rnn_pd->dst_md(0), rnn_pd->dst_md(1),
            rnn_pd->dst_md(2), fakedesc, fakedesc, fakedesc, fakedesc, fakedesc,
            fakedesc, fakedesc, fakedesc, fakedesc, rnn_pd->workspace_md(0),
            off);
}

template <>
status_t _ref_rnn_common_t<prop_kind::forward>::pd_t::set_default_params() {
    using namespace format_tag;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    return status::success;
}

template <>
status_t _ref_rnn_common_t<prop_kind::backward>::pd_t::set_default_params() {
    using namespace format_tag;
    // TODO
    int arch_ld = 64;
    if (src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_layer_md_, tnc));
    if (weights_layer_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_layer_md_, ldgoi));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, weights_layer_md_, ldgoi));
    }
    if (dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_layer_md_, tnc));

    if (weights_iter_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(weights_iter_md_, ldgoi));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, weights_iter_md_, ldgoi));
    }

    if (diff_src_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_src_layer_md_, tnc));
    if (diff_weights_layer_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_layer_md_, ldigo));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, diff_weights_layer_md_, ldigo));
    }
    if (diff_weights_iter_md_.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_weights_iter_md_, ldigo));
        if (!rnn_conf.is_int8)
            CHECK(rnn_utils::set_good_strides(
                    arch_ld, diff_weights_iter_md_, ldigo));
    }
    if (diff_dst_layer_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(diff_dst_layer_md_, tnc));

    // Optional parameters
    if ((!types::is_zero_md(&src_iter_md_))
            && (src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_md_, ldnc));
    if ((!types::is_zero_md(&src_iter_c_md_))
            && (src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&bias_md_))
            && (bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(bias_md_, ldgo));
    if ((!types::is_zero_md(&dst_iter_md_))
            && (dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&dst_iter_c_md_))
            && (dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(dst_iter_c_md_, ldnc));

    if ((!types::is_zero_md(&diff_src_iter_md_))
            && (diff_src_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_src_iter_c_md_))
            && (diff_src_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_src_iter_c_md_, ldnc));
    if ((!types::is_zero_md(&diff_bias_md_))
            && (diff_bias_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_bias_md_, ldgo));
    if ((!types::is_zero_md(&diff_dst_iter_md_))
            && (diff_dst_iter_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_md_, ldnc));
    if ((!types::is_zero_md(&diff_dst_iter_c_md_))
            && (diff_dst_iter_c_md_.format_kind == format_kind::any))
        CHECK(memory_desc_init_by_tag(diff_dst_iter_c_md_, ldnc));

    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::pd_t::init(impl::engine_t *engine) {
    using namespace prop_kind;
    using namespace utils;
    using namespace rnn_utils;
    using namespace format_tag;

    assert(engine->kind() == engine_kind::gpu);

    const alg_kind_t cell_kind = this->desc()->cell_kind;

    data_type_t src_layer_dt = this->desc()->src_layer_desc.data_type;
    data_type_t weights_iter_dt = this->desc()->weights_iter_desc.data_type;
    data_type_t weights_layer_dt = this->desc()->weights_layer_desc.data_type;
    data_type_t bias_dt = this->desc()->bias_desc.data_type;

    bool src_is_u8 = src_layer_dt == data_type::u8;
    bool src_is_f16 = src_layer_dt == data_type::f16;
    if (src_is_u8)
        acc_data_t = data_type::s32;
    else if (src_is_f16 && aprop == prop_kind::forward_inference)
        acc_data_t = data_type::f16;
    else
        acc_data_t = data_type::f32;

    src_type = src_layer_dt;
    weights_type = weights_layer_dt;

    VDISPATCH_RNN(
            one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                    alg_kind::lbr_gru, alg_kind::vanilla_gru),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_RNN(!this->is_lstm_peephole(), "is_lstm_peephole");
    VDISPATCH_RNN(!this->is_lstm_projection(), "is_lstm_projection");
    VDISPATCH_RNN(IMPLICATION(aprop == prop_kind::forward,
                          one_of(this->desc()->prop_kind, forward_training,
                                  forward_inference)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(IMPLICATION(aprop == backward,
                          one_of(this->desc()->prop_kind, backward)),
            VERBOSE_BAD_PROPKIND);
    VDISPATCH_RNN(
            IMPLICATION(src_type == data_type::bf16, bias_dt == data_type::f32),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(((aprop == prop_kind::forward && src_layer_dt == data_type::u8
                           && weights_layer_dt == data_type::s8
                           && cell_kind == alg_kind::vanilla_lstm)
                          || (aprop == prop_kind::forward
                                  && one_of(src_layer_dt, data_type::f16,
                                          data_type::f32, data_type::bf16)
                                  && weights_layer_dt == src_layer_dt)
                          || (aprop == prop_kind::backward
                                  && one_of(weights_layer_dt, data_type::f32,
                                          data_type::f16, data_type::bf16)
                                  && weights_layer_dt == src_layer_dt)),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN(weights_iter_dt == weights_layer_dt, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_RNN_SC(this->set_default_params(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_RNN(this->with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_RNN(IMPLICATION(src_layer_dt == data_type::u8,
                          this->desc()->prop_kind == forward_inference),
            VERBOSE_UNSUPPORTED_DT_CFG);

    init_rnn_conf(rnn_conf, *this->desc(), this->src_md(0), this->src_md(1),
            this->weights_md(0), this->weights_md(1), this->dst_md(0),
            this->dst_md(1), this->diff_dst_md(0), this->desc()->bias_desc,
            acc_data_t);

    if (rnn_conf.is_int8) {
        auto has_trivial_strides = [](const memory_desc_wrapper &md) {
            return md.is_dense(true);
        };
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_layer_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_iter_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->src_iter_c_md_),
                status::unimplemented, VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_layer_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_iter_md_), status::unimplemented,
                VERBOSE_NONTRIVIAL_STRIDE);
        VCONDCHECK(primitive, create, dispatch, rnn,
                has_trivial_strides(this->dst_iter_c_md_),
                status::unimplemented, VERBOSE_NONTRIVIAL_STRIDE);
    }

    // Check that only supported attr have been passed.
    primitive_attr_t::skip_mask_t attr_mask
            = primitive_attr_t::skip_mask_t::rnn_tparams;
    if (weights_layer_dt == data_type::s8) {
        attr_mask = attr_mask | primitive_attr_t::skip_mask_t::rnn_data_qparams
                | primitive_attr_t::skip_mask_t::rnn_weights_qparams
                | primitive_attr_t::skip_mask_t::fpmath_mode;
    }
    VDISPATCH_RNN(this->attr()->has_default_values(attr_mask),
            VERBOSE_UNSUPPORTED_ATTR);

    // TODO: implement something like check layout consistency
    switch (aprop) {
        case (prop_kind::forward): break;
        case (prop_kind::backward):
            VDISPATCH_RNN(utils::one_of(this->desc()->prop_kind, backward),
                    VERBOSE_BAD_PROPKIND);
            break;
        default: return status::unimplemented;
    }

    // Set weights descriptors to desired format
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_layer_md_, rnn_conf),
            "unsupported weights layer memory descriptor");
    VDISPATCH_RNN_SC(set_weights_desc(this->weights_iter_md_, rnn_conf),
            "unsupported weights iter memory descriptor");

    // Check dimensions consistency
    int ls_multiplier
            = (this->direction() == dnnl_bidirectional_concat) ? 2 : 1;

    VDISPATCH_RNN((ls_multiplier * this->DHC() == this->DLC()),
            VERBOSE_INCONSISTENT_DIM, "DHC", (int)this->DHC(), "DLC",
            (int)this->DLC());
    VDISPATCH_RNN(
            (ls_multiplier * this->SLC()) == this->DLC() || (this->L() == 1),
            VERBOSE_INCONSISTENT_DIM, "SLC", (int)this->SLC(), "DLC",
            (int)this->DLC());
    VDISPATCH_RNN((this->SIC() == this->DHC() || (this->T() == 1)),
            VERBOSE_INCONSISTENT_DIM, "SIC", (int)this->SIC(), "DHC",
            (int)this->DHC());

    set_rnn_conf(rnn_conf, *this->desc(), this->src_md(0), this->diff_src_md(0),
            this->diff_dst_md(0), this->weights_md(0), this->weights_md(1),
            this->diff_weights_md(0), this->diff_weights_md(1));

    dim_t workspace_size = get_workspace_size(rnn_conf);

    // initialize the workspace_pd if needed
    if (rnn_conf.use_workspace) {
        dims_t ws_dims = {workspace_size};
        VDISPATCH_RNN_SC(memory_desc_init_by_tag(
                                 this->ws_md_, 1, ws_dims, data_type::u8, x),
                "memory_desc_init_by_tag()");
    }

    VDISPATCH_RNN_SC(init_ocl_conf<aprop>(rnn_conf, this, this->off),
            "init_ocl_conf<>()");

    // IMPORTANT SYCL STUFF
    copy_init_layer_conf_ = sycl_rnn_copy_init_layer_conf_t();
    copy_init_layer_conf_.batch = rnn_conf.mb;
    copy_init_layer_conf_.slc = rnn_conf.slc;
    copy_init_layer_conf_.n_iter = rnn_conf.n_iter;
    copy_init_layer_conf_.n_layer = rnn_conf.n_layer;
    copy_init_layer_conf_.n_dir = rnn_conf.n_dir;
    copy_init_layer_conf_.n_states = rnn_conf.n_states;
    copy_init_layer_conf_.states_ws_ld = rnn_conf.states_ws_ld;
    copy_init_layer_conf_.lr = !one_of(rnn_conf.exec_dir, r2l, r2l);
    copy_init_layer_conf_.rl = !one_of(rnn_conf.exec_dir, l2r, l2r);
    copy_init_layer_conf_.src_md = xpu::sycl::md_t(this->src_md(0));

    copy_init_iter_conf_ = sycl_rnn_copy_init_iter_conf_t();
    copy_init_iter_conf_.batch = rnn_conf.mb;
    copy_init_iter_conf_.sic = rnn_conf.sic;
    copy_init_iter_conf_.dhc = rnn_conf.dhc;
    copy_init_iter_conf_.n_iter = rnn_conf.n_iter;
    copy_init_iter_conf_.n_layer = rnn_conf.n_layer;
    copy_init_iter_conf_.n_dir = rnn_conf.n_dir;
    copy_init_iter_conf_.n_states = rnn_conf.n_states;
    copy_init_iter_conf_.states_ws_ld = rnn_conf.states_ws_ld;
    copy_init_iter_conf_.quantize = this->with_src_iter()
            && this->src_md(1)->data_type == data_type::f32 && rnn_conf.is_int8;
    copy_init_iter_conf_.with_iter_c = this->with_src_iter_c();
    copy_init_iter_conf_.src_iter_md = xpu::sycl::md_t(this->src_md(1));
    if (this->with_src_iter_c()) {
        copy_init_iter_conf_.src_iter_c_md = xpu::sycl::md_t(this->src_md(2));
    }
    copy_init_iter_conf_.scale = (this->attr()->rnn_data_qparams_.scale_);
    copy_init_iter_conf_.shift = (this->attr()->rnn_data_qparams_.shift_);

    copy_res_layer_conf_ = sycl_rnn_copy_res_layer_conf_t();
    copy_res_layer_conf_.batch = rnn_conf.mb;
    copy_res_layer_conf_.slc = rnn_conf.sic;
    copy_res_layer_conf_.dhc = rnn_conf.dhc;
    copy_res_layer_conf_.n_iter = rnn_conf.n_iter;
    copy_res_layer_conf_.n_layer = rnn_conf.n_layer;
    copy_res_layer_conf_.n_dir = rnn_conf.n_dir;
    copy_res_layer_conf_.n_states = rnn_conf.n_states;
    copy_res_layer_conf_.states_ws_ld = rnn_conf.states_ws_ld;
    copy_res_layer_conf_.dst_md = xpu::sycl::md_t(this->dst_md(0));
    copy_res_layer_conf_.lr = !one_of(rnn_conf.exec_dir, r2l, r2l);
    copy_res_layer_conf_.rl = !one_of(rnn_conf.exec_dir, l2r, l2r);
    copy_res_layer_conf_.dequantize
            = this->dst_md(0)->data_type == data_type::f32 && rnn_conf.is_int8;
    copy_res_layer_conf_.direction = this->direction();

    copy_res_iter_conf_ = sycl_rnn_copy_res_iter_conf_t();
    copy_res_iter_conf_.dst_md = xpu::sycl::md_t(this->dst_md(1));
    if (this->with_dst_iter_c()) {
        copy_res_iter_conf_.dst_iter_c_md = xpu::sycl::md_t(this->dst_md(2));
    }
    copy_res_iter_conf_.dhc = rnn_conf.dhc;
    copy_res_iter_conf_.shift = (this->attr()->rnn_data_qparams_.shift_);
    copy_res_iter_conf_.scale = (this->attr()->rnn_data_qparams_.scale_);
    copy_res_iter_conf_.n_dir = rnn_conf.n_dir;
    copy_res_iter_conf_.dequantize
            = this->dst_md(1)->data_type == data_type::f32 && rnn_conf.is_int8;
    copy_res_iter_conf_.with_dst_iter_c = this->with_dst_iter_c();
    copy_res_iter_conf_.batch = rnn_conf.mb;
    copy_res_iter_conf_.n_iter = rnn_conf.n_iter;
    copy_res_iter_conf_.n_layer = rnn_conf.n_layer;
    copy_res_iter_conf_.states_ws_ld = rnn_conf.states_ws_ld;

    dim_t batch = rnn_conf.mb;
    dim_t n_gates = rnn_conf.n_gates;
    dim_t slc = rnn_conf.slc;
    dim_t sic = rnn_conf.sic;
    dim_t dhc = rnn_conf.dhc;

    sycl_rnn_bias_conf_t_ = sycl_rnn_bias_conf_t();
    sycl_rnn_bias_conf_t_.dst_md = xpu::sycl::md_t(this->dst_md(1));
    sycl_rnn_bias_conf_t_.batch = rnn_conf.mb;
    sycl_rnn_bias_conf_t_.dhc = rnn_conf.dhc;
    sycl_rnn_bias_conf_t_.gates_ws_ld = rnn_conf.gates_ws_ld;
    sycl_rnn_bias_conf_t_.states_ws_ld = rnn_conf.states_ws_ld;
    sycl_rnn_bias_conf_t_.activation_kind = this->activation_kind();
    sycl_rnn_bias_conf_t_.alpha = this->desc()->alpha;

    auto fpmath_mode = this->attr()->fpmath_.mode_;

    // The inputs of create_gemm_pd describe a gemm in column major.
    // Below, we have to transpose the a and b descriptor to describe
    // the GEMM as a row major problem.
    auto create_gemm_pd =
            [&](std::shared_ptr<primitive_desc_t> &gemm_pd, dim_t m, dim_t n,
                    dim_t k, strides_t<2> a_strides, strides_t<2> b_strides,
                    strides_t<2> c_strides, data_type_t a_dt, data_type_t b_dt,
                    data_type_t c_dt, float beta) -> status_t {
        memory_desc_t a_md, b_md, c_md;
        // ANTON
        dims_t a_dims = {n, k}, b_dims = {k, m}, c_dims = {n, m};

        dims_t b_strides_md = {b_strides[0], b_strides[1]};
        CHECK(memory_desc_init_by_strides(b_md, 2, b_dims, b_dt, b_strides_md));
        dims_t a_strides_md = {a_strides[0], a_strides[1]};
        CHECK(memory_desc_init_by_strides(a_md, 2, a_dims, a_dt, a_strides_md));
        dims_t c_strides_md = {c_strides[0], c_strides[1]};
        CHECK(memory_desc_init_by_strides(c_md, 2, c_dims, c_dt, c_strides_md));

        primitive_attr_t attr;
        CHECK(attr.post_ops_.append_sum(beta));
        CHECK(attr.set_fpmath_mode(fpmath_mode));
        attr.deterministic_ = this->attr()->deterministic_;
        CHECK(dnnl::impl::create_gemm_pd(gemm_pd, engine, &a_md, &b_md, &c_md,
                &glob_zero_md, c_dt, &attr));
        return status::success;
    };

    dim_t layer_merged_size
            = rnn_conf.merge_gemm_layer ? batch * rnn_conf.n_iter : batch;
    dim_t iter_merged_size
            = rnn_conf.merge_gemm_iter ? batch * rnn_conf.n_iter : batch;

    float gemm_iter_fwd_beta = this->is_lbr() ? 0.0f : 1.0f;
    float gemm_iter_bwd_beta = this->is_lbr() ? 1.0f : 0.0f;
    if (aprop == prop_kind::forward || rnn_conf.recompute_gates) {
        if (!rnn_conf.cell_fusion.gemm_layer) {
            //VDISPATCH_RNN_SC(
            //        create_gemm_pd(gemm_layer_fwd_pd_, slc, layer_merged_size,
            //                n_gates * dhc, {rnn_conf.states_ws_ld, 1},
            //                {off.weights_layer[4], off.weights_layer[1]},
            //                {slc, 1}, weights_type,
            //                src_type, rnn_conf.acc_data_type, 0.0),
            //        "create_gemm_pd(gemm_layer_fwd_pd_)");

            // ANTON
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_layer_fwd_pd_, n_gates * dhc,
                            layer_merged_size, slc, {rnn_conf.states_ws_ld, 1},
                            {off.weights_layer[2], off.weights_layer[4]},
                            {rnn_conf.scratch_gates_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 0.0),
                    "create_gemm_pd(gemm_layer_fwd_pd_)");
            if (!rnn_conf.copy_src_layer) {
                if (off.src_layer[1] != rnn_conf.states_ws_ld)
                    VDISPATCH_RNN_SC(
                            create_gemm_pd(gemm_layer_fwd_src_pd_,
                                    n_gates * dhc, layer_merged_size, slc,
                                    {off.src_layer[1], off.src_layer[2]},
                                    {off.weights_layer[2],
                                            off.weights_layer[4]},
                                    {rnn_conf.scratch_gates_ld, 1},
                                    weights_type, src_type,
                                    rnn_conf.acc_data_type, 0.0),
                            "create_gemm_pd(gemm_layer_fwd_src_pd_)");
                else
                    gemm_layer_fwd_src_pd_ = gemm_layer_fwd_pd_;
            }
        }
        if (!rnn_conf.cell_fusion.gemm_iter) {
            if (rnn_conf.is_vanilla_gru) {
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_pd_, (n_gates - 1) * dhc,
                                batch, sic, {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.scratch_gates_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_pd_)");
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_2_pd_, dhc, batch, sic,
                                {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.scratch_gates_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_2_pd_)");
            } else {
                VDISPATCH_RNN_SC(
                        create_gemm_pd(gemm_iter_fwd_pd_, n_gates * dhc, batch,
                                sic, {rnn_conf.states_ws_ld, 1},
                                {off.weights_iter[2], off.weights_iter[4]},
                                {rnn_conf.gates_ws_ld, 1}, weights_type,
                                src_type, rnn_conf.acc_data_type,
                                gemm_iter_fwd_beta),
                        "create_gemm_pd(gemm_iter_fwd_pd_)");
            }
        }
    }

    if (aprop == prop_kind::backward) {
        if (rnn_conf.is_vanilla_gru) {
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_pd_, sic, batch,
                            (n_gates - 1) * dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 1.0f),
                    "create_gemm_pd(gemm_iter_bwd_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_2_pd_, sic, batch, dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type, 0.0f),
                    "create_gemm_pd(gemm_iter_bwd_2_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_pd_, (n_gates - 1) * dhc,
                            sic, iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_2_pd_, dhc, sic,
                            iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_2_pd_)");
        } else {
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_iter_bwd_pd_, sic, batch, n_gates * dhc,
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.weights_iter[4], off.weights_iter[2]},
                            {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                            src_type, rnn_conf.acc_data_type,
                            gemm_iter_bwd_beta),
                    "create_gemm_pd(gemm_iter_bwd_pd_)");
            VDISPATCH_RNN_SC(
                    create_gemm_pd(gemm_diff_wei_iter_pd_, n_gates * dhc, sic,
                            iter_merged_size, {1, rnn_conf.states_ws_ld},
                            {rnn_conf.scratch_diff_gates_ld, 1},
                            {off.diff_weights_iter[2],
                                    off.diff_weights_iter[4]},
                            weights_type, src_type, rnn_conf.acc_data_type,
                            1.0f),
                    "create_gemm_pd(gemm_diff_wei_iter_pd_)");
        }
        VDISPATCH_RNN_SC(
                create_gemm_pd(gemm_layer_bwd_pd_, slc, layer_merged_size,
                        n_gates * dhc, {rnn_conf.scratch_diff_gates_ld, 1},
                        {off.weights_layer[4], off.weights_layer[2]},
                        {rnn_conf.scratch_diff_states_ld, 1}, weights_type,
                        src_type, rnn_conf.acc_data_type, 0.0f),
                "create_gemm_pd(gemm_layer_bwd_pd_)");
        VDISPATCH_RNN_SC(
                create_gemm_pd(gemm_diff_wei_layer_pd_, n_gates * dhc, slc,
                        layer_merged_size, {1, rnn_conf.states_ws_ld},
                        {rnn_conf.scratch_diff_gates_ld, 1},
                        {off.diff_weights_layer[2], off.diff_weights_layer[4]},
                        weights_type, src_type, rnn_conf.acc_data_type, 1.0f),
                "create_gemm_pd(gemm_diff_wei_layer_pd_)");
        if (!rnn_conf.copy_src_layer) {
            if (off.src_layer[1] != rnn_conf.states_ws_ld)
                VDISPATCH_RNN_SC(create_gemm_pd(gemm_diff_wei_layer_src_pd_,
                                         n_gates * dhc, slc, layer_merged_size,
                                         {off.src_layer[2], off.src_layer[1]},
                                         {rnn_conf.scratch_diff_gates_ld, 1},
                                         {off.diff_weights_layer[2],
                                                 off.diff_weights_layer[4]},
                                         weights_type, src_type,
                                         rnn_conf.acc_data_type, 1.0f),
                        "create_gemm_pd(gemm_diff_wei_layer_src_pd_)");
            else
                gemm_diff_wei_layer_src_pd_ = gemm_diff_wei_layer_pd_;
        }
    }

    // Setup binary for bias
    if (this->cell_kind() == dnnl_vanilla_rnn) {
        memory_desc_t dst_md;
        dims_t dst_dims = {batch, 1, 1, n_gates * dhc};
        dims_t dst_str = {n_gates * dhc, n_gates * dhc, n_gates * dhc, 1};
        CHECK(memory_desc_init_by_strides(
                dst_md, 4, dst_dims, dnnl_f32, dst_str));

        primitive_attr_t binary_attr;
        auto binary_desc = binary_desc_t();
        binary_desc.primitive_kind = primitive_kind::binary;
        binary_desc.alg_kind = alg_kind::binary_add;
        binary_desc.src_desc[0] = dst_md; // src value
        binary_desc.src_desc[1] = this->bias_md_; // add value
        binary_desc.dst_desc = dst_md;

        primitive_desc_iterator_t it(
                engine, (op_desc_t *)&binary_desc, &binary_attr, nullptr);
        while (++it != it.end()) {
            bias_binary_pd_ = *it;
            if (bias_binary_pd_) { break; }
        }
        if (!bias_binary_pd_) return status::unimplemented;
    } else {
        return status::unimplemented;
    }

    // Setup eltwise for vanilla activation
    if (this->cell_kind() == dnnl_vanilla_rnn) {
        primitive_attr_t eltwise_attr;
        auto eltwise_desc = eltwise_desc_t();
        CHECK(eltwise_desc_init(&eltwise_desc, prop_kind_t::dnnl_forward,
                this->activation_kind(), this->dst_md(), this->dst_md(),
                nullptr, nullptr, 1, 0));
        if (!eltwise_attr.is_initialized()) return status::out_of_memory;
        primitive_desc_iterator_t it(engine,
                reinterpret_cast<op_desc_t *>(&eltwise_desc), &eltwise_attr,
                nullptr);
        if (!it.is_initialized()) return status::invalid_arguments;
        vanilla_cell_act_pd_ = *(++it);

        if (!vanilla_cell_act_pd_) return status::unimplemented;
    } else {
        return status::unimplemented;
    }

    init_scratchpad(rnn_conf.use_workspace ? 0 : workspace_size);
    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::init(impl::engine_t *engine) {
    using namespace rnn_utils;

    switch (pd()->cell_kind()) {
        // case dnnl_vanilla_lstm:
        //     break;
        case dnnl_vanilla_rnn:
            cell_func = &class_name::cell_execution;
            //elemwise_common = &class_name::rnn_elemwise;
            //bias_common = &class_name::rnn_bias;
            break;
        default: break;
    }

    grid_computation = &class_name::linear_execution;

    const conf_t &rnn = pd()->rnn_conf;
    rnn_utils::set_workspace_offsets(rnn, ws_gates_offset_, ws_states_offset_,
            ws_c_states_offset_, ws_grid_comp_offset_, ws_bias_offset_);

    // IMPORTANT SYCL STUFF
    const auto copy_layer_kid
            = ::sycl::get_kernel_id<ref_rnn_copy_init_layer_t>();
    const auto copy_iter_kid
            = ::sycl::get_kernel_id<ref_rnn_copy_init_iter_t>();
    const auto copy_res_layer_kid
            = ::sycl::get_kernel_id<ref_rnn_copy_res_layer_t>();
    const auto copy_res_iter_kid
            = ::sycl::get_kernel_id<ref_rnn_copy_res_iter>();
    const auto bias_kid = ::sycl::get_kernel_id<ref_rnn_bias>();

    this->create_kernel(engine, copy_layer_kid, &copy_init_layer_kernel_);
    this->create_kernel(engine, copy_iter_kid, &copy_init_iter_kernel_);
    this->create_kernel(engine, copy_res_layer_kid, &copy_res_layer_kernel_);
    this->create_kernel(engine, copy_res_iter_kid, &copy_res_iter_kernel_);
    this->create_kernel(engine, bias_kid, &bias_kernel_);

    bool gemm_ok = true;
    auto create_nested_gemm =
            [&](const std::shared_ptr<primitive_desc_t> &prim_desc,
                    std::shared_ptr<impl::primitive_t> &prim) {
                std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t>
                        pair;
                bool gemm_ok = prim_desc->create_primitive_nested(pair, engine)
                        == status::success;
                prim = pair.first;
                return gemm_ok;
            };

    gemm_ok = gemm_ok
            && create_nested_gemm(pd()->gemm_layer_fwd_pd_, gemm_layer_fwd_);
    if (pd()->gemm_layer_fwd_src_pd_) {
        gemm_ok = gemm_ok
                && create_nested_gemm(
                        pd()->gemm_layer_fwd_src_pd_, gemm_layer_fwd_src_);
    }
    gemm_ok = gemm_ok
            && create_nested_gemm(pd()->gemm_iter_fwd_pd_, gemm_iter_fwd_);

    switch (aprop) {
        case prop_kind::forward:
            if (rnn.is_vanilla_gru) {
                gemm_ok = gemm_ok
                        && create_nested_gemm(
                                pd()->gemm_iter_fwd_2_pd_, gemm_iter_fwd_2_);
            }
            break;
        case prop_kind::backward:
            gemm_ok = gemm_ok
                    && create_nested_gemm(
                            pd()->gemm_layer_bwd_pd_, gemm_layer_bwd_);
            gemm_ok = gemm_ok
                    && create_nested_gemm(
                            pd()->gemm_iter_bwd_pd_, gemm_iter_bwd_);
            gemm_ok = gemm_ok
                    && create_nested_gemm(pd()->gemm_diff_wei_layer_pd_,
                            gemm_diff_wei_layer_);

            if (pd()->gemm_diff_wei_layer_src_pd_) {
                gemm_ok = gemm_ok
                        && create_nested_gemm(pd()->gemm_diff_wei_layer_src_pd_,
                                gemm_diff_wei_layer_src_);
            }
            gemm_ok = gemm_ok
                    && create_nested_gemm(
                            pd()->gemm_diff_wei_iter_pd_, gemm_diff_wei_iter_);

            if (rnn.is_vanilla_gru) {
                if (pd()->gemm_iter_bwd_2_pd_)
                    gemm_ok = gemm_ok
                            && create_nested_gemm(pd()->gemm_iter_bwd_2_pd_,
                                    gemm_iter_bwd_2_);
            }
            if (pd()->gemm_diff_wei_iter_2_pd_) {
                gemm_ok = gemm_ok
                        && create_nested_gemm(pd()->gemm_diff_wei_iter_2_pd_,
                                gemm_diff_wei_iter_2_);
            }
            break;
        default: assert(!"unknown prop_kind"); return status::invalid_arguments;
    }

    if (!gemm_ok) return status::runtime_error;

    std::pair<std::shared_ptr<impl::primitive_t>, cache_state_t> pair;
    if (pd()->cell_kind() == dnnl_vanilla_rnn) {
        bool binary_ok
                = pd()->bias_binary_pd_->create_primitive_nested(pair, engine)
                == status::success;
        bias_binary_ = pair.first;
        bool activation_ok
                = pd()->vanilla_cell_act_pd_->create_primitive_nested(
                          pair, engine)
                == status::success;
        vanilla_cell_act_ = pair.first;
        if (!binary_ok || !activation_ok) { return status::unimplemented; }
    } else {
        return status::unimplemented;
    }
    return status::success;
} // namespace sycl

template <prop_kind_t aprop>
gemm_sig((_ref_rnn_common_t<aprop>::gemm_primitive)) {
    // We flip A and B here since the GEMM API is row major but the
    // RNN code describes GEMM in column major fashion
    intel::gemm_exec_args_t gemm_args;
    gemm_args.a = b.get();
    gemm_args.b = a.get();
    gemm_args.c = c.get();

    auto gemm_ctx = intel::gemm_exec_ctx_t(ctx, gemm_args);

    std::unique_ptr<nested_scratchpad_t> ns;
    const auto init_gemm_nested_scratchpad
            = [&](const std::shared_ptr<impl::primitive_t> &gemm, int key) {
                  ns = utils::make_unique<nested_scratchpad_t>(ctx, key, gemm);
                  gemm_ctx.set_scratchpad_grantor(ns->grantor());
              };

    switch (gemm_kind) {
        case gemm_iter_fwd:
            init_gemm_nested_scratchpad(
                    gemm_iter_fwd_, rnn_utils::scratch_t::key_gemm_iter_fwd);
            CHECK(intel::gpu_gemm(gemm_iter_fwd_)->execute(gemm_ctx));
            break;
        case gemm_iter_fwd_2:
            init_gemm_nested_scratchpad(gemm_iter_fwd_2_,
                    rnn_utils::scratch_t::key_gemm_iter_fwd_2);
            CHECK(intel::gpu_gemm(gemm_iter_fwd_2_)->execute(gemm_ctx));
            break;
        case gemm_layer_fwd:
            init_gemm_nested_scratchpad(
                    gemm_layer_fwd_, rnn_utils::scratch_t::key_gemm_layer_fwd);
            CHECK(intel::gpu_gemm(gemm_layer_fwd_)->execute(gemm_ctx));
            break;
        case gemm_layer_fwd_src:
            init_gemm_nested_scratchpad(gemm_layer_fwd_src_,
                    rnn_utils::scratch_t::key_gemm_layer_fwd_src);
            CHECK(intel::gpu_gemm(gemm_layer_fwd_src_)->execute(gemm_ctx));
            break;
        case gemm_iter_bwd:
            init_gemm_nested_scratchpad(
                    gemm_iter_bwd_, rnn_utils::scratch_t::key_gemm_iter_bwd);
            CHECK(intel::gpu_gemm(gemm_iter_bwd_)->execute(gemm_ctx));
            break;
        case gemm_iter_bwd_2:
            init_gemm_nested_scratchpad(gemm_iter_bwd_2_,
                    rnn_utils::scratch_t::key_gemm_iter_bwd_2);
            CHECK(intel::gpu_gemm(gemm_iter_bwd_2_)->execute(gemm_ctx));
            break;
        case gemm_layer_bwd:
            init_gemm_nested_scratchpad(
                    gemm_layer_bwd_, rnn_utils::scratch_t::key_gemm_layer_bwd);
            CHECK(intel::gpu_gemm(gemm_layer_bwd_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_iter:
            init_gemm_nested_scratchpad(gemm_diff_wei_iter_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_iter);
            CHECK(intel::gpu_gemm(gemm_diff_wei_iter_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_layer:
            init_gemm_nested_scratchpad(gemm_diff_wei_layer_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_layer);
            CHECK(intel::gpu_gemm(gemm_diff_wei_layer_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_layer_src:
            init_gemm_nested_scratchpad(gemm_diff_wei_layer_src_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_layer_src);
            CHECK(intel::gpu_gemm(gemm_diff_wei_layer_src_)->execute(gemm_ctx));
            break;
        case gemm_diff_wei_iter_2:
            init_gemm_nested_scratchpad(gemm_diff_wei_iter_2_,
                    rnn_utils::scratch_t::key_gemm_diff_wei_iter_2);
            CHECK(intel::gpu_gemm(gemm_diff_wei_iter_2_)->execute(gemm_ctx));
            break;
        default: assert(!"unknown gemm_kind"); return status::runtime_error;
    }

    return status::success;
}

//*************** Grid computations strategy: linear ***************//
template <prop_kind_t aprop>
grid_execution_sig((_ref_rnn_common_t<aprop>::linear_execution)) {
    const conf_t &rnn = pd()->rnn_conf;
    dim_t n_layer = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;
    dim_t n_iter = rnn.n_iter;

    // TODO
    //     if (aprop == prop_kind::backward && pd()->diff_weights_overwrite()) {
    //         compute::compute_stream_t *compute_stream
    //                 = utils::downcast<compute::compute_stream_t *>(ctx.stream());
    //         auto zero = [&](const memory_storage_t &data, int arg_id) {
    //             auto mdw = memory_desc_wrapper(pd()->arg_md(arg_id));
    //             return compute_stream->fill(data, 0, mdw.size(),
    //                     compute_stream->ctx().get_deps(),
    //                     compute_stream->ctx().get_deps());
    //         };

    //         CHECK(zero(diff_bias, DNNL_ARG_DIFF_BIAS));
    //         CHECK(zero(user_data.diff_wei_layer(), DNNL_ARG_DIFF_WEIGHTS_LAYER));
    //         CHECK(zero(user_data.diff_wei_iter(), DNNL_ARG_DIFF_WEIGHTS_ITER));
    //     }
    // Grid Computation for RNN with a cell execution call

    for (dim_t dir = 0; dir < n_dir; dir++) {
        for (dim_t j = 0; j < n_layer; j++) {
            dim_t lay = (aprop == prop_kind::forward) ? j : n_layer - j - 1;

            auto grid_iter = rnn.merge_gemm_iter
                    ? workspace.states_range(lay, n_layer, dir, dir, -1, -1)
                    : sub_buffer_t();
            /*
            if ((aprop == prop_kind::forward || rnn.recompute_gates)
                    && rnn.merge_gemm_layer && !rnn.cell_fusion.gemm_layer) {
                auto grid_layer = (!rnn.copy_src_layer && lay == 0)
                        ? user_data.src_layer(dir, 0, true)
                        : workspace.states_range(
                                lay - 1, lay - 1, dir, dir, -1, n_iter-1);

                auto gemm_grid_layer_fwd = (!rnn.copy_src_layer && lay == 0)
                        ? gemm_layer_fwd_src
                        : gemm_layer_fwd;
                nvidia::stream_t *stream
                        = utils::downcast<nvidia::stream_t *>(ctx.stream());
                printf("\n ========= BEFORE GEMM ==========\n");
                stream->wait();
                PRINT_VEC2(grid_layer.get(),  16)
                PRINT_VEC2(user_data.wei_layer(lay, dir, true).get(), 16 * 16)
                printf("\n =====INPUT ==========\n");

                void *raw_data = nullptr;
                grid_layer.get()->map_data(
                        &raw_data, nullptr, 16 * sizeof(float));
                for (auto i = 0; i < 16; i++) {
                    std::cout << static_cast<float *>(raw_data)[i] << ", ";
                }
                std::cout << "\n\n";
                grid_layer.get()->unmap_data(raw_data, nullptr);
                printf("\n ========= WEIGHTS ==========\n");

                user_data.wei_layer(lay, dir, true).get()->map_data(
                        &raw_data, nullptr, 16*16 * sizeof(float));
                for (auto i = 0; i < 16*16; i++) {
                    std::cout << static_cast<float *>(raw_data)[i] << ", ";
                }
                std::cout << "\n\n";
                user_data.wei_layer(lay, dir, true).get()->unmap_data(raw_data, nullptr);

                PRINT_VEC2(scratch.gates(), 16)

                // ANTON
                //CHECK(gemm_primitive(engine, ctx,
                //        user_data.wei_layer(lay, dir, true), grid_layer,
                //        *scratch.gates(), gemm_grid_layer_fwd));
                //CHECK(gemm_primitive(engine, ctx, grid_layer,
                //        user_data.wei_layer(lay, dir, true), *scratch.gates(),
                //        gemm_grid_layer_fwd));

                CHECK(gemm_primitive(engine, ctx,
                        user_data.wei_layer(lay, dir, false), grid_layer,
                        *scratch.gates(), gemm_grid_layer_fwd));

                printf("\n ========= AFTER GEMM ==========\n");
                stream->wait();
                PRINT_VEC2(scratch.gates(), 16*2)
            }
*/

            for (dim_t i = 0; i < n_iter; i += rnn.iter_loop) {
                dim_t iter = (aprop == prop_kind::forward) ? i : n_iter - i - 1;
                CHECK((this->*cell_func)(engine, ctx, dir, lay, iter, user_data,
                        workspace, scratch, diff_bias, scales, tm_scales,
                        bias_binary_, activation_primitives));
            }

            if (aprop == prop_kind::backward && rnn.merge_gemm_layer) {
                auto grid_layer = (!rnn.copy_src_layer && lay == 0)
                        ? user_data.src_layer(dir, 0)
                        : workspace.states(lay - 1, dir, 0);

                auto gemm_diff_wei_grid_layer
                        = (!rnn.copy_src_layer && lay == 0)
                        ? gemm_diff_wei_layer_src
                        : gemm_diff_wei_layer;

                // TODO: Fix sub-buffer size
                auto diff_states
                        = scratch.diff_states(lay, dir, rnn.n_states, 0);

                CHECK(gemm_primitive(engine, ctx,
                        user_data.wei_layer(lay, dir, true),
                        *scratch.diff_gates(), diff_states, gemm_layer_bwd));
                CHECK(gemm_primitive(engine, ctx, *scratch.diff_gates(),
                        grid_layer, user_data.diff_wei_layer(lay, dir, true),
                        gemm_diff_wei_grid_layer));
            }

            if (aprop == prop_kind::backward && rnn.merge_gemm_iter) {
                CHECK(gemm_primitive(engine, ctx, *scratch.diff_gates(),
                        grid_iter, user_data.diff_wei_iter(lay, dir, true),
                        gemm_diff_wei_iter));
            }
        }
    }
    return status::success;
}
//********* GRID computations strategy: utility functions **********//

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::bias_prepare(const exec_ctx_t &ctx,
        dim_t n_layer, dim_t n_dir, dim_t n_bias, dim_t n_gates, dim_t dhc,
        const memory_storage_t &ws_bias, const memory_storage_t &scales,
        const memory_storage_t &wei_layer, const memory_storage_t &wei_iter,
        const memory_storage_t &bias) const {
    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::copy_init_layer(const exec_ctx_t &ctx,
        bool lr, bool rl, dim_t batch, dim_t dhc, dim_t slc, dim_t n_iter,
        dim_t n_layer, dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld, const memory_storage_t &ws_states,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &input,
        const memory_storage_t &diff_dst_layer) const {
    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    printf("\n ========= BEFORE COPY INIT ==========\n");
    stream->wait();
    PRINT_VEC(ws_states, 256)

    parallel_for(ctx, copy_init_layer_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &input)
                          ->get_in_memory_arg(ctx.stream(),
                                  cgh); // CTX_IN_SYCL_KERNEL_MEMORY(DNNL_ARG_SRC);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws_states)
                          ->get_out_memory_arg(ctx.stream(),
                                  cgh); // CTX_OUT_SYCL_KERNEL_MEMORY(DNNL_ARG_DST);

        ref_rnn_copy_init_layer_t copy_kernel(
                pd()->copy_init_layer_conf_, src_mem_arg, dst_mem_arg);
        size_t local_batch = 4;
        size_t local_iter = 4;
        size_t local_channel = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_iter = calc_global_range(static_cast<size_t>(n_iter));
        size_t global_channels = calc_global_range(static_cast<size_t>(slc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_iter, global_batch,
                                            global_channels),
                        ::sycl::range<3>(
                                local_iter, local_batch, local_channel)),
                copy_kernel);
    });
    printf("\n ========= AFTER COPY INIT ==========\n");
    stream->wait();
    PRINT_VEC(ws_states, 256)

    return status::success;
}
template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::copy_init_iter(const exec_ctx_t &ctx,
        dim_t batch, dim_t dhc, dim_t sic, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld, const rnn_utils::workspace_t &ws,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &firstit_states,
        const memory_storage_t &firstit_c_states,
        const memory_storage_t &diff_dst_iter,
        const memory_storage_t &diff_dst_iter_c, const float shift,
        const float scale, const bool quantize) const {
    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());
    parallel_for(ctx, copy_init_iter_kernel_, [&](::sycl::handler &cgh) {
        auto src_iter_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &firstit_states)
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto src_iter_c_mem_arg
                = xpu::sycl::memory_storage_base_t::empty_in_memory_arg(
                        ctx.stream(), cgh);
        auto ws_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_out_memory_arg(ctx.stream(), cgh);

        ref_rnn_copy_init_iter_t copy_kernel(pd()->copy_init_iter_conf_,
                src_iter_mem_arg, src_iter_c_mem_arg, ws_mem_arg);
        size_t local_batch = 4;
        size_t local_channel = 4;
        size_t local_lay_dir = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_channels = calc_global_range(
                std::max(static_cast<size_t>(sic), static_cast<size_t>(dhc)));
        size_t global_lay_dir
                = calc_global_range(static_cast<size_t>(n_layer * n_dir));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_lay_dir,
                                            global_batch, global_channels),
                        ::sycl::range<3>(
                                local_lay_dir, local_batch, local_channel)),
                copy_kernel);
    });
    stream->wait();
    printf("\n ========= AFTER COPY ITER ==========\n");
    PRINT_VEC(ws.states(), 256)
    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::copy_res_layer(const exec_ctx_t &ctx,
        bool lr, bool rl, dim_t batch, dim_t dhc, dim_t slc, dim_t n_iter,
        dim_t n_layer, dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &dst_last_layer,
        const memory_storage_t &diff_src_layer,
        const rnn_utils::workspace_t &ws, const float shift, const float scale,
        const bool dequantize) const {

    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());
    printf("\n ========= BEFORE COPY RES ==========\n");
    stream->wait();
    PRINT_VEC(ws.states(), 256)
    parallel_for(ctx, copy_res_layer_kernel_, [&](::sycl::handler &cgh) {
        auto ws_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &dst_last_layer)
                          ->get_out_memory_arg(ctx.stream(), cgh);

        ref_rnn_copy_res_layer_t copy_kernel(
                pd()->copy_res_layer_conf_, ws_mem_arg, dst_mem_arg);
        size_t local_batch = 4;
        size_t local_iter = 4;
        size_t local_channel = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_iter = calc_global_range(static_cast<size_t>(n_iter));
        size_t global_channels
                = calc_global_range(static_cast<size_t>(n_states * dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_iter, global_batch,
                                            global_channels),
                        ::sycl::range<3>(
                                local_iter, local_batch, local_channel)),
                copy_kernel);
    });
    printf("\n ========= AFTER COPY RES ==========\n");
    stream->wait();
    PRINT_VEC(dst_last_layer, 15)
    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::copy_res_iter(const exec_ctx_t &ctx,
        dim_t batch, dim_t dhc, dim_t sic, dim_t n_iter, dim_t n_layer,
        dim_t n_dir, dim_t n_states, dim_t states_ws_ld,
        dim_t scratch_diff_states_ld,
        const memory_storage_t *scratch_diff_states,
        const memory_storage_t &dst_last_iter,
        const memory_storage_t &dst_last_iter_c,
        const memory_storage_t &diff_src_iter,
        const memory_storage_t &diff_src_iter_c,
        const rnn_utils::workspace_t &ws, const float shift, const float scale,
        const bool dequantize) const {

    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    parallel_for(ctx, copy_res_iter_kernel_, [&](::sycl::handler &cgh) {
        auto src_iter
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.states())
                          ->get_in_memory_arg(ctx.stream(), cgh);
        auto dst_iter
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &dst_last_iter)
                          ->get_out_memory_arg(ctx.stream(), cgh);
        auto dst_iter_c = pd()->copy_res_iter_conf_.with_dst_iter_c
                ? utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &dst_last_iter_c)
                          ->get_out_memory_arg(ctx.stream(), cgh)
                : xpu::sycl::memory_storage_base_t::empty_out_memory_arg(
                        ctx.stream(), cgh);
        ref_rnn_copy_res_iter copy_kernel(
                pd()->copy_res_iter_conf_, src_iter, dst_iter, dst_iter_c);

        size_t local_batch = 4;
        size_t local_channel = 4;
        size_t local_lay_dir = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_channels = calc_global_range(static_cast<size_t>(dhc));
        size_t global_lay_dir
                = calc_global_range(static_cast<size_t>(n_layer * n_dir));
        cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(global_lay_dir,
                                            global_batch, global_channels),
                        ::sycl::range<3>(
                                local_lay_dir, local_batch, local_channel)),
                copy_kernel);
    });

    return status::success;
}

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::rnn_bias(const exec_ctx_t &ctx, dim_t batch,
        dim_t dhc, dim_t iter, dim_t lay, dim_t dir,
        const rnn_utils::workspace_t &ws, const rnn_utils::scratch_t &scratch,
        const rnn_utils ::user_data_t &user_data) const {
    nvidia::stream_t *stream
            = utils::downcast<nvidia::stream_t *>(ctx.stream());

    printf("\n ========= BEFORE BIAS ==========\n");
    stream->wait();
    PRINT_VEC(ws.states(), 256)

    parallel_for(ctx, bias_kernel_, [&](::sycl::handler &cgh) {
        auto src_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &scratch.sub_gates(0).get_storage())
                          ->get_inout_memory_arg(ctx.stream(), cgh);
        auto bias_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        user_data.sub_bias(lay, dir).get())
                          ->get_in_memory_arg(ctx.stream(), cgh);

        auto dst_mem_arg
                = utils::downcast<const xpu::sycl::memory_storage_base_t *>(
                        &ws.sub_state(lay, dir, iter-1
                        ).get_storage())
                          ->get_out_memory_arg(ctx.stream(), cgh);
        ref_rnn_bias bias_kernel(pd()->sycl_rnn_bias_conf_t_, src_mem_arg,
                bias_mem_arg, dst_mem_arg);

        size_t local_batch = 4;
        size_t local_channel = 4;
        size_t global_batch = calc_global_range(static_cast<size_t>(batch));
        size_t global_channels = calc_global_range(static_cast<size_t>(dhc));
        cgh.parallel_for(
                ::sycl::nd_range<3>(
                        ::sycl::range<3>(global_channels, global_batch, 1),
                        ::sycl::range<3>(local_channel, local_batch, 1)),
                bias_kernel);
    });

    printf("\n ========= AFTER BIAS ==========\n");
    stream->wait();
    PRINT_VEC(ws.states(), 256)

    return status::success;
}

// //********************* Execution function *********************//

template <prop_kind_t aprop>
status_t _ref_rnn_common_t<aprop>::execute_(const exec_ctx_t &ctx) const {

    impl::engine_t *engine = ctx.stream()->engine();

    auto rnn_pd = this->pd();

    const conf_t &rnn = this->pd()->rnn_conf;

    dim_t n_layer = rnn.n_layer;
    dim_t n_dir = rnn.n_dir;
    dim_t n_states = rnn.n_states;
    dim_t n_iter = rnn.n_iter;
    dim_t n_gates = rnn.n_gates;
    dim_t n_bias = rnn.n_bias;
    dim_t batch = rnn.mb;
    dim_t slc = rnn.slc;
    dim_t sic = rnn.sic;
    dim_t dhc = rnn.dhc;
    dim_t dlc = rnn.dlc;

    bool is_fwd = rnn.is_fwd;
    bool is_vanilla_gru = rnn.is_vanilla_gru;

    auto &src_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_LAYER);
    auto &src_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER);
    auto &src_c_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_SRC_ITER_C);
    auto &wei_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_LAYER);
    auto &wei_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_WEIGHTS_ITER);
    auto &bias_native_ = CTX_IN_STORAGE(DNNL_ARG_BIAS);

    auto &dst_last_layer_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_LAYER)
                                          : CTX_IN_STORAGE(DNNL_ARG_DST_LAYER);
    auto &dst_last_iter_native_ = is_fwd ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER)
                                         : CTX_IN_STORAGE(DNNL_ARG_DST_ITER);
    auto &dst_last_iter_c_native_ = is_fwd
            ? CTX_OUT_STORAGE(DNNL_ARG_DST_ITER_C)
            : CTX_IN_STORAGE(DNNL_ARG_DST_ITER_C);

    auto &diff_dst_layer_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_LAYER);
    auto &diff_dst_iter_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER);
    auto &diff_dst_iter_c_native_ = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST_ITER_C);

    auto scratch_workspace
            = ctx.get_scratchpad_grantor().get_memory_storage(key_rnn_space);
    auto &workspace_ = rnn.is_training ? is_fwd
                    ? CTX_OUT_STORAGE(DNNL_ARG_WORKSPACE)
                    : CTX_IN_STORAGE(DNNL_ARG_WORKSPACE)
                                       : *scratch_workspace;
    const auto &workspace = rnn_utils::workspace_t(workspace_, rnn);

    const auto scratch
            = rnn_utils::scratch_t(rnn, ctx.get_scratchpad_grantor());

    auto &diff_src_layer_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_LAYER);
    auto &diff_src_iter_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER);
    auto &diff_src_iter_c_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC_ITER_C);

    auto &diff_weights_layer_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_LAYER);
    auto &diff_weights_iter_native_
            = CTX_OUT_STORAGE(DNNL_ARG_DIFF_WEIGHTS_ITER);
    auto &diff_bias_native_ = CTX_OUT_STORAGE(DNNL_ARG_DIFF_BIAS);

    const rnn_utils::user_data_t user_data(src_layer_native_, wei_layer_native_,
            wei_iter_native_, bias_native_, diff_src_layer_native_,
            diff_dst_layer_native_, diff_weights_layer_native_,
            diff_weights_iter_native_, rnn, pd()->off);

    DPRINT("\n%s\n", "+++++++++++++++");
    DPRINT(" aprop = %d\n", (int)aprop);
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  n_layer         = %lld\n", into<long long>(n_layer));
    DPRINT("  n_dir           = %lld\n", into<long long>(n_dir));
    DPRINT("  n_iter          = %lld\n", into<long long>(n_iter));
    DPRINT("  n_gates         = %lld\n", into<long long>(n_gates));
    DPRINT("  n_bias          = %lld\n", into<long long>(n_bias));
    DPRINT("  n_states        = %lld\n", into<long long>(n_states));
    DPRINT("  n_weights_layer = %lld\n", into<long long>(rnn_pd->SLC()));
    DPRINT("  n_weights_iter  = %lld\n", into<long long>(rnn_pd->SIC()));
    DPRINT("  batch           = %lld\n", into<long long>(batch));
    DPRINT("  slc             = %lld\n", into<long long>(slc));
    DPRINT("  sic             = %lld\n", into<long long>(sic));
    DPRINT("  dhc             = %lld\n", into<long long>(dhc));
    DPRINT("  dlc             = %lld\n", into<long long>(dlc));
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  is_fwd          = %s\n", is_fwd ? "yes" : "no");
    DPRINT("  is_vanilla_gru  = %s\n", is_vanilla_gru ? "yes" : "no");
    DPRINT("  use_workspace   = %s\n", rnn.use_workspace ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");
    DPRINT("  with_src_iter   = %s\n", rnn_pd->with_src_iter() ? "yes" : "no");
    DPRINT("  with_src_iter_c = %s\n",
            rnn_pd->with_src_iter_c() ? "yes" : "no");
    DPRINT("  with_bias       = %s\n", rnn_pd->with_bias() ? "yes" : "no");
    DPRINT("  with_dst_iter   = %s\n", rnn_pd->with_dst_iter() ? "yes" : "no");
    DPRINT("  with_dst_iter_c = %s\n",
            rnn_pd->with_dst_iter_c() ? "yes" : "no");
    DPRINT("%s\n", "+++++++++++++++");

    // TODO: implement without copies
    bool is_lr = !one_of(rnn.exec_dir, r2l, r2l);
    bool is_rl = !one_of(rnn.exec_dir, l2r, l2r);

    const memory_storage_t *scales_buf = nullptr;
    if (pd()->rnn_conf.is_int8 && pd()->rnn_conf.copy_bias) {
        scales_buf = &CTX_GPU_RES_STORAGE(SCALES_);
    }

    // bias prepare if needed
    if (rnn.copy_bias) {
        CHECK(bias_prepare(ctx, n_layer, n_dir, n_bias, n_gates, dhc,
                workspace.bias(), *scales_buf, wei_layer_native_,
                wei_iter_native_, user_data.bias()));
    }

    float shift = (pd()->attr()->rnn_data_qparams_.shift_);
    float scale = (pd()->attr()->rnn_data_qparams_.scale_);

    if ((rnn.is_fwd && rnn.copy_src_layer)
            || (!rnn.is_fwd && rnn.copy_diff_dst_layer)) {
        CHECK(copy_init_layer(ctx, is_lr, is_rl, batch, dhc, slc, n_iter,
                n_layer, n_dir, n_states, rnn.states_ws_ld,
                rnn.scratch_diff_states_ld, workspace.states(),
                scratch.diff_states(), src_layer_native_,
                diff_dst_layer_native_));
    }
    const bool quantize = pd()->with_src_iter()
            && pd()->src_md(1)->data_type == data_type::f32 && rnn.is_int8;
    CHECK(copy_init_iter(ctx, batch, dhc, sic, n_iter, n_layer, n_dir, n_states,
            rnn.states_ws_ld, rnn.scratch_diff_states_ld, workspace,
            scratch.diff_states(), src_iter_native_, src_c_iter_native_,
            diff_dst_iter_native_, diff_dst_iter_c_native_, shift, scale,
            quantize));

    const memory_storage_t *tm_scales_buf = nullptr;
    if (pd()->rnn_conf.is_testmode && pd_->attr()->rnn_tparams_.scales_) {
        tm_scales_buf = &CTX_GPU_RES_STORAGE(TM_SCALES_);
    }

    std::vector<std::shared_ptr<impl::primitive_t>> activation_primitives {
            vanilla_cell_act_};
    // run the execution on the grid
    CHECK((this->*grid_computation)(engine, ctx, user_data, workspace, scratch,
            diff_bias_native_, scales_buf, tm_scales_buf, bias_binary_,
            activation_primitives));

    // Finally we copy the results to the result buffers

    if (rnn.is_fwd || rnn.copy_diff_src_layer) {
        const bool dequantize_l
                = pd()->dst_md(0)->data_type == data_type::f32 && rnn.is_int8;
        CHECK(copy_res_layer(ctx, is_lr, is_rl, batch, dhc, slc, n_iter,
                n_layer, n_dir, n_states, rnn.states_ws_ld,
                rnn.scratch_diff_states_ld, scratch.diff_states(),
                dst_last_layer_native_, diff_src_layer_native_, workspace,
                shift, scale, dequantize_l));
    }
    const bool dequantize_i = pd()->with_dst_iter()
            && pd()->dst_md(1)->data_type == data_type::f32 && rnn.is_int8;
    CHECK(copy_res_iter(ctx, batch, dhc, sic, n_iter, n_layer, n_dir, n_states,
            rnn.states_ws_ld, rnn.scratch_diff_states_ld, scratch.diff_states(),
            dst_last_iter_native_, dst_last_iter_c_native_,
            diff_src_iter_native_, diff_src_iter_c_native_, workspace, shift,
            scale, dequantize_i));

    return status::success;
};
// Fix for MSVS warning C4661.
template <>
cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template <>
cell_execution_sig(ref_rnn_bwd_t::cell_execution);
//template <>
//elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
//template <>
//bias_sig(ref_rnn_fwd_t::rnn_bias);
//template <>
//elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);
//template <>
//elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
//template <>
//elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);
//template <>
//elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
//template <>
//elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);

template struct _ref_rnn_common_t<prop_kind::forward>;
template struct _ref_rnn_common_t<prop_kind::backward>;

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
