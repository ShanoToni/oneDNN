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

// Common for RNN and LSTM cell execution

#include "gpu/generic/sycl/rnn/ref_rnn.hpp"
#include "gpu/nvidia/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::utils;
using namespace rnn_utils;

template <size_t out_ndims, size_t in_ndims>
strides_t<out_ndims> inner(const strides_t<in_ndims> &s) {
    static_assert(in_ndims >= out_ndims,
            "The output strides are expected to be smaller than the input "
            "strides");
    strides_t<out_ndims> ret;
    for (size_t i = 0; i < out_ndims; i++) {
        ret[i] = s[i + in_ndims - out_ndims];
    }
    return ret;
}

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

status_t compute_cell_fwd(const exec_ctx_t &ctx, int lay, int dir, int iter,
        const workspace_t &workspace, const user_data_t user_data,
        const sub_buffer_t &weights_layer, const sub_buffer_t &weights_iter,
        const sub_buffer_t &cell_layer, const strides_t<4> &cell_layer_strides,
        const sub_buffer_t &cell_iter, const strides_t<4> &cell_iter_strides,
        const sub_buffer_t &scratch_gates,
        const strides_t<2> &scratch_gates_strides, float alpha,
        const memory_storage_t *tm_scales, const conf_t &conf,
        const rnn_offsets_t &offsets) {
    return status::success;
}

template <prop_kind_t aprop>
cell_execution_sig((_ref_rnn_common_t<aprop>::cell_execution)) {
    const conf_t &rnn = this->pd()->rnn_conf;
    // const ocl_conf_t &ocl_conf = this->pd()->ocl_conf;
    const rnn_offsets_t &offsets = this->pd()->off;

    // const bool use_cell = ocl_conf.cell_comp.is_enabled;u
    auto use_cell = false; // TODO

    strides_t<4> user_layer_strides {[&]() {
        auto s = user_data.src_layer_strides(dir);
        return strides_t<4> {0, 0, s[0], s[1]};
    }()};
    auto cell_layer = !rnn.copy_src_layer && lay == 0
            ? user_data.src_layer(dir, iter)
            : workspace.states_range(
                    lay - 1, lay - 1, dir, dir, iter - 1, iter - 1);
    auto &cell_layer_strides = !rnn.copy_src_layer && lay == 0
            ? user_layer_strides
            : workspace.states_strides();
    //auto cell_iter = workspace.states(lay, dir, iter - 1);
    //auto cell_iter = workspace.states(iter - 1, dir, lay);
    auto cell_iter = workspace.states_range(
            lay, lay, dir, dir, iter - 2, iter - 2);

    auto &cell_iter_strides = workspace.states_strides();
    // TODO ANTON
    auto scratch_gates = scratch.gates(0);
    strides_t<2> scratch_gates_strides
            = {scratch.calc_off_gates(1), rnn.scratch_gates_ld};

    auto wei_layer = user_data.wei_layer(lay, dir);
    auto wei_iter = user_data.wei_iter(lay, dir);

    if ((aprop == prop_kind::forward) || rnn.recompute_gates) {

        if (!rnn.merge_gemm_layer && !rnn.cell_fusion.gemm_layer) {
            auto gemm_cell_layer_fwd = !rnn.copy_src_layer && lay == 0
                    ? gemm_layer_fwd_src
                    : gemm_layer_fwd;
            CHECK(gemm_primitive(engine, ctx, wei_layer, cell_layer,
                    scratch_gates, gemm_cell_layer_fwd));
        }

        if (!rnn.cell_fusion.gemm_iter) {
            nvidia::stream_t *stream
                    = utils::downcast<nvidia::stream_t *>(ctx.stream());
            printf("\n ========= BEFORE ITER GEMM ==========\n");
            stream->wait();
            PRINT_VEC2(scratch.gates(), 16 * 2)
            PRINT_VEC(workspace.states(), 16)
            PRINT_VEC(user_data.wei_iter(), 16 * 16)
            CHECK(gemm_primitive(engine, ctx, wei_iter, cell_iter,
                    scratch_gates, gemm_iter_fwd));

            printf("\n ========= AFTER ITER GEMM ==========\n");
            stream->wait();
            PRINT_VEC2(scratch.gates(), 16 * 2)
        }
    }

    if (aprop == prop_kind::forward) {
        if (!use_cell) {
            CHECK(rnn_bias(ctx, rnn.mb, rnn.dhc, iter, lay, dir, workspace,
                    scratch, user_data));
        } else {
            CHECK(compute_cell_fwd(ctx, lay, dir, iter, workspace, user_data,
                    wei_layer, wei_iter, cell_layer, cell_layer_strides,
                    cell_iter, cell_iter_strides, scratch_gates,
                    scratch_gates_strides, pd()->desc()->alpha, tm_scales, rnn,
                    offsets));
        }

    } else { // backward TODO
        /*
        auto diff_states_iter = scratch.diff_states(lay, dir, 0, iter + 1);
        auto diff_states_iter_s1 = rnn.n_states == 2
                ? scratch.diff_states(lay, dir, 1, iter + 1)
                : sub_buffer_t();
        auto diff_states_layer
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? user_data.diff_dst_layer(dir, iter)
                : scratch.diff_states(lay + 1, dir, rnn.n_states, iter);
        auto diff_states_layer_ld
                = !rnn.copy_diff_dst_layer && lay + 1 == rnn.n_layer
                ? offsets.diff_dst_layer[1]
                : rnn.scratch_diff_states_ld;

        auto diff_states = scratch.diff_states(lay, dir, 0, iter);
        auto diff_states_s1 = rnn.n_states == 2
                ? scratch.diff_states(lay, dir, 1, iter)
                : sub_buffer_t();
        auto diff_states1 = !rnn.copy_diff_src_layer && lay == 0
                ? user_data.diff_src_layer(dir, iter)
                : scratch.diff_states(lay, dir, rnn.n_states, iter);
        auto diff_gates = scratch.diff_gates(iter);

        CHECK((this->*bias_common)(ctx, dir, lay, iter, rnn.dhc, rnn.mb, 1,
                user_data, workspace, scratch_gates, diff_gates, diff_states,
                diff_states_s1, diff_states_iter, diff_states_iter_s1,
                diff_states_layer, diff_states_layer_ld, scales, tm_scales,
                diff_bias, bias_primitive, activation_primitives));

        CHECK(gemm_primitive(
                engine, ctx, wei_iter, diff_gates, diff_states, gemm_iter_bwd));

        if (!rnn.merge_gemm_layer) {
            CHECK(gemm_primitive(engine, ctx, wei_layer, diff_gates,
                    diff_states1, gemm_layer_bwd));

            auto gemm_diff_wei_cell_layer = !rnn.copy_src_layer && lay == 0
                    ? gemm_diff_wei_layer_src
                    : gemm_diff_wei_layer;

            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_layer,
                    user_data.diff_wei_layer(lay, dir),
                    gemm_diff_wei_cell_layer));
        }

        if (!rnn.merge_gemm_iter) {
            CHECK(gemm_primitive(engine, ctx, diff_gates, cell_iter,
                    user_data.diff_wei_iter(lay, dir), gemm_diff_wei_iter));
        }
        */
    }
    return status::success;
}
template cell_execution_sig(ref_rnn_fwd_t::cell_execution);
template cell_execution_sig(ref_rnn_bwd_t::cell_execution);
} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
