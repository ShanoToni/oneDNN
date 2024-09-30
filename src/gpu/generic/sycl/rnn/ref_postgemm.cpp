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
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

using namespace dnnl::impl::gpu::intel::gpu_utils;
using namespace rnn_utils;

#define PRINT_VEC_PTR(data, size) \
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


template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::rnn_elemwise)) {
    memory_desc_t bias_md = types::zero_md();
    dims_t dims = {dim_t{1}, dim_t{1}, dim_t{1}, dim_t{16}};
    memory_desc_init_by_tag(bias_md, 4,
                            dims, data_type::f32, format_tag::ldgo);
    memory_t a(ctx.stream()->engine(), &bias_md,
                user_data.bias(lay, dir).get_ptr()->clone());
    memory_t b(ctx.stream()->engine(), &bias_md,
                scratch_gates.get_ptr()->clone());
    memory_t c(ctx.stream()->engine(), &bias_md,
                workspace.states(lay, dir, iter).get_ptr()->clone());
    
    PRINT_VEC_PTR(user_data.bias(lay, dir).get_ptr(), 16)
    PRINT_VEC_PTR(scratch_gates.get_ptr()->clone(), 16)
    PRINT_VEC_PTR(workspace.states(lay, dir, iter).get_ptr(), 128)
    
    exec_args_t bias_args;
    bias_args[DNNL_ARG_SRC_0] = {&a, true};
    bias_args[DNNL_ARG_SRC_1] = {&b, true};
    bias_args[DNNL_ARG_DST] = {&c, false};
    exec_ctx_t bias_ctx(ctx, std::move(bias_args));
    // auto bias_exec_ctx = ctx.into_exec_ctx_t(std::move(bias_args));
    auto status = bias_primitive->execute(bias_ctx);
    ctx.stream()->wait();
    PRINT_VEC_PTR(workspace.states(lay, dir, iter).get_ptr(), 128)

    return status;
    // auto nd_range = get_nd_range({dhc,
    //         utils::div_up(
    //                 batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    // const compute::kernel_t &kernel = (aprop == prop_kind::forward)
    //         ? kernels_[kernel_id::elemwise_fwd]
    //         : kernels_[kernel_id::elemwise_bwd];

    // arg_list_t arg_list;
    // if (aprop == prop_kind::backward) {
    //     arg_list.append(into<int32_t>(dir));
    //     arg_list.append(into<int32_t>(lay));
    //     arg_list.append(into<int32_t>(iter));
    // }
    // if (aprop == prop_kind::forward) {
    //     arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    // } else {
    //     arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
    //     arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
    //             pd()->ocl_conf.acc_dt);
    // }
    // auto bias = user_data.bias(lay, dir);
    // arg_list.append(bias, pd()->ocl_conf.bia_dt);
    // arg_list.append(pd()->desc()->alpha);
    // // for test mode
    // arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    // data_type_t ws_dt = pd()->ocl_conf.src_dt;
    // auto states_t_l = workspace.states(lay, dir, iter);
    // arg_list.append(states_t_l, ws_dt);

    // auto c_states_t_l = workspace.c_states(lay, dir, iter);
    // auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    // arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    // arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    // auto gates = workspace.gates(lay, dir, iter);
    // arg_list.append(gates, pd()->ocl_conf.aux_dt);

    // auto ws_grid = workspace.grid_comp(lay, dir, iter);
    // arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    // arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    // arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    // if (aprop == prop_kind::forward)
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // else {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // }

    // arg_list.append(into<int32_t>(batch));
    // arg_list.append(into<int32_t>(dhc));
    // if (aprop == dnnl_backward) {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    //     arg_list.append(into<int32_t>(diff_states_layer_ld));
    // }

    // arg_list.append(pd()->rnn_conf.tm_cscale);
    // if (aprop != dnnl_forward) {
    //     auto diff_dt = pd()->ocl_conf.diff_dt;
    //     arg_list.append(scratch_diff_states, diff_dt);
    //     arg_list.append(scratch_diff_states_iter, diff_dt);
    //     arg_list.append(scratch_diff_states_layer, diff_dt);
    //     arg_list.append(diff_bias);
    //     arg_list.append(pd()->off.diff_bias);
    // }
    // return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig(ref_rnn_fwd_t::rnn_elemwise);
template elemwise_sig(ref_rnn_bwd_t::rnn_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise)) {
    return status::success;
    // auto nd_range = get_nd_range({dhc,
    //         utils::div_up(
    //                 batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    // const compute::kernel_t &kernel = (aprop == prop_kind::forward)
    //         ? kernels_[kernel_id::elemwise_fwd]
    //         : kernels_[kernel_id::elemwise_bwd];

    // arg_list_t arg_list;
    // if (aprop == prop_kind::backward) {
    //     arg_list.append(into<int32_t>(dir));
    //     arg_list.append(into<int32_t>(lay));
    //     arg_list.append(into<int32_t>(iter));
    // }
    // if (aprop == prop_kind::forward) {
    //     arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    // } else {
    //     arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
    //     arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
    //             pd()->ocl_conf.acc_dt);
    // }
    // auto bias = user_data.bias(lay, dir);
    // arg_list.append(bias, pd()->ocl_conf.bia_dt);
    // arg_list.append(pd()->desc()->alpha);
    // // for test mode
    // arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    // data_type_t ws_dt = pd()->ocl_conf.src_dt;
    // auto states_t_l = workspace.states(lay, dir, iter);
    // arg_list.append(states_t_l, ws_dt);

    // auto c_states_t_l = workspace.c_states(lay, dir, iter);
    // auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    // arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
    // arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

    // auto gates = workspace.gates(lay, dir, iter);
    // arg_list.append(gates, pd()->ocl_conf.aux_dt);

    // auto ws_grid = workspace.grid_comp(lay, dir, iter);
    // arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

    // arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    // arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
    // if (aprop == prop_kind::forward) {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // } else {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // }
    // arg_list.append(into<int32_t>(batch));
    // arg_list.append(into<int32_t>(dhc));
    // if (aprop == dnnl_backward) {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
    //     arg_list.append(into<int32_t>(diff_states_layer_ld));
    // }

    // arg_list.append(pd()->rnn_conf.tm_cscale);
    // if (aprop != dnnl_forward) {
    //     auto diff_dt = pd()->ocl_conf.diff_dt;
    //     arg_list.append(scratch_diff_states, diff_dt);
    //     arg_list.append(scratch_diff_states_iter, diff_dt);
    //     arg_list.append(scratch_diff_states_layer, diff_dt);
    //     arg_list.append(scratch_diff_states_s1, diff_dt);
    //     arg_list.append(scratch_diff_states_iter_s1, diff_dt);
    //     arg_list.append(diff_bias);
    //     arg_list.append(pd()->off.diff_bias);
    // }
    // return parallel_for(ctx, nd_range, kernel, arg_list.args);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise);

template <prop_kind_t aprop>
elemwise_sig((_ref_rnn_common_t<aprop>::lstm_elemwise_u8s8)) {

    return status::success;
    // auto nd_range = get_nd_range({dhc,
    //         utils::div_up(
    //                 batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

    // float data_shift = pd()->attr()->rnn_data_qparams_.shift_;
    // float data_scale = pd()->attr()->rnn_data_qparams_.scale_;

    // arg_list_t arg_list;
    // arg_list.append(into<int32_t>(dir));
    // arg_list.append(into<int32_t>(lay));
    // arg_list.append(into<int32_t>(iter));
    // if (aprop == prop_kind::forward) {
    //     arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
    // } else {
    //     arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
    //     arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
    //             pd()->ocl_conf.acc_dt);
    // }
    // arg_list.append(scales ? *scales : memory_storage_t::empty_storage());
    // arg_list.append(pd()->desc()->alpha);
    // arg_list.append(data_shift);
    // arg_list.append(data_scale);
    // // for test mode
    // arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

    // data_type_t ws_dt = pd()->ocl_conf.src_dt;
    // auto states_t1_l = workspace.states(lay, dir, iter);
    // arg_list.append(states_t1_l, ws_dt);

    // auto c_states_t_l = workspace.c_states(lay, dir, iter);
    // auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
    // arg_list.append(c_states_t_l, data_type::f32);
    // arg_list.append(c_states_tm1_l, data_type::f32);

    // auto gates = workspace.gates(lay, dir, iter);
    // arg_list.append(gates, pd()->ocl_conf.aux_dt);

    // arg_list.append(workspace.bias());

    // arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
    // if (aprop == prop_kind::forward) {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // } else {
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
    //     arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
    // }
    // arg_list.append(into<int32_t>(batch));
    // arg_list.append(into<int32_t>(dhc));
    // arg_list.append(into<int32_t>(pd()->rnn_conf.n_layer));
    // arg_list.append(into<int32_t>(pd()->rnn_conf.n_dir));
    // arg_list.append(pd()->rnn_conf.tm_cscale);
    // return parallel_for(
    //         ctx, nd_range, kernels_[kernel_id::elemwise_fwd], arg_list.args);
}
template elemwise_sig(ref_rnn_fwd_t::lstm_elemwise_u8s8);
template elemwise_sig(ref_rnn_bwd_t::lstm_elemwise_u8s8);

// template <prop_kind_t aprop>
// elemwise_sig_gru_lbr((_ref_rnn_common_t<aprop>::gru_lbr_elemwise)) {
//     auto nd_range = get_nd_range({dhc,
//             utils::div_up(
//                     batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

//     const compute::kernel_t &kernel = (aprop == prop_kind::forward)
//             ? kernels_[kernel_id::elemwise_fwd]
//             : kernels_[kernel_id::elemwise_bwd];

//     arg_list_t arg_list;
//     if (aprop == prop_kind::backward) {
//         arg_list.append(into<int32_t>(dir));
//         arg_list.append(into<int32_t>(lay));
//         arg_list.append(into<int32_t>(iter));
//     }
//     if (aprop == prop_kind::forward) {
//         arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
//     } else {
//         arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
//         arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
//                 pd()->ocl_conf.acc_dt);
//     }
//     auto bias = user_data.bias(lay, dir);
//     arg_list.append(bias, pd()->ocl_conf.bia_dt);
//     arg_list.append(pd()->desc()->alpha);
//     // for test mode
//     arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

//     data_type_t ws_dt = pd()->ocl_conf.src_dt;
//     auto states_t1_l = workspace.states(lay, dir, iter);
//     auto states_tm1_l = workspace.states(lay, dir, iter - 1);
//     arg_list.append(
//             aprop == prop_kind::forward ? states_t1_l : states_tm1_l, ws_dt);

//     auto c_states_t_l = workspace.c_states(lay, dir, iter);
//     auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
//     arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
//     arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

//     auto gates = workspace.gates(lay, dir, iter);
//     arg_list.append(gates, pd()->ocl_conf.aux_dt);

//     auto ws_grid = workspace.grid_comp(lay, dir, iter);
//     arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

//     arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
//     arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
//     if (aprop == prop_kind::forward) {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
//     } else {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
//     }
//     arg_list.append(into<int32_t>(batch));
//     arg_list.append(into<int32_t>(dhc));
//     if (aprop == dnnl_backward) {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
//         arg_list.append(into<int32_t>(diff_states_layer_ld));
//     }

//     if (aprop == dnnl_forward) { arg_list.append(states_tm1_l, ws_dt); }
//     arg_list.append(scratch_cell);
//     if (aprop != dnnl_forward) {
//         auto diff_dt = pd()->ocl_conf.diff_dt;
//         arg_list.append(scratch_diff_states, diff_dt);
//         arg_list.append(scratch_diff_states_iter, diff_dt);
//         arg_list.append(scratch_diff_states_layer, diff_dt);
//         arg_list.append(diff_bias);
//         arg_list.append(pd()->off.diff_bias);
//     }
//     return parallel_for(ctx, nd_range, kernel, arg_list.args);
// }
// template elemwise_sig_gru_lbr(ref_rnn_fwd_t::gru_lbr_elemwise);
// template elemwise_sig_gru_lbr(ref_rnn_bwd_t::gru_lbr_elemwise);

// template <prop_kind_t aprop>
// elemwise_sig_gru((_ref_rnn_common_t<aprop>::gru_elemwise)) {
//     auto nd_range = get_nd_range({dhc,
//             utils::div_up(
//                     batch, aprop == prop_kind::forward ? 1 : bwd_batch_block)});

//     const compute::kernel_t &kernel = (aprop == prop_kind::forward)
//             ? kernels_[kernel_id::elemwise_fwd]
//             : kernels_[kernel_id::elemwise_bwd];

//     arg_list_t arg_list;
//     if (aprop == prop_kind::backward) {
//         arg_list.append(into<int32_t>(dir));
//         arg_list.append(into<int32_t>(lay));
//         arg_list.append(into<int32_t>(iter));
//     }
//     if (aprop == prop_kind::forward) {
//         arg_list.append(scratch_gates, pd()->ocl_conf.acc_dt);
//     } else {
//         arg_list.append(scratch_diff_gates, pd()->ocl_conf.src_dt);
//         arg_list.append(scratch_gates ? scratch_gates : scratch_diff_gates,
//                 pd()->ocl_conf.acc_dt);
//     }
//     auto bias = user_data.bias(lay, dir);
//     arg_list.append(bias, pd()->ocl_conf.bia_dt);
//     arg_list.append(pd()->desc()->alpha);
//     arg_list.append(tm_scales ? *tm_scales : memory_storage_t::empty_storage());

//     data_type_t ws_dt = pd()->ocl_conf.src_dt;
//     auto states_t1_l = workspace.states(lay, dir, iter);
//     auto states_tm1_l = workspace.states(lay, dir, iter - 1);
//     arg_list.append(
//             aprop == prop_kind::forward ? states_t1_l : states_tm1_l, ws_dt);

//     auto c_states_t_l = workspace.c_states(lay, dir, iter);
//     auto c_states_tm1_l = workspace.c_states(lay, dir, iter - 1);
//     arg_list.append(c_states_t_l, pd()->ocl_conf.aux_dt);
//     arg_list.append(c_states_tm1_l, pd()->ocl_conf.aux_dt);

//     auto gates = workspace.gates(lay, dir, iter);
//     arg_list.append(gates, pd()->ocl_conf.aux_dt);

//     auto ws_grid = workspace.grid_comp(lay, dir, iter);
//     arg_list.append(ws_grid, pd()->ocl_conf.aux_dt);

//     arg_list.append(into<int32_t>(pd()->rnn_conf.states_ws_ld));
//     arg_list.append(into<int32_t>(pd()->rnn_conf.gates_ws_ld));
//     if (aprop == prop_kind::forward) {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
//     } else {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_gates_ld));
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_gates_ld));
//     }
//     arg_list.append(into<int32_t>(batch));
//     arg_list.append(into<int32_t>(dhc));
//     if (aprop == dnnl_backward) {
//         arg_list.append(into<int32_t>(pd()->rnn_conf.scratch_diff_states_ld));
//         arg_list.append(into<int32_t>(diff_states_layer_ld));
//     }

//     if (aprop == dnnl_forward) { arg_list.append(states_tm1_l, ws_dt); }
//     arg_list.append(part);
//     if (aprop != dnnl_forward) {
//         auto diff_dt = pd()->ocl_conf.diff_dt;
//         arg_list.append(scratch_cell);
//         arg_list.append(scratch_dhG1, diff_dt);
//         arg_list.append(scratch_diff_states, diff_dt);
//         arg_list.append(scratch_diff_states_iter, diff_dt);
//         arg_list.append(scratch_diff_states_layer, diff_dt);
//         arg_list.append(diff_bias);
//         arg_list.append(pd()->off.diff_bias);
//     }
//     return parallel_for(ctx, nd_range, kernel, arg_list.args);
// }
// template elemwise_sig_gru(ref_rnn_fwd_t::gru_elemwise);
// template elemwise_sig_gru(ref_rnn_bwd_t::gru_elemwise);
} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl
