/*******************************************************************************
* Copyright 2020-2024 Intel Corporation
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

// #include "gpu/intel/ocl/gemm/ref_gemm.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/nvidia/cudnn_gemm.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_gemm_t::execute(const intel::gemm_exec_ctx_t &ctx) const {
    exec_args_t mm_args;
    memory_t a(ctx.stream()->engine(), pd()->mm_pd_->src_md(0),
                ctx.args().a->clone());
    memory_t b(ctx.stream()->engine(), pd()->mm_pd_->src_md(1),
                ctx.args().b->clone());
    memory_t c(ctx.stream()->engine(), pd()->mm_pd_->dst_md(),
                ctx.args().c->clone());
    memory_t bias(ctx.stream()->engine(), pd()->mm_pd_->src_md(2),
                ctx.args().bias->clone());

    mm_args[DNNL_ARG_SRC] = {&b, true};
    mm_args[DNNL_ARG_WEIGHTS] = {&a, true};
    mm_args[DNNL_ARG_DST] = {&c, false};
    mm_args[DNNL_ARG_BIAS] = {&bias, true};

    auto mm_exec_ctx = ctx.into_exec_ctx_t(std::move(mm_args));
    auto status = matmul_->execute(mm_exec_ctx);
//     if (exec_d->batch() == 0 || exec_d->n() == 0) return status::success;

//     dim_t off_a0 = a.offset() / types::data_type_size(exec_d->a_type());
//     dim_t off_b0 = b.offset() / types::data_type_size(exec_d->b_type());
//     dim_t off_c0 = c.offset() / types::data_type_size(exec_d->c_type());
//     dim_t off_bias0 = pd()->with_bias()
//             ? bias.offset() / types::data_type_size(exec_d->bias_type())
//             : 0;

//     const auto &scales = memory_storage_t::empty_storage();
//     const auto &a0 = GEMM_CTX_ARG_STORAGE(a_zero_point);
//     const auto &b0 = GEMM_CTX_ARG_STORAGE(b_zero_point);
//     const auto &c0 = GEMM_CTX_ARG_STORAGE(c_zero_point);

//     int c0_mask = 0;
//     CHECK(pd()->attr()->zero_points_.get(DNNL_ARG_C, &c0_mask));

//     const dim_t MB = exec_d->batch();
//     const dim_t M = exec_d->m();
//     const dim_t N = exec_d->n();
//     const dim_t K = exec_d->k();
//     const dim_t stride_a = exec_d->stride_a();
//     const dim_t stride_b = exec_d->stride_b();
//     const dim_t stride_c = exec_d->stride_c();
//     const dim_t lda = exec_d->lda();
//     const dim_t ldb = exec_d->ldb();
//     const dim_t ldc = exec_d->ldc();

//     const dim_t scale_stride = 1;
//     const float eltwise_alpha = pd()->attr_info.eltwise_alpha;
//     const float eltwise_beta = pd()->attr_info.eltwise_beta;
//     const float eltwise_scale = pd()->attr_info.eltwise_scale;
//     const int bias_mask = exec_d->bias_mask();
//     const float beta = pd()->attr_info.sum_scale;

//     const int tra = exec_d->transa() == transpose::trans;
//     const int trb = exec_d->transb() == transpose::trans;

//     const compute::range_t gws = {1, (size_t)N, (size_t)MB};
//     const auto nd_range = compute::nd_range_t(gws);

//     status_t status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
