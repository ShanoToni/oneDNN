/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
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

#ifndef GPU_INTEL_JIT_PASS_SEND_HPP
#define GPU_INTEL_JIT_PASS_SEND_HPP

#include "gpu/intel/jit/ir/ir.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace jit {

stmt_t lift_buffer_offsets_in_send(const stmt_t &s, ir_context_t &ir_ctx);

stmt_t inject_send(const stmt_t &s, ir_context_t &ir_ctx);

// Lifts loop-invariant header assignments related to block 2D messages.
stmt_t lift_send_2d_header_store(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace jit
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
