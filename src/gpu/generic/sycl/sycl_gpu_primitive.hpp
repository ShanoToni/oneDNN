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

#ifndef GPU_GENERIC_SYCL_SYCL_GPU_PRIMITIVE_HPP
#define GPU_GENERIC_SYCL_SYCL_GPU_PRIMITIVE_HPP

#include "common/primitive.hpp"

#include "xpu/sycl/memory_storage.hpp"

#include "gpu/generic/sycl/sycl_gpu_kernel.hpp"

#include <tuple>

namespace dnnl {
namespace impl {
namespace gpu {
namespace generic {
namespace sycl {

struct primitive_t : public impl::primitive_t {
    using impl::primitive_t::primitive_t;

protected:
    template <class... SpecConstants, class TupleType = std::tuple<>>
    status_t create_kernel(impl::engine_t *engine, ::sycl::kernel_id kid,
            kernel_t *kernel,
            const TupleType &spec_constants_values = std::tuple<>()) {
        static_assert(is_tuple<TupleType>::value);

        auto ctx = utils::downcast<const xpu::sycl::engine_impl_t *>(
                engine->impl())
                           ->context();
        auto input_bundle
                = ::sycl::get_kernel_bundle<::sycl::bundle_state::input>(
                        ctx, {kid});
        if constexpr (sizeof...(SpecConstants) > 0) {
            apply_specialization_constants<SpecConstants..., TupleType, 0>(
                    spec_constants_values, input_bundle);
        }
        try {
            (*kernel) = kernel_t(::sycl::build(input_bundle));
        } catch (const ::sycl::exception &e) {
            // TODO: what is the error reporting mechanism in oneDNN ?
            return status::
                    unimplemented; // would this be the correct error code to return ?
        }
        return status::success;
    }

    status_t parallel_for(const exec_ctx_t &ctx, const kernel_t &kernel,
            const std::function<void(::sycl::handler &)> &cgf) const {
        return kernel.parallel_for(*ctx.stream(), cgf);
    }

private:
    template <class SpecConstant, class... RemainingSpecConstants,
            typename TupleType, int id>
    void apply_specialization_constants(const TupleType &spec_constants_values,
            ::sycl::kernel_bundle<::sycl::bundle_state::input>
                    &input_kernel_bundle) {
        input_kernel_bundle.template set_specialization_constant<SpecConstant>(
                std::get<id>(spec_constants_values));
        if constexpr (sizeof...(RemainingSpecConstants) != 0) {
            apply_specialization_constants<RemainingSpecConstants..., TupleType,
                    id + 1>(spec_constants_values, input_kernel_bundle);
        }
    }

    template <typename>
    struct is_tuple : std::false_type {};
    template <typename... T>
    struct is_tuple<std::tuple<T...>> : std::true_type {};

    template <typename>
    struct is_specialization_constant : std::false_type {};
    template <typename T>
    struct is_specialization_constant<::sycl::specialization_id<T>>
        : std::true_type {};
};

} // namespace sycl
} // namespace generic
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
