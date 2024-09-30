/*******************************************************************************
* Copyright 2022-2024 Intel Corporation
* Copyright 2022 Codeplay Software Limited
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

#ifndef XPU_SYCL_MEMORY_STORAGE_HELPER_HPP
#define XPU_SYCL_MEMORY_STORAGE_HELPER_HPP

#include <optional>
#include "xpu/sycl/memory_storage.hpp"

#ifdef DNNL_SYCL_CUDA
#include "gpu/nvidia/sycl_cuda_compat.hpp"
#endif

#ifdef DNNL_SYCL_HIP
#include "gpu/amd/sycl_hip_compat.hpp"
#endif

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

#define CTX_IN_SYCL_MEMORY(arg) \
    dnnl::impl::xpu::sycl::interop_memory_arg_t<::sycl::access::mode::read>( \
            &CTX_IN_STORAGE(arg), cgh)

#define CTX_OUT_SYCL_MEMORY(arg) \
    dnnl::impl::xpu::sycl::interop_memory_arg_t<::sycl::access::mode::write>( \
            &CTX_OUT_STORAGE(arg), cgh)

#define CTX_SCRATCH_SYCL_MEMORY(arg) \
    dnnl::impl::xpu::sycl::interop_memory_arg_t< \
            ::sycl::access::mode::read_write>( \
            ctx.get_scratchpad_grantor().get_memory_storage(arg).get(), cgh)

template <::sycl::access_mode mode>
class interop_memory_arg_t {
#if defined(DNNL_SYCL_CUDA)
    static constexpr auto be = ::sycl::backend::ext_oneapi_cuda;
#elif defined(DNNL_SYCL_HIP)
    static constexpr auto be = ::sycl::backend::ext_oneapi_hip;
#else
    static_assert(false,
            "This file is not expected to be used for Intel GPU vendor.");
#endif

#define DUMP_CUDA_TENSOR(dev_ptr, size, datatype) \
    { \
        std::vector<datatype> host_ctr(size); \
        cudaMemcpy(host_ctr.data(), dev_ptr, size * sizeof(datatype), \
                cudaMemcpyDeviceToHost); \
        cudaDeviceSynchronize(); \
        std::cout << #dev_ptr << "\n"; \
        for (auto i = 0; i < size; i++) { \
            std::cout << static_cast<datatype>(host_ctr[i]) << ", "; \
            if ((i + 1) % 32 == 0) std::cout << std::endl; \
        } \
        std::cout << "\n\n"; \
    }

#define DUMP_SYCL_CUDA_TENSOR(mem, size, datatype) \
    { \
        auto usm_ptr = utils::downcast<const usm_memory_storage_t *>(mem)->usm_ptr(); \
        stream_t *stream = nullptr; \
        mem->engine()->get_service_stream(stream); \
        ::sycl::queue sycl_queue = *utils::downcast<xpu::sycl::stream_impl_t *>(stream->impl())->queue(); \
        void *host_ptr = ::sycl::malloc_host(size*sizeof(datatype), sycl_queue.get_context()); \
        sycl_queue.wait_and_throw(); \
        sycl_queue.memcpy(host_ptr, usm_ptr, size*sizeof(datatype)).wait(); \
        std::cout << #mem << "\n"; \
        for (auto i = 0; i < size; i++) { \
            std::cout << static_cast<datatype>(static_cast<datatype *>(host_ptr)[i]) << ", "; \
            if ((i + 1) % 32 == 0) std::cout << std::endl; \
        } \
        ::sycl::free(host_ptr, sycl_queue.get_context()); \
        std::cout << "\n\n"; \
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


public:
    interop_memory_arg_t() = default;
    interop_memory_arg_t(memory_storage_t *raw_mem, ::sycl::handler &cgh) {
        if (!raw_mem || raw_mem->is_null()) { return; }
        auto *mem = static_cast<memory_storage_base_t *>(raw_mem);
        PRINT_VEC((*raw_mem), 16)
        dim_t offset = mem->offset();
        PRINT_VEC((*mem), 16)
        switch (mem->memory_kind()) {
            case sycl::memory_kind::buffer: {
                auto *buffer_storage
                        = utils::downcast<buffer_memory_storage_t *>(mem);
                PRINT_VEC((*buffer_storage), 16)
                acc_.emplace(buffer_storage->buffer(), cgh);
                PRINT_VEC((*buffer_storage), 16)

                offset_ = buffer_storage->base_offset() + offset; //  + buffer_storage->base_offset();
                std::cout << "================ Interop Memory Op (buffer) "
                             "===================\n";
                std::cout << "Offset: " << offset_ << "\n";
                break;
            }
            case sycl::memory_kind::usm: {
                DUMP_SYCL_CUDA_TENSOR(mem, 16, float)
                raw_ptr_ = utils::downcast<const usm_memory_storage_t *>(mem)
                                   ->usm_ptr();
                PRINT_VEC((*mem), 16)
                DUMP_CUDA_TENSOR(raw_ptr_, 16, float)
                DUMP_SYCL_CUDA_TENSOR(mem, 16, float)
                offset_ = offset;
                std::cout << "================ Interop Memory Op (usm)"
                             "===================\n";
                std::cout << "Offset: " << offset_ << "\n";
                break;
            }
            default: assert(!"unexpected memory kind");
        }
    }

    interop_memory_arg_t(::sycl::buffer<uint8_t> buf, ::sycl::handler &cgh,
            size_t offset = 0)
        : offset_ {offset} {
        acc_.emplace(buf, cgh);
    }

    template <typename T = void>
    T *get_native_pointer(
#ifdef DNNL_SYCL_CUDA
            const gpu::nvidia::compat::interop_handle
#endif
#ifdef DNNL_SYCL_HIP
            const gpu::amd::compat::interop_handle
#endif
                    &ih) const {
        void *raw_ptr = nullptr;

        std::cout << "================ Interop Memory Op (get_native_pointer) "
                     "===================\n";
        std::cout << "Offset: " << offset_ << "\n";
        if (acc_.has_value()) {
            DUMP_CUDA_TENSOR(reinterpret_cast<uint8_t *>(ih.get_native_mem<be>(acc_.value())), 16, float)
            raw_ptr = reinterpret_cast<T *>(
                    reinterpret_cast<uint8_t *>(
                            ih.get_native_mem<be>(acc_.value()))
                    + offset_);
        } else {
            DUMP_CUDA_TENSOR(raw_ptr_, 16, float)
            raw_ptr =  reinterpret_cast<T *>(
                    reinterpret_cast<uint8_t *>(raw_ptr_) + offset_);
        }
        DUMP_CUDA_TENSOR(raw_ptr, 16, float)
        return reinterpret_cast<T *>(raw_ptr);
    }

    bool empty() const { return !raw_ptr_ && !acc_.has_value(); }

private:
    void *raw_ptr_ = nullptr;
    std::optional<::sycl::accessor<uint8_t, 1, mode>> acc_;
    size_t offset_=0;
};

#undef DUMP_CUDA_TENSOR

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
