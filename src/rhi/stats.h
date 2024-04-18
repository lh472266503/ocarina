//
// Created by Zero on 2023/8/23.
//

#pragma once

#include "core/stl.h"
#include "core/basic_types.h"
#include "core/thread_safety.h"

namespace ocarina {

class MemoryStats : public thread_safety<> {
    struct BufferData {
        string name;
        size_t size;
    };
    struct TexData {
        string name;
        uint2 res;
    };

private:
    static MemoryStats *s_stats;
    MemoryStats() = default;
    MemoryStats(const MemoryStats &) = delete;
    MemoryStats(MemoryStats &&) = delete;
    MemoryStats operator=(const MemoryStats &) = delete;
    MemoryStats operator=(MemoryStats &&) = delete;

private:
    size_t buffer_size_{};
    size_t tex_size_{};
    map<handle_ty, BufferData> buffer_map_;
    map<handle_ty, TexData> tex_map_;

private:
    [[nodiscard]] size_t buffer_size(handle_ty handle) const noexcept {
        OC_ASSERT(buffer_map_.contains(handle));
        return buffer_map_.at(handle).size;
    }

    [[nodiscard]] uint2 tex_res(handle_ty handle) const noexcept {
        OC_ASSERT(tex_map_.contains(handle));
        return tex_map_.at(handle).res;
    }

public:
    static MemoryStats &instance();
    static void destroy_instance();
    void on_buffer_allocate(handle_ty handle, size_t size, string name = "");
    void on_buffer_free(handle_ty handle);
    void on_tex_allocate(handle_ty handle, uint2 res, string name = "");
    void on_tex_free(handle_ty handle);
    [[nodiscard]] string total_buffer_info() const noexcept;
    [[nodiscard]] string buffer_detail_info() const noexcept;
    [[nodiscard]] string buffer_info() const noexcept;
    [[nodiscard]] string total_tex_info() const noexcept;
    [[nodiscard]] string tex_detail_info() const noexcept;
    [[nodiscard]] string tex_info() const noexcept;
};

}// namespace ocarina