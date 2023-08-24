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
    size_t _buffer_size{};
    size_t _tex_size{};
    map<handle_ty, BufferData> _buffer_map;
    map<handle_ty, TexData> _tex_map;

private:
    [[nodiscard]] size_t buffer_size(handle_ty handle) const noexcept {
        OC_ASSERT(_buffer_map.contains(handle));
        return _buffer_map.at(handle).size;
    }

    [[nodiscard]] uint2 tex_res(handle_ty handle) const noexcept {
        OC_ASSERT(_tex_map.contains(handle));
        return _tex_map.at(handle).res;
    }

public:
    static MemoryStats &instance();
    static void destroy_instance();
    void on_buffer_allocate(handle_ty handle, size_t size, string name = "");
    void on_buffer_free(handle_ty handle);
    void on_tex_allocate(handle_ty handle, uint2 res, string name = "");
    void on_tex_free(handle_ty handle);
};

}// namespace ocarina