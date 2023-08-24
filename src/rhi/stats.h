//
// Created by Zero on 2023/8/23.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

class MemoryStats {
private:
    static MemoryStats *s_stats;
    MemoryStats() = default;
    MemoryStats(const MemoryStats &) = delete;
    MemoryStats(MemoryStats &&) = delete;
    MemoryStats operator=(const MemoryStats &) = delete;
    MemoryStats operator=(MemoryStats &&) = delete;

private:
    size_t _buffer_size;
    size_t _tex_size;
    map<handle_ty, pair<string, size_t>> _buffer_map;
    map<handle_ty, pair<string, size_t>> _tex_map;

public:
    static MemoryStats &instance();
    static void destroy_instance();
    void on_buffer_allocate(handle_ty handle, size_t size, string name = "");
    void on_buffer_free(handle_ty handle);
    void on_tex_allocate(handle_ty handle, size_t size, string name = "");
    void on_tex_free(handle_ty handle);
};

}// namespace ocarina