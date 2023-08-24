//
// Created by Zero on 2023/8/23.
//

#include "core/stl.h"
#include "stats.h"

namespace ocarina {

MemoryStats *MemoryStats::s_stats = nullptr;

MemoryStats &MemoryStats::instance() {
    if (s_stats == nullptr) {
        s_stats = new MemoryStats();
    }
    return *s_stats;
}

void MemoryStats::destroy_instance() {
    if (s_stats) {
        delete s_stats;
        s_stats = nullptr;
    }
}

void MemoryStats::on_buffer_allocate(ocarina::handle_ty handle, size_t size, std::string name) {
    with_lock([&] {
        _buffer_size += size;
        _buffer_map.insert(make_pair(handle, BufferData{name, size}));
    });
}

void MemoryStats::on_buffer_free(ocarina::handle_ty handle) {
    with_lock([&] {
        _buffer_size -= buffer_size(handle);
        _buffer_map.erase(handle);
    });
}

void MemoryStats::on_tex_allocate(ocarina::handle_ty handle, uint2 res, std::string name) {
    with_lock([&] {
        _tex_size += res.x * res.y;
        _tex_map.insert(make_pair(handle, TexData{name, res}));
    });
}

void MemoryStats::on_tex_free(ocarina::handle_ty handle) {
    with_lock([&] {
        uint2 res = tex_res(handle);
        _tex_size -= res.x * res.y;
        _tex_map.erase(handle);
    });
}

}// namespace ocarina
