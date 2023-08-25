//
// Created by Zero on 2023/8/23.
//

#include "core/stl.h"
#include "stats.h"
#include "core/util.h"
#include "core/string_util.h"

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

string MemoryStats::total_buffer_info() const noexcept {
    return ocarina::format("total buffer memory is {} \n", bytes_string(_buffer_size));
}

string MemoryStats::buffer_detail_info() const noexcept {
    string ret;
    for (const auto &item : _buffer_map) {
        const BufferData &data = item.second;
        ret += ocarina::format("size {}, block {}\n", bytes_string(data.size), data.name);
    }
    return ret;
}

string MemoryStats::buffer_info() const noexcept {
    return total_buffer_info() + buffer_detail_info();
}

void MemoryStats::on_buffer_allocate(ocarina::handle_ty handle, size_t size, std::string name) {
    with_lock([&] {
        _buffer_size += size;
        _buffer_map.insert(make_pair(handle, BufferData{name, size}));
    });
}

void MemoryStats::on_buffer_free(ocarina::handle_ty handle) {
    if (!_buffer_map.contains(handle)) {
        return;
    }
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
