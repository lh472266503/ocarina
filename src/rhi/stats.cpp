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
    return ocarina::format("total buffer memory is {} \n", bytes_string(buffer_size_));
}

string MemoryStats::buffer_detail_info() const noexcept {
    string ret;
    for (const auto &item : buffer_map_) {
        const BufferData &data = item.second;
        double percent = double(data.size) / buffer_size_;
        ret += ocarina::format("size {}, percent {:.2f} %, block {}\n", bytes_string(data.size),percent * 100, data.name);
    }
    return ret;
}

string MemoryStats::buffer_info() const noexcept {
    return buffer_detail_info() + total_buffer_info();
}

string MemoryStats::total_tex_info() const noexcept {
    return ocarina::format("total tex memory is {} \n", bytes_string(tex_size_));
}

string MemoryStats::tex_detail_info() const noexcept {
    string ret;
    for (const auto &item : tex_map_) {
        const TexData &data = item.second;
        ret += ocarina::format("res ({}, {}), memory size {}, tex name {}\n",
                               data.res.x, data.res.y,
                               bytes_string(data.res.x * data.res.y),
                               data.name);
    }
    return ret;
}

string MemoryStats::tex_info() const noexcept {
    return tex_detail_info() + total_tex_info();
}

void MemoryStats::on_buffer_allocate(ocarina::handle_ty handle, size_t size, std::string name) {
    with_lock([&] {
        buffer_size_ += size;
        buffer_map_.insert(make_pair(handle, BufferData{name, size}));
    });
}

void MemoryStats::on_buffer_free(ocarina::handle_ty handle) {
    if (!buffer_map_.contains(handle)) {
        return;
    }
    with_lock([&] {
        buffer_size_ -= buffer_size(handle);
        buffer_map_.erase(handle);
    });
}

void MemoryStats::on_tex_allocate(ocarina::handle_ty handle, uint2 res, std::string name) {
    with_lock([&] {
        tex_size_ += res.x * res.y;
        tex_map_.insert(make_pair(handle, TexData{name, res}));
    });
}

void MemoryStats::on_tex_free(ocarina::handle_ty handle) {
    with_lock([&] {
        uint2 res = tex_res(handle);
        tex_size_ -= res.x * res.y;
        tex_map_.erase(handle);
    });
}

}// namespace ocarina
