//
// Created by Zero on 2023/8/23.
//

#include "core/stl.h"
#include "stats.h"

namespace ocarina {

MemoryStats *MemoryStats::s_stats = nullptr;

MemoryStats &MemoryStats::instance() {
    if (s_stats) {
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

}

void MemoryStats::on_buffer_free(ocarina::handle_ty handle) {

}

void MemoryStats::on_tex_allocate(ocarina::handle_ty handle, size_t size, std::string name) {

}

void MemoryStats::on_tex_free(ocarina::handle_ty handle) {

}

}// namespace ocarina
