//
// Created by Zero on 2023/8/23.
//

#include "core/stl.h"
#include "stats.h"

namespace ocarina {

Stats *Stats::s_stats = nullptr;

Stats &Stats::instance() {
    if (s_stats) {
        s_stats = new Stats();
    }
    return *s_stats;
}

void Stats::destroy_instance() {
    if (s_stats) {
        delete s_stats;
        s_stats = nullptr;
    }
}

void Stats::on_buffer_allocate(ocarina::handle_ty handle, size_t size, std::string name) {

}

void Stats::on_buffer_free(ocarina::handle_ty handle) {

}

void Stats::on_tex_allocate(ocarina::handle_ty handle, size_t size, std::string name) {

}

void Stats::on_tex_free(ocarina::handle_ty handle) {

}

}// namespace ocarina
