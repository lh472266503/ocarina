//
// Created by Zero on 2023/8/23.
//

#pragma once

#include "core/stl.h"

namespace ocarina {

class Stats {
private:
    static Stats *s_stats;
    Stats() = default;
    Stats(const Stats &) = delete;
    Stats(Stats &&) = delete;
    Stats operator=(const Stats &) = delete;
    Stats operator=(Stats &&) = delete;

private:
    size_t _buffer_size;
    size_t _tex_size;
    map<handle_ty, pair<string, size_t>> _buffer_map;
    map<handle_ty, pair<string, size_t>> _tex_map;

public:
    static Stats &instance();
    static void destroy_instance();
    void on_buffer_allocate(handle_ty handle, size_t size, string name = "");
    void on_buffer_free(handle_ty handle);
    void on_tex_allocate(handle_ty handle, size_t size, string name = "");
    void on_tex_free(handle_ty handle);
};

}// namespace ocarina