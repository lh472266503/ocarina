//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "dsl/rtx_type.h"
#include "core/image_base.h"
#include "core/concepts.h"
#include "core/thread_pool.h"
#include "params.h"
#include "graphics_descriptions.h"

namespace ocarina {


class Renderer : public concepts::Noncopyable {
public:
    Renderer() = default;
    ~Renderer();

    using RenderCallback = ocarina::function<void(double)>;
    using SetupCallback = ocarina::function<void()>;
    using ReleaseCallback = ocarina::function<void()>;

    void set_render_callback(RenderCallback cb);
    void set_setup_callback(SetupCallback cb);
    void set_release_callback(ReleaseCallback cb);
    void set_clear_color(const float4& color)
    {
        clear_color = color;
    }
private:
    SetupCallback setup;
    RenderCallback render;
    ReleaseCallback release;
    float4 clear_color = {0, 0, 0, 1};
};

}// namespace ocarina