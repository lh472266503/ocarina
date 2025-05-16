//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "rhi/renderpass.h"
#include "renderer.h"
#include "rhi/device.h"

namespace ocarina {
Renderer::~Renderer() {
    if (release != nullptr) {
        release();
    }
}

void Renderer::set_render_callback(RenderCallback cb) {
    render = cb;
}

void Renderer::set_setup_callback(SetupCallback cb) {
    setup = cb;
}

void Renderer::set_release_callback(ReleaseCallback cb) {
    release = cb;
}

void Renderer::render_frame()
{
    double dt = 0;
    if (update_frame)
        update_frame(dt);
    if (render)
        render(dt);
    else
    {
        device_->begin_frame();
        for (auto& render_pass : render_passes_)
        {
            render_pass->begin_render_pass();
            render_pass->draw_items();
            render_pass->end_render_pass();
        }
        device_->end_frame();
    }
}

}// namespace ocarina