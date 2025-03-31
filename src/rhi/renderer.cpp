//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "renderer.h"


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

}// namespace ocarina