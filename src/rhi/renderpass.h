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
struct PipelineState;
class IndexBuffer;

struct DrawCallItem {
    PipelineState* pipeline_state = nullptr;
    IndexBuffer* index_buffer = nullptr;
};

class RenderPass {
public:
    RenderPass() = default;
    virtual ~RenderPass(){};

    void clear_draw_call_items() {
        draw_call_items_.clear();
    }

    void add_draw_call(DrawCallItem&& item) {
        draw_call_items_.emplace_back(std::move(item));
    }

    virtual void begin_render_pass() = 0;
    virtual void end_render_pass() = 0;
protected:
    std::list<DrawCallItem> draw_call_items_;
    float4 viewport_ = {0, 0, 0, 0};
    int4 scissor_ = {0, 0, 0, 0};
};

}// namespace ocarina