//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/stl.h"
#include "graphics_descriptions.h"

namespace ocarina {
struct PipelineState;
class IndexBuffer;
class RenderTarget;

struct DrawCallItem {
    PipelineState* pipeline_state = nullptr;
    IndexBuffer* index_buffer = nullptr;
};

class RenderPass {
public:
    RenderPass(const RenderPassCreation &render_pass_creation) {}
    virtual ~RenderPass(){};

    void clear_draw_call_items() {
        draw_call_items_.clear();
    }

    void add_draw_call(DrawCallItem&& item) {
        draw_call_items_.emplace_back(std::move(item));
    }

    void add_render_target(RenderTarget* render_target) {
        OC_ASSERT(render_target_count_ < kMaxRenderTargets);
        render_target_[render_target_count_++] = render_target;
    }

    virtual void begin_render_pass() = 0;
    virtual void end_render_pass() = 0;
    virtual void draw_items() = 0;
protected:
    bool is_use_swapchain_framebuffer() const {
        return render_target_count_ == 0;
    }

    std::list<DrawCallItem> draw_call_items_;
    float4 viewport_ = {0, 0, 0, 0};
    int4 scissor_ = {0, 0, 0, 0};
    uint2 size_ = {0, 0};

    std::string name_ = "RenderPass";

    //if render_target_count == 0, use the swapchain backbuffer as render target
    uint32_t render_target_count_ = 0;
    constexpr static const int kMaxRenderTargets = 8;
    RenderTarget *render_target_[kMaxRenderTargets] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    RenderTarget* depth_stencil_target_ = nullptr;
};

}// namespace ocarina