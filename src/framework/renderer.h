//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "core/thread_pool.h"
#include "rhi/params.h"
#include "rhi/graphics_descriptions.h"

namespace ocarina {
class Primitive;

class Renderer : public concepts::Noncopyable {
public:
    Renderer() = default;
    ~Renderer();

    using UpdateFrameCallBack = ocarina::function<void(double)>;
    using RenderCallback = ocarina::function<void(double)>;
    using SetupCallback = ocarina::function<void()>;
    using ReleaseCallback = ocarina::function<void()>;
    using UpdateDescriptorPerObjectCallback = ocarina::function<void(Primitive&)>;

    void set_update_frame_callback(UpdateFrameCallBack cb)
    {
        update_frame = cb;
    }
    void set_render_callback(RenderCallback cb);
    void set_setup_callback(SetupCallback cb);
    void set_release_callback(ReleaseCallback cb);
    void set_clear_color(const float4& color)
    {
        clear_color = color;
    }

    void render_frame();
    void add_opaque_primitive(Primitive* primitive)
    {
        opaque_primitives_.push_back(primitive);
    }

    void add_transparent_primitive(Primitive* primitive)
    {
        transparent_primitives_.push_back(primitive);
    }

private:
    UpdateFrameCallBack update_frame = nullptr;
    SetupCallback setup = nullptr;
    RenderCallback render = nullptr;
    ReleaseCallback release = nullptr;
    float4 clear_color = {0, 0, 0, 1};

protected:
    std::list<Primitive*> opaque_primitives_;
    std::list<Primitive*> transparent_primitives_;
};

}// namespace ocarina