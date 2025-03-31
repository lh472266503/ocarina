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


class RenderPass {
public:
    RenderPass() = default;
    ~RenderPass(){};

public:
    class Impl {
    public:
        virtual ~Impl() = default;
        
    };
};

}// namespace ocarina