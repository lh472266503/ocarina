//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/graphics_descriptions.h"

namespace ocarina {
class DescriptorSetLayout;


class Material {
public:
    Material() = default;
    ~Material();

private:
    DescriptorSetLayout *descriptor_set_layout_ = nullptr;
};

}// namespace ocarina