//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include <vulkan/vulkan.h>
#include "rhi/graphics_descriptions.h"

namespace ocarina {


struct ShaderReflection{
    enum ResourceType {
        ConstantBuffer,
        Texture,
        SRV,
        UAV,
        Sampler,
        InputAttachment,
        Atomic
    };

    int thread_group_size[3] = {0, 0, 0};

    struct ShaderResource {
        ShaderResource() = default;

        ShaderResource(const ShaderResource &other) {
            shader_type = other.shader_type;
            register_ = other.register_;
            register_count = other.register_count;
            parameter_type = other.parameter_type;
            descriptor_set = other.descriptor_set;
            format = other.format;
            name = other.name;
            location = other.location;
            vertex_attribute_type = other.vertex_attribute_type;
        }

        ShaderResource &operator=(const ShaderResource &other) {
            shader_type = other.shader_type;
            register_ = other.register_;
            register_count = other.register_count;
            parameter_type = other.parameter_type;
            descriptor_set = other.descriptor_set;
            format = other.format;
            name = other.name;
            location = other.location;
            vertex_attribute_type = other.vertex_attribute_type;
            return *this;
        }

        ShaderResource(ShaderResource &&rvalue) noexcept
        {
            shader_type = rvalue.shader_type;
            register_ = rvalue.register_;
            register_count = rvalue.register_count;
            parameter_type = rvalue.parameter_type;
            descriptor_set = rvalue.descriptor_set;
            format = rvalue.format;
            name = std::move(rvalue.name);
            location = rvalue.location;
            vertex_attribute_type = rvalue.vertex_attribute_type;
        }

        ShaderResource& operator=(ShaderResource&& rvalue) noexcept
        {
            shader_type = rvalue.shader_type;
            register_ = rvalue.register_;
            register_count = rvalue.register_count;
            parameter_type = rvalue.parameter_type;
            descriptor_set = rvalue.descriptor_set;
            format = rvalue.format;
            name = std::move(rvalue.name);
            location = rvalue.location;
            vertex_attribute_type = rvalue.vertex_attribute_type;
            return *this;
        }

        uint32_t shader_type : 4 = 0;
        uint32_t register_ : 8 = 0;
        uint32_t register_count : 8 = 0;
        uint32_t location : 4 = 0;
        uint32_t offset : 5 = 0;
        uint32_t parameter_type : 3 = 0;
        uint32_t descriptor_set = 0;
        VkFormat format = VK_FORMAT_UNDEFINED;
        VertexAttributeType::Enum vertex_attribute_type = VertexAttributeType::Enum::Count;

        std::string name;

        // Keep inline with m_Register.
        static const uint32_t s_max_register_size = 1 << 15;
    };

    std::vector<ShaderResource> shader_resources;
    std::vector<ShaderResource> input_layouts;
};
}// namespace ocarina