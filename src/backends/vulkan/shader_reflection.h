//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "rhi/resources/shader.h"


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
            name = other.name;
        }

        ShaderResource &operator=(const ShaderResource &other) {
            shader_type = other.shader_type;
            register_ = other.register_;
            register_count = other.register_count;
            parameter_type = other.parameter_type;
            descriptor_set = other.descriptor_set;
            name = other.name;
            return *this;
        }

        ShaderResource(ShaderResource &&rvalue) noexcept
        {
            shader_type = rvalue.shader_type;
            register_ = rvalue.register_;
            register_count = rvalue.register_count;
            parameter_type = rvalue.parameter_type;
            descriptor_set = rvalue.descriptor_set;
            name = std::move(rvalue.name);
        }

        ShaderResource& operator=(ShaderResource&& rvalue) noexcept
        {
            shader_type = rvalue.shader_type;
            register_ = rvalue.register_;
            register_count = rvalue.register_count;
            parameter_type = rvalue.parameter_type;
            descriptor_set = rvalue.descriptor_set;
            name = std::move(rvalue.name);
            return *this;
        }

        uint32_t shader_type : 4 = 0;
        uint32_t register_ : 15 = 0;
        uint32_t register_count : 10 = 0;
        uint32_t parameter_type : 3 = 0;
        uint32_t descriptor_set = 0;

        std::string name;

        // Keep inline with m_Register.
        static const uint32_t s_max_register_size = 1 << 15;
    };

    std::vector<ShaderResource> shader_resources;
    std::vector<ShaderResource> input_layouts;
};
}// namespace ocarina