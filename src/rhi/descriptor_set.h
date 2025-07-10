//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/concepts.h"



namespace ocarina {

class DescriptorSet : concepts::Noncopyable {
public:
    //DescriptorSet() {}
    virtual ~DescriptorSet() {}
};

class DescriptorSetLayout : concepts::Noncopyable {
public:
    DescriptorSetLayout() {}
    virtual ~DescriptorSetLayout() {}

    virtual std::unique_ptr<DescriptorSet> allocate_descriptor_set() = 0;
};

class DescriptorSetWriter : concepts::Noncopyable {
public:
    DescriptorSetWriter() {}
    virtual ~DescriptorSetWriter() {}
    //virtual void write_descriptor_set(DescriptorSet &descriptor_set) = 0;
    //// Add more methods to set different types of descriptors
    //virtual void set_uniform_buffer(uint32_t binding, const void *data, size_t size) = 0;
    //virtual void set_storage_buffer(uint32_t binding, const void *data, size_t size) = 0;
    //virtual void set_texture(uint32_t binding, const void *texture) = 0;
    //virtual void set_sampler(uint32_t binding, const void *sampler) = 0;
    virtual void bind_buffer(int32_t name_id, handle_ty buffer) = 0;
    virtual void bind_texture(int32_t name_id, handle_ty texture) = 0;

protected:
    unique_ptr<DescriptorSet> descriptor_set_ = nullptr;
    DescriptorSetLayout *descriptor_set_layout_ = nullptr;
};

}// namespace ocarina