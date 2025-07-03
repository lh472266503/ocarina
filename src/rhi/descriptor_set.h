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

}// namespace ocarina