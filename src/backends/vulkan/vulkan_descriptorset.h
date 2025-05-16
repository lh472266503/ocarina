//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "rhi/resources/shader.h"
#include <vulkan/vulkan.h>

namespace ocarina {

class VulkanDevice;
class VulkanShader;

class VulkanDescriptor {
public:

private:
    union DescriptorInfos {
        VkDescriptorBufferInfo buffer_info;
        VkDescriptorImageInfo image_info;
        VkBufferView buffer_view;
    };
    DescriptorInfos info_;
    VkDescriptorType type_;
};

class VulkanDescriptorSet {
private:
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    
public:
    //VulkanDescriptorSet(VulkanDevice *device);
    OC_MAKE_MEMBER_GETTER(descriptor_set, );

    void copy_descriptors(VulkanDescriptor *descriptor);
};



struct VulkanDescriptorSetLayout {
    static constexpr uint8_t MAX_BINDINGS = 16;
    

    DescriptorCount descriptor_count;
    VkDescriptorSetLayout layout_;
};

struct DescriptorPoolCreation {
    uint32_t ubo;
    uint32_t srv;
    uint32_t uav;
    uint32_t samplers;

    bool operator==(DescriptorPoolCreation const &right) const noexcept {
        return ubo == right.ubo && srv == right.srv && uav == right.uav &&
               samplers == right.samplers;
    }

    DescriptorCount to_descriptor_count() const noexcept {
        DescriptorCount count;
        
        count.ubo = this->ubo;
        count.srv = this->srv;
        count.uav = this->uav;
        count.samplers = this->samplers;
        return count;
    }
};

class VulkanDescriptorPool : public concepts::Noncopyable
{
public:
    VulkanDescriptorPool(const DescriptorPoolCreation &creation, VulkanDevice* device);
    ~VulkanDescriptorPool();
    VkDescriptorSet get_descriptor_set(VkDescriptorSetLayout layout);
    OC_MAKE_MEMBER_GETTER(descriptor_pool, );
    OC_MAKE_MEMBER_GETTER(descriptor_pool_creation, );

    bool can_allocate(const DescriptorCount &count) const
    {
        return descriptor_pool_creation_.to_descriptor_count() == count;
    }
private :
    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
    VulkanDevice *device_ = nullptr;
    std::map<VkDescriptorSetLayout, VkDescriptorSet> descriptor_sets;
    DescriptorPoolCreation descriptor_pool_creation_;
};


class VulkanDescriptorManager : public concepts::Noncopyable {
public:
    VulkanDescriptorManager(VulkanDevice *device) : device_(device) {
    }
    ~VulkanDescriptorManager(){};

    VkDescriptorSet get_descriptor_set(const VulkanDescriptorSetLayout &layout, VulkanDevice *device);
    void clear();

    VkDescriptorSetLayout create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count);

private:
    VulkanDevice *device_ = nullptr;
    std::vector<std::unique_ptr<VulkanDescriptorPool>> pools_;
};

}// namespace ocarina