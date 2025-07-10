//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "core/stl.h"
#include "core/concepts.h"
#include "rhi/resources/shader.h"
#include "rhi/descriptor_set.h"
#include <vulkan/vulkan.h>

namespace ocarina {

class VulkanDevice;
class VulkanShader;
class VulkanBuffer;
class VulkanDescriptorSet;

class VulkanDescriptor
{
public:
    uint32_t binding = 0;
    std::string name_;
};

class VulkanDescriptorBuffer : public VulkanDescriptor {
public:
    VkDescriptorBufferInfo buffer_info = {};
    VulkanBuffer *buffer_;
    
};

class VulkanDescriptorImage : public VulkanDescriptor{

};


class VulkanDescriptorSetLayout : public DescriptorSetLayout {
    static constexpr uint8_t MAX_BINDINGS = 16;
public:
    VulkanDescriptorSetLayout(VulkanDevice* device);
    ~VulkanDescriptorSetLayout() override;
    void add_binding(const char* name,
        uint32_t binding,
        VkDescriptorType descriptor_type,
        VkShaderStageFlags stage_flags,
        uint32_t count = 1);

    void build_layout();

    DescriptorCount get_descriptor_count() const noexcept {
        return descriptor_count_;
    }

    OC_MAKE_MEMBER_GETTER(layout, );
    OC_MAKE_MEMBER_GETTER(descriptor_pool, );

    std::unique_ptr<DescriptorSet> allocate_descriptor_set() override;
private:
    DescriptorCount descriptor_count_;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;

    std::unordered_map<uint32_t, uint32_t> name_to_bindings_;
    std::unordered_map<uint32_t, VkDescriptorSetLayoutBinding> bindings_;

    VulkanDevice* device_ = nullptr;

    VkDescriptorPool descriptor_pool_ = VK_NULL_HANDLE;
};

class VulkanDescriptorSet : public DescriptorSet {
private:
    VkDescriptorSet descriptor_set_ = VK_NULL_HANDLE;
    VulkanDescriptorSetLayout *layout_ = nullptr;
    VulkanDevice *device_ = nullptr;

public:
    VulkanDescriptorSet(VulkanDevice *device, VulkanDescriptorSetLayout* layout);
    ~VulkanDescriptorSet() override;
    OC_MAKE_MEMBER_GETTER(descriptor_set, );

    //void copy_descriptors(VulkanDescriptor *descriptor);
    
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

    //VkDescriptorSet get_descriptor_set(const VulkanDescriptorSetLayout &layout, VulkanDevice *device);
    void clear();

    VulkanDescriptorSetLayout* create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count);

private:
    VulkanDevice *device_ = nullptr;
    std::unordered_map<uint64_t, VulkanDescriptorSetLayout*> descriptor_set_layouts_;
};

}// namespace ocarina