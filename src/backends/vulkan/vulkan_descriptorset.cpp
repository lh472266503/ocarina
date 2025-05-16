
#include "vulkan_device.h"
#include "vulkan_descriptorset.h"
#include "vulkan_shader.h"
#include "util.h"

namespace ocarina {

VulkanDescriptorPool::VulkanDescriptorPool(const DescriptorPoolCreation &creation, VulkanDevice *device) : device_(device), descriptor_pool_creation_(creation) {
    VkDescriptorPoolSize sizes[4];
    uint8_t npools = 0;

    if (creation.ubo > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = creation.ubo};
    }

    if (creation.srv > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = creation.ubo};
    }

    if (creation.uav > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = creation.uav};
    }

    if (creation.samplers > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = creation.samplers};
    }

    VkDescriptorPoolCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = npools;
    info.pPoolSizes = sizes;
    info.maxSets = 1;

    info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    vkCreateDescriptorPool(device_->logicalDevice(), &info, nullptr, &descriptor_pool_);
}

VulkanDescriptorPool::~VulkanDescriptorPool() {
    descriptor_sets.clear();
    vkDestroyDescriptorPool(device_->logicalDevice(), descriptor_pool_, nullptr);
}

VkDescriptorSet VulkanDescriptorPool::get_descriptor_set(VkDescriptorSetLayout layout) {
    auto it = descriptor_sets.find(layout);
    if (it != descriptor_sets.end())
    {
        return it->second;
    }
    
    // Creating a new set
    VkDescriptorSetLayout layouts[1] = {layout};
    VkDescriptorSetAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = nullptr,
        .descriptorPool = descriptor_pool_,
        .descriptorSetCount = 1,
        .pSetLayouts = layouts,
    };
    VkDescriptorSet vkSet;
    VkResult result = vkAllocateDescriptorSets(device_->logicalDevice(), &allocInfo, &vkSet);
    descriptor_sets.insert(std::make_pair(layout, vkSet));
    return vkSet;
}

VkDescriptorSet VulkanDescriptorManager::get_descriptor_set(const VulkanDescriptorSetLayout &layout, VulkanDevice *device) {
    for (auto &pool : pools_) {
        if (!pool->can_allocate(layout.descriptor_count)) {
            continue;
        }
        else
        {
            return pool->get_descriptor_set(layout.layout_);
        }
    }

    DescriptorPoolCreation pool_creation;
    pool_creation.ubo = layout.descriptor_count.ubo;
    pool_creation.srv = layout.descriptor_count.srv;
    pool_creation.uav = layout.descriptor_count.uav;
    pool_creation.samplers = layout.descriptor_count.samplers;
    pools_.push_back(std::make_unique<VulkanDescriptorPool>(pool_creation, device));
    auto &pool = pools_.back();
    auto ret = pool->get_descriptor_set(layout.layout_);
    return ret;
}

VkDescriptorSetLayout VulkanDescriptorManager::create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count) {
    constexpr uint32_t max_bindings = 256;
    static VkDescriptorSetLayoutBinding bindings[max_bindings];
    VkDescriptorSetLayoutBinding binding;
    uint32_t binding_index = 0;
    for (size_t i = 0; i < shaders_count; i++) {
        VulkanShader* shader = shaders[i];
        
        binding.stageFlags = shader->stage();

        DescriptorCount descriptor_count = shader->descriptor_count();
        for (uint8_t j = 0; j < descriptor_count.ubo; ++j)
        {
            binding.binding = binding_index;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.pImmutableSamplers = nullptr;
            bindings[binding_index++] = binding;
        }

        for (uint8_t j = 0; j < descriptor_count.srv; ++j)
        {
            binding.binding = binding_index;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
            binding.descriptorCount = 1;
            binding.pImmutableSamplers = nullptr;
            bindings[binding_index++] = binding;
        }

        for (uint8_t j = 0; j < descriptor_count.uav; ++j)
        {
            binding.binding = binding_index;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            binding.descriptorCount = 1;
            binding.pImmutableSamplers = nullptr;
            bindings[binding_index++] = binding;
        }

        for (uint8_t j = 0; j < descriptor_count.samplers; ++j)
        {
            binding.binding = binding_index;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
            binding.descriptorCount = 1;
            binding.pImmutableSamplers = nullptr;
            bindings[binding_index++] = binding;
        }
    }

    VkDescriptorSetLayoutCreateInfo info;
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pBindings = bindings;
    info.bindingCount = binding_index;

    VkDescriptorSetLayout descriptorSetLayout;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_->logicalDevice(), &info, nullptr, &descriptorSetLayout));

    return descriptorSetLayout;
}

void VulkanDescriptorManager::clear()
{
    pools_.clear();
}

}// namespace ocarina


