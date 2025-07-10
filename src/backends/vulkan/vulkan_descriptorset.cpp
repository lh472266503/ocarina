
#include "vulkan_device.h"
#include "vulkan_descriptorset.h"
#include "vulkan_shader.h"
#include "util.h"

namespace ocarina {

VulkanDescriptorSet::VulkanDescriptorSet(VulkanDevice *device, VulkanDescriptorSetLayout *layout) : layout_(layout), device_(device) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = layout->descriptor_pool();
    VkDescriptorSetLayout layouts[1] = {layout->layout()};
    allocInfo.pSetLayouts = layouts;
    allocInfo.descriptorSetCount = 1;

    // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
    // a new pool whenever an old pool fills up. But this is beyond our current scope
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice(), &allocInfo, &descriptor_set_));

}

VulkanDescriptorSet::~VulkanDescriptorSet()
{
    vkFreeDescriptorSets(device_->logicalDevice(), layout_->descriptor_pool(), 1, &descriptor_set_);
}

VulkanDescriptorSetLayout::VulkanDescriptorSetLayout(VulkanDevice* device) : device_(device)
{

}

VulkanDescriptorSetLayout::~VulkanDescriptorSetLayout()
{
    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device_->logicalDevice(), descriptor_pool_, nullptr);
    }

    if (layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device_->logicalDevice(), layout_, nullptr);
    }
}

void VulkanDescriptorSetLayout::add_binding(const char* name,
    uint32_t binding,
    VkDescriptorType descriptor_type,
    VkShaderStageFlags stage_flags,
    uint32_t count)
{
    char *end;
    uint32_t nameid = std::strtoll(name, &end, 10);

    if (name_to_bindings_.find(nameid) == name_to_bindings_.end())
    {
        name_to_bindings_.insert(std::make_pair(nameid, binding));
        VkDescriptorSetLayoutBinding descriptor_binding;
        descriptor_binding.binding = binding;
        descriptor_binding.descriptorType = descriptor_type;
        descriptor_binding.stageFlags = stage_flags; 
        descriptor_binding.descriptorCount = count; 

        bindings_.insert(std::make_pair(binding, descriptor_binding));

        switch (descriptor_type)
        {
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
            descriptor_count_.srv++;
            break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
            descriptor_count_.ubo++;
            break;
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
            descriptor_count_.uav++;
            break;
        case VK_DESCRIPTOR_TYPE_SAMPLER:
            descriptor_count_.samplers++;
            break;
        }
    }
}

void VulkanDescriptorSetLayout::build_layout()
{
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    layout_bindings.reserve(bindings_.size());
    for (auto& it : bindings_)
    {
        layout_bindings.push_back(it.second);
    }
    VkDescriptorSetLayoutCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    info.bindingCount = (uint32_t)layout_bindings.size();
    info.pBindings = layout_bindings.data();
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device_->logicalDevice(), &info, nullptr, &layout_));

    VkDescriptorSetLayout layouts[1] = { layout_ };
    VkDescriptorPoolSize sizes[4];
    uint8_t npools = 0;

    if (descriptor_count_.ubo > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = descriptor_count_.ubo };
    }

    if (descriptor_count_.srv > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            .descriptorCount = descriptor_count_.ubo };
    }

    if (descriptor_count_.uav > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = descriptor_count_.uav };
    }

    if (descriptor_count_.samplers > 0)
    {
        sizes[npools++] = {
            .type = VK_DESCRIPTOR_TYPE_SAMPLER,
            .descriptorCount = descriptor_count_.samplers };
    }

    VkDescriptorPoolCreateInfo pool_create_info{};
    pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_create_info.poolSizeCount = npools;
    pool_create_info.pPoolSizes = sizes;
    pool_create_info.maxSets = 1;

    pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    VK_CHECK_RESULT(vkCreateDescriptorPool(device_->logicalDevice(), &pool_create_info, nullptr, &descriptor_pool_));
}

std::unique_ptr<DescriptorSet> VulkanDescriptorSetLayout::allocate_descriptor_set() {
    //ocarina::make_unique_with_allocator<VulkanDescriptorSet>(device_, descriptor_pool_);
    return ocarina::make_unique_with_allocator<VulkanDescriptorSet>(device_, this);
}

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


VulkanDescriptorSetLayout* VulkanDescriptorManager::create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count) {

    uint64_t key = 0;
    uint32_t shiftbit = 32;
    for (uint32_t i = 0; i < shaders_count; ++i)
    {
        uint64_t handle = (uint64_t)shaders[i]->shader_module() << shiftbit;
        key |= handle;
        shiftbit -= 32;
    }

    auto it = descriptor_set_layouts_.find(key);
    if (it != descriptor_set_layouts_.end())
    {
        return it->second;
    }
    VulkanDescriptorSetLayout* layout = ocarina::new_with_allocator<VulkanDescriptorSetLayout>(device_);

    for (uint32_t i = 0; i < shaders_count; ++i)
    {
        VulkanShader* shader = shaders[i];

        uint32_t variables_count = shader->get_shader_variables_count();
        for (uint32_t j = 0; j < variables_count; ++j)
        {
            const VulkanShaderVariableBinding& binding = shader->get_shader_variable(j);
            layout->add_binding(binding.name, binding.binding, binding.type, binding.shader_stage, binding.count);
        }
    }

    layout->build_layout();

    descriptor_set_layouts_.insert(std::make_pair(key, layout));

    return layout;
}

void VulkanDescriptorManager::clear()
{
    //pools_.clear();
    for (auto layout : descriptor_set_layouts_)
    {
        VulkanDescriptorSetLayout* descriptor_set_layout = layout.second;
        ocarina::delete_with_allocator<VulkanDescriptorSetLayout>(descriptor_set_layout);
    }

    descriptor_set_layouts_.clear();
}

}// namespace ocarina


