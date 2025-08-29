
#include "vulkan_device.h"
#include "vulkan_descriptorset.h"
#include "vulkan_shader.h"
#include "util.h"
#include "vulkan_driver.h"
#include "vulkan_descriptorset_writer.h"

namespace ocarina {

VulkanDescriptorSet::VulkanDescriptorSet(VulkanDevice *device, VulkanDescriptorSetLayout *layout) : layout_(layout), device_(device) {
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = layout->descriptor_pool();
    VkDescriptorSetLayout layouts[1] = {layout->layout()};
    allocInfo.pSetLayouts = layouts;
    allocInfo.descriptorSetCount = 1;
    set_is_global(layout->is_global_ubo());
    // Might want to create a "DescriptorPoolManager" class that handles this case, and builds
    // a new pool whenever an old pool fills up. But this is beyond our current scope
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device->logicalDevice(), &allocInfo, &descriptor_set_));


    writer_ = ocarina::new_with_allocator<VulkanDescriptorSetWriter>(device, this);
}

VulkanDescriptorSet::~VulkanDescriptorSet()
{
    if (writer_) {
        ocarina::delete_with_allocator(writer_);
        writer_ = nullptr;
    }
    vkFreeDescriptorSets(device_->logicalDevice(), layout_->descriptor_pool(), 1, &descriptor_set_);
}

void VulkanDescriptorSet::update_buffer(uint64_t name_id, void *data, uint32_t size) {
    if (writer_) {
        writer_->update_buffer(name_id, data, size);
    }
}

void VulkanDescriptorSet::update_texture(uint64_t name_id, Texture *texture) {
    if (writer_) {
        writer_->update_texture(name_id, texture);
    }
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
    uint32_t size,
    uint32_t count)
{
    char *end;
    uint64_t nameid = hash64(name);

    if (name_to_bindings_.find(nameid) == name_to_bindings_.end())
    {
        name_to_bindings_.insert(std::make_pair(nameid, binding));
        //VkDescriptorSetLayoutBinding descriptor_binding;
        //descriptor_binding.binding = binding;
        //descriptor_binding.descriptorType = descriptor_type;
        //descriptor_binding.stageFlags = stage_flags; 
        //descriptor_binding.descriptorCount = count; 
        VulkanShaderVariableBinding descriptor_binding;
        strcpy(descriptor_binding.name, name);
        descriptor_binding.binding = binding;
        descriptor_binding.type = descriptor_type;
        descriptor_binding.shader_stage = stage_flags;
        descriptor_binding.count = count;
        descriptor_binding.size = size;

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
        VkDescriptorSetLayoutBinding descriptor_binding;
        descriptor_binding.binding = it.second.binding;
        descriptor_binding.descriptorType = it.second.type;
        descriptor_binding.stageFlags = it.second.shader_stage;
        descriptor_binding.descriptorCount = it.second.count; 
        layout_bindings.push_back(descriptor_binding);
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

DescriptorSet *VulkanDescriptorSetLayout::allocate_descriptor_set() {
    //ocarina::make_unique_with_allocator<VulkanDescriptorSet>(device_, descriptor_pool_);
    return ocarina::new_with_allocator<VulkanDescriptorSet>(device_, this);
}

VulkanShaderVariableBinding VulkanDescriptorSetLayout::get_binding(uint64_t binding) {
    auto it = bindings_.find(binding);
    if (it != bindings_.end()) {
        return it->second;
    }
    return {};
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


std::array<DescriptorSetLayout*, MAX_DESCRIPTOR_SETS_PER_SHADER> VulkanDescriptorManager::create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count) {

    std::array<DescriptorSetLayout *, MAX_DESCRIPTOR_SETS_PER_SHADER> descriptor_set_layouts_array = {};

    uint32_t count = 0;

    VulkanDescriptorSetLayout *layout = nullptr;

    DescriptorLayoutKey layout_key;

    for (uint32_t i = 0; i < shaders_count; ++i)
    {
        VulkanShader* shader = shaders[i];

        uint32_t variables_count = shader->get_shader_variables_count();
        for (uint32_t j = 0; j < variables_count; ++j)
        {
            const VulkanShaderVariableBinding& binding = shader->get_shader_variable(j);
            if (binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            {
                //one uniform one descriptorset
                DescriptorLayoutKey ubo_layout_key;
                ubo_layout_key.bindings.push_back(binding);
                auto it = descriptor_set_layouts_.find(ubo_layout_key);
                if (it != descriptor_set_layouts_.end()) {
                    descriptor_set_layouts_array[count++] = it->second;
                }
                else
                {
                    layout = ocarina::new_with_allocator<VulkanDescriptorSetLayout>(device_); 
                    layout->add_binding(binding.name, binding.binding, binding.type, binding.shader_stage, binding.size, binding.count);
                    layout->build_layout();
                    layout->set_name(binding.name);
                    if (strcmp(binding.name, "global_ubo") == 0) {
                        layout->set_is_global_ubo(true);
                        VulkanDriver::instance().add_global_descriptor_set(hash64(binding.name), static_cast<VulkanDescriptorSet*>(layout->allocate_descriptor_set()));
                    }
                    descriptor_set_layouts_array[count++] = layout;
                    descriptor_set_layouts_.insert(std::make_pair(ubo_layout_key, layout));
                    layout = nullptr;
                }
            } 
            else
                layout_key.bindings.push_back(binding);
            //layout->add_binding(binding.name, binding.binding, binding.type, binding.shader_stage, binding.count);
        }
    }

    if (layout_key.bindings.size() > 0) {
        layout_key.Normalize();
        auto it = descriptor_set_layouts_.find(layout_key);
        if (it != descriptor_set_layouts_.end()) {
            descriptor_set_layouts_array[count++] = it->second;
        } else {
            layout = ocarina::new_with_allocator<VulkanDescriptorSetLayout>(device_);
            for (const auto &binding : layout_key.bindings) {
                layout->add_binding(binding.name, binding.binding, binding.type, binding.shader_stage, binding.size, binding.count);
            }
            layout->build_layout();
            descriptor_set_layouts_array[count++] = layout;
            descriptor_set_layouts_.insert(std::make_pair(layout_key, layout));
            layout = nullptr;
        }
    }

    return descriptor_set_layouts_array;
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


