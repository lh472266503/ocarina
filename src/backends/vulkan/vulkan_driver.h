#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include <vulkan/vulkan.h>

namespace ocarina {
class VulkanPipelineManager;
class VulkanPipeline;
class FileManager;
class VulkanDevice;
class VulkanShaderManager;
struct InstanceCreation;

class VulkanDriver : public concepts::Noncopyable {
public:
    ~VulkanDriver();
    static VulkanDriver& instance()
    {
        static VulkanDriver s_instance;
        return s_instance;
    }
    VulkanDevice *create_device(FileManager *file_manager, const InstanceCreation &instance_creation);
    void bind_pipeline(const VulkanPipeline &pipeline);
    void terminate();

private:
    VulkanDriver();
    VulkanDevice *vulkan_device_;
    std::unique_ptr<VulkanPipelineManager> vulkan_pipeline_manager;
    std::unique_ptr<VulkanShaderManager> vulkan_shader_manager;
};
}// namespace ocarina