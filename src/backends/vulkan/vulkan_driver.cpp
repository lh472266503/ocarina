#include "vulkan_driver.h"
#include "vulkan_device.h"
#include "util.h"
#include "vulkan_pipeline.h"
#include "util/file_manager.h"
#include "rhi/graphics_descriptions.h"
#include "vulkan_shader.h"

namespace ocarina {

VulkanDriver::VulkanDriver() {
    vulkan_pipeline_manager = std::make_unique<VulkanPipelineManager>();
    vulkan_shader_manager = std::make_unique<VulkanShaderManager>();
}

VulkanDriver::~VulkanDriver() {
   
}

VulkanDevice* VulkanDriver::create_device(FileManager* file_manager, const InstanceCreation& instance_creation)
{
    vulkan_device_ = ocarina::new_with_allocator<ocarina::VulkanDevice>(file_manager, instance_creation);
    return vulkan_device_;
}

void VulkanDriver::bind_pipeline(const VulkanPipeline& pipeline)
{

}

void VulkanDriver::terminate()
{
    vulkan_pipeline_manager->clear(vulkan_device_);
}

}// namespace ocarina


