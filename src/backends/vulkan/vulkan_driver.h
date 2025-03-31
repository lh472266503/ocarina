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
class VulkanDescriptorManager;
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
    void render();
    inline VkDevice device() const;

private:
    void setup_frame_buffer();
    void setup_depth_stencil(uint32_t width, uint32_t height);
    void release_frame_buffer();
    void create_command_pool();
    void release_command_pool();
    void create_command_buffers();
    void release_command_buffers();
    void initialize();
    void prepare_frame();
    void window_resize();
private:
    VulkanDriver();
    VulkanDevice *vulkan_device_;
    std::unique_ptr<VulkanPipelineManager> vulkan_pipeline_manager;
    std::unique_ptr<VulkanShaderManager> vulkan_shader_manager;
    std::unique_ptr<VulkanDescriptorManager> vulkan_descriptor_manager;

    VkQueue graphics_queue{VK_NULL_HANDLE};
    VkQueue present_queue{VK_NULL_HANDLE};
    // Contains command buffers and semaphores to be presented to the queue
    VkSubmitInfo submit_info;

    /** @brief Default command pool for the graphics queue family index */
    VkCommandPool command_pool = VK_NULL_HANDLE;
    // Command buffers used for rendering
    std::vector<VkCommandBuffer> draw_cmd_buffers;
    // Active frame buffer index
    uint32_t current_buffer = 0;

    struct {
        VkImage image;
        VkDeviceMemory mem;
        VkImageView view;
    } depth_stencil;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } semaphores;

    VkRenderPass renderpass_framebuffer{VK_NULL_HANDLE};
    VkFormat depth_stencil_format;
    std::vector<VkFramebuffer> frame_buffers;
};
}// namespace ocarina