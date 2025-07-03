#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "rhi/graphics_descriptions.h"
#include <vulkan/vulkan.h>
#include "vulkan_pipeline.h"

namespace ocarina {
class VulkanPipelineManager;
class VulkanPipeline;
class FileManager;
class VulkanDevice;
class VulkanShaderManager;
class VulkanDescriptorManager;
class VulkanShader;
struct InstanceCreation;
struct PipelineState;
class VulkanRenderPass;
class VulkanVertexBuffer;
class VulkanIndexBuffer;
struct VulkanVertexStreamBinding;
class VulkanDescriptorSetLayout;

class VulkanDriver : public concepts::Noncopyable {
public:
    ~VulkanDriver();
    static VulkanDriver& instance()
    {
        static VulkanDriver s_instance;
        return s_instance;
    }
    VulkanDevice *create_device(FileManager *file_manager, const InstanceCreation &instance_creation);
    void bind_pipeline(VkPipelineLayout pipeline_layout, const VulkanPipeline &pipeline);
    void terminate();
    void submit_frame();
    inline VkDevice device() const;
    VulkanShader *create_shader(ShaderType shader_type,
                                const std::string &filename,
                                const std::set<std::string> &options,
                                const std::string &entry_point);
    VulkanShader* get_shader(handle_ty shader) const;
    OC_MAKE_MEMBER_GETTER(current_buffer, )
    VkCommandBuffer get_current_command_buffer() const
    {
        return draw_cmd_buffers[current_buffer_];
    }

    std::tuple<VkPipelineLayout, VulkanPipeline> get_pipeline(const PipelineState &pipeline_state, VkRenderPass render_pass);

    void begin_frame();
    void end_frame();

    VulkanDescriptorSetLayout* create_descriptor_set_layout(VulkanShader **shaders, uint32_t shaders_count);
    VkPipelineLayout get_pipeline_layout(VkDescriptorSetLayout *descriptset_layouts, uint8_t descriptset_layouts_count);

    VulkanRenderPass* create_render_pass(const RenderPassCreation& render_pass_creation);
    void destroy_render_pass(VulkanRenderPass* render_pass);

    VkFramebuffer get_frame_buffer(uint32_t index)
    {
        return frame_buffers[index];
    }

    VkResult copy_buffer(VulkanBuffer* src, VulkanBuffer* dst);
    VkResult copy_buffer(VulkanBuffer *src, VkBuffer dst);

    void set_vertex_buffer(const VulkanVertexStreamBinding& vertex_stream);
    void draw_triangles(VulkanIndexBuffer* index_buffer);
    //VkResult copy_buffer(VulkanBuffer* src, VkBuffer dst);
private:
    void setup_frame_buffer();
    //void setup_depth_stencil(uint32_t width, uint32_t height);
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
    uint32_t current_buffer_ = 0;

    //struct {
    //    VkImage image;
    //    VkDeviceMemory mem;
    //    VkImageView view;
    //} depth_stencil;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } semaphores;

    VkRenderPass renderpass_framebuffer{VK_NULL_HANDLE};
    //VkFormat depth_stencil_format;
    std::vector<VkFramebuffer> frame_buffers;
};
}// namespace ocarina