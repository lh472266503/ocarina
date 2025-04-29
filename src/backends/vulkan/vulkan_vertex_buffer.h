#pragma once
#include "rhi/vertex_buffer.h"
#include <vulkan/vulkan.h>
namespace ocarina {
class VulkanDevice;
class VulkanShader;
class VulkanVertexBuffer;
struct VulkanVertexStreamBinding
{
    std::vector<VkBuffer> buffers_;
    std::vector<uint32_t> offsets_;
    std::vector<VkVertexInputBindingDescription> binding_descriptions_;
    std::vector<VkVertexInputAttributeDescription> attribute_descriptions_;

    static void create_from_vertex_shader(VulkanShader *vertex_shader, VulkanVertexBuffer* vertex_buffer, VulkanVertexStreamBinding& binding);
};

class VulkanVertexBuffer : public VertexBuffer {
public:
    VulkanVertexBuffer(VulkanDevice* device);
    ~VulkanVertexBuffer();
    
    VulkanVertexStreamBinding *get_or_create_vertex_binding(VulkanShader *vertex_shader);

    
protected:
    virtual void upload_data(VertexAttributeType::Enum type) override;

private:
    std::unordered_map<handle_ty, VulkanVertexStreamBinding*> vertex_bindings_;
    //VulkanVertexStreamBinding vulkan_vertex_binding_;
};

}// namespace ocarina