#pragma once
#include "core/header.h"
#include "core/concepts.h"
#include "core/stl.h"
#include "rhi/graphics_descriptions.h"
#include <vulkan/vulkan.h>
namespace ocarina {
class VulkanTexture;
class VulkanDevice;

struct SwapChainBuffer
{
    VkImage image_;
    VkImageView imageView_;
};

class VulkanSwapchain : public concepts::Noncopyable {
public:
    VulkanSwapchain();
    ~VulkanSwapchain();

    void create_swapchain(const SwapChainCreation &creation, VulkanDevice* vulkan_device);
    void release();
    OC_MAKE_MEMBER_GETTER(swapChain, )

    void queue_present();

private:
    void setup_backbuffers(const VkSwapchainCreateInfoKHR &swapChainCreateInfo);
    void release_backbuffers();
    VkPresentModeKHR get_preferred_presentmode(VkPhysicalDevice physicalDevice, VkSurfaceKHR surface, bool vsync);

private:
    VkSwapchainKHR swapChain_ = VK_NULL_HANDLE;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    //VkDevice device_ = VK_NULL_HANDLE;
    VulkanDevice *vulkan_device_;
    //uint32_t imageCount_ = 0;
    std::vector<SwapChainBuffer> backBuffers_;

    // Synchronization semaphores
    struct {
        // Swap chain image presentation
        VkSemaphore presentComplete;
        // Command buffer submission and execution
        VkSemaphore renderComplete;
    } semaphores;
};
}// namespace ocarina