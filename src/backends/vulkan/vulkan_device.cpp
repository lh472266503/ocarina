//
// Created by Zero on 06/06/2022.
//

#include "vulkan_device.h"
#include "util/file_manager.h"
#include "util.h"
#include "vulkan_shader.h"
#include "vulkan_driver.h"

namespace ocarina {

VulkanDevice::VulkanDevice(FileManager *file_manager, const ocarina::InstanceCreation &instance_creation)
    : Device::Impl(file_manager), m_instance(instance_creation), m_windowHandle(instance_creation.windowHandle) {

    init_vulkan();
}

VulkanDevice::~VulkanDevice()
{
    VulkanDriver::instance().terminate();
    shutdown();
}

void VulkanDevice::init_hardware_info() {

}

handle_ty VulkanDevice::create_buffer(size_t size, const string &desc) noexcept {
    return 0;
}

namespace detail {
void context_log_cb(unsigned int level, const char *tag, const char *message, void * /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
}
}// namespace detail


handle_ty VulkanDevice::create_stream() noexcept {
    return 0;
}

handle_ty VulkanDevice::create_texture(uint3 res, PixelStorage pixel_storage,
                                     uint level_num,
                                     const string &desc) noexcept {
    return 0;
}

handle_ty VulkanDevice::create_shader(const Function &function) noexcept {
    return 0;
}

handle_ty VulkanDevice::create_shader_from_file(const std::string &file_name, ShaderType shader_type) noexcept {
    //VulkanShader *shader = VulkanShader::create_from_HLSL(this, shader_type, file_name, "main");
    //if (shader)
    //{
    //    return (handle_ty)shader->shader_module();
    //}


    return InvalidUI64;
}

handle_ty VulkanDevice::create_mesh(const MeshParams &params) noexcept {
    return 0;
}

void VulkanDevice::destroy_mesh(handle_ty handle) noexcept {
    
}

handle_ty VulkanDevice::create_bindless_array() noexcept {
    return 0;
}

void VulkanDevice::destroy_bindless_array(handle_ty handle) noexcept {
    
}

void VulkanDevice::register_shared_buffer(void *&shared_handle, ocarina::uint &gl_handle) noexcept {
    
}

void VulkanDevice::register_shared_tex(void *&shared_handle, ocarina::uint &gl_handle) noexcept {
    
}

void VulkanDevice::mapping_shared_buffer(void *&shared_handle, handle_ty &handle) noexcept {
    
}

void VulkanDevice::mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept {
    
}

void VulkanDevice::unmapping_shared(void *&shared_handle) noexcept {
    
}

void VulkanDevice::unregister_shared(void *&shared_handle) noexcept {
    
}

void VulkanDevice::destroy_buffer(handle_ty handle) noexcept {
    
}

void VulkanDevice::destroy_shader(handle_ty handle) noexcept {

}

void VulkanDevice::destroy_texture(handle_ty handle) noexcept {

}

void VulkanDevice::destroy_stream(handle_ty handle) noexcept {

}
handle_ty VulkanDevice::create_accel() noexcept {
    return 0;
}
void VulkanDevice::destroy_accel(handle_ty handle) noexcept {
    
}
CommandVisitor *VulkanDevice::command_visitor() noexcept {
    return nullptr;
}

void VulkanDevice::init_vulkan()
{
    // Physical device
    uint32_t gpuCount = 0;
    // Get number of available physical devices
    VK_CHECK_RESULT(vkEnumeratePhysicalDevices(m_instance.instance(), &gpuCount, nullptr));
    if (gpuCount == 0) {
        OC_ERROR_FORMAT("No device with Vulkan support found", -1);
        return;
    }
    // Enumerate devices
    std::vector<VkPhysicalDevice> physicalDevices(gpuCount);
    VkResult err = vkEnumeratePhysicalDevices(m_instance.instance(), &gpuCount, physicalDevices.data());
    if (err) {
        OC_ERROR_FORMAT("Could not enumerate physical devices : {}\n", errorString(err));
        return;
    }

    // GPU selection
    // Select physical device to be used for the Vulkan example
    // Defaults to the first device unless specified by command line
    uint32_t selectedDevice = 0;


    physicalDevice_ = physicalDevices[selectedDevice];

    // Store properties (including limits), features and memory properties of the physical device (so that examples can check against them)
    vkGetPhysicalDeviceProperties(physicalDevice_, &m_deviceProperties);
    vkGetPhysicalDeviceFeatures(physicalDevice_, &m_deviceFeatures);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &m_deviceMemoryProperties);

    // Get list of supported extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, nullptr);
    if (extCount > 0) {
        std::vector<VkExtensionProperties> extensions(extCount);
        if (vkEnumerateDeviceExtensionProperties(physicalDevice_, nullptr, &extCount, &extensions.front()) == VK_SUCCESS) {
            for (auto ext : extensions) 
            {
                m_supportedExtensions.push_back(ext.extensionName);
            }
        }
    }


    // Retrieve Physical Device's Queue Families
    uint32_t queueFamilyPropertiesCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyPropertiesCount, nullptr);

    queueFamilyProperties_.resize(queueFamilyPropertiesCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice_, &queueFamilyPropertiesCount, queueFamilyProperties_.data());

    static const uint32_t queueGraphicsIndex = uint32_t(QueueType::Graphics);
    static const uint32_t queueCopyIndex = uint32_t(QueueType::Copy);
    static const uint32_t queueComputeIndex = uint32_t(QueueType::Compute);

    uint32_t graphicsFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    uint32_t computeFlags = VK_QUEUE_COMPUTE_BIT;
    uint32_t copyFlags = VK_QUEUE_TRANSFER_BIT;

    queueFamilyIndices_[queueGraphicsIndex] = getQueueFamilyIndex(graphicsFlags);
    queueFamilyIndices_[queueCopyIndex] = getQueueFamilyIndex(copyFlags);
    queueFamilyIndices_[queueComputeIndex] = getQueueFamilyIndex(computeFlags);

    get_enable_features();
    get_enable_extentions();

    create_logical_device();

    m_swapChain.create_surface(m_instance.instance(), m_windowHandle);
    SwapChainCreation swapchain_creation{};
    m_swapChain.create_swapchain(swapchain_creation, this);
}

void VulkanDevice::create_logical_device()
{

    VkDeviceQueueCreateInfo queues[uint32_t(QueueType::NumQueueType)];
    memset(queues, 0, sizeof(VkDeviceQueueCreateInfo) * uint32_t(QueueType::NumQueueType));
    const float defaultQueuePriority(0.0f);
    for (int i = 0; i < uint32_t(QueueType::NumQueueType); ++i)
    {
        queues[i].queueFamilyIndex = queueFamilyIndices_[i];
        queues[i].queueCount = 1;
        queues[i].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queues[i].pQueuePriorities = &defaultQueuePriority;
    }
    

    VkDeviceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    info.queueCreateInfoCount = uint32_t(QueueType::NumQueueType);
    info.pQueueCreateInfos = queues;
    info.enabledLayerCount = 0;
    info.ppEnabledLayerNames = nullptr;
    info.enabledExtensionCount = m_enableExtensions.size();
    info.ppEnabledExtensionNames = m_enableExtensions.data();
    info.pEnabledFeatures = &m_enabledFeatures;
    //VkPhysicalDeviceFeatures physicalDeviceFeatures = m_Adapter.GetVulkanPhysicalDeviceFeatures();

    VkResult result = vkCreateDevice(physicalDevice_, &info, nullptr, &logicalDevice_);
    
}

void VulkanDevice::get_enable_features() {
    
}

void VulkanDevice::get_enable_extentions()
{
    for (auto &extension : m_supportedExtensions) {
        // Swap chain extension - required
        if (!strcmp(VK_KHR_SWAPCHAIN_EXTENSION_NAME, extension.c_str())) {
            m_enableExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        }
    }
}

void VulkanDevice::shutdown()
{
    m_swapChain.release();
    vkDestroyDevice(logicalDevice_, nullptr);
}

uint32_t VulkanDevice::getQueueFamilyIndex(uint32_t queueFlags) const {

    if ((queueFlags & VK_QUEUE_COMPUTE_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties_.size()); i++) {
            if ((queueFamilyProperties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && ((queueFamilyProperties_[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0)) {
                return i;
            }
        }
    }

    // Dedicated queue for transfer
    // Try to find a queue family index that supports transfer but not graphics and compute
    if ((queueFlags & VK_QUEUE_TRANSFER_BIT) == queueFlags) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties_.size()); i++) {
            if ((queueFamilyProperties_[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && ((queueFamilyProperties_[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) && ((queueFamilyProperties_[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0)) {
                return i;
            }
        }
    }

    for (uint32_t i = 0; i < queueFamilyProperties_.size(); i++) {
        if ((queueFamilyProperties_[i].queueFlags & queueFlags) == queueFlags)
        {
            return i;
        }
    }

    return InvalidUI32;
}

}// namespace ocarina

OC_EXPORT_API ocarina::VulkanDevice *create_device(ocarina::FileManager *file_manager, const ocarina::InstanceCreation& instance_creation) {
    //return ocarina::new_with_allocator<ocarina::VulkanDevice>(file_manager, instance_creation);
    return ocarina::VulkanDriver::instance().create_device(file_manager, instance_creation);
}

OC_EXPORT_API void destroy(ocarina::VulkanDevice *device) {
    ocarina::delete_with_allocator(device);
}

