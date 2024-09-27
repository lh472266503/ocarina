//
// Created by Zero on 06/06/2022.
//

#include "vulkan_device.h"
#include "util/file_manager.h"
#include "util.h"

namespace ocarina {

VulkanDevice::VulkanDevice(FileManager *file_manager, const ocarina::InstanceCreation &instance_creation)
    : Device::Impl(file_manager), m_instance(instance_creation) {

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
}

void VulkanDevice::shutdown()
{
    vkDestroyDevice(logicalDevice_, nullptr);
}

}// namespace ocarina

OC_EXPORT_API ocarina::VulkanDevice *create(ocarina::FileManager *file_manager, const ocarina::InstanceCreation& instance_creation) {
    return ocarina::new_with_allocator<ocarina::VulkanDevice>(file_manager, instance_creation);
}

OC_EXPORT_API void destroy(ocarina::VulkanDevice *device) {
    ocarina::delete_with_allocator(device);
}

