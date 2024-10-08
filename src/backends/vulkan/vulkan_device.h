//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "rhi/device.h"
#include <vulkan/vulkan.h>
#include "vulkan_instance.h"
#include "vulkan_swapchain.h"

namespace ocarina {
class VulkanDevice : public Device::Impl {
public:
    static constexpr size_t size(Type::Tag tag) {
        using Tag = Type::Tag;
        switch (tag) {
            case Tag::BUFFER: return sizeof(BufferProxy<>);
            case Tag::BYTE_BUFFER: return sizeof(BufferProxy<>);
            case Tag::ACCEL: return sizeof(handle_ty);
            case Tag::TEXTURE: return sizeof(TextureProxy);
            case Tag::BINDLESS_ARRAY: return sizeof(BindlessArrayProxy);
            default:
                return 0;
        }
    }
    // return size of type on device memory
    static constexpr size_t size(const Type *type) {
        auto ret = size(type->tag());
        return ret == 0 ? type->size() : ret;
    }
    static constexpr size_t alignment(Type::Tag tag) {
        using Tag = Type::Tag;
        switch (tag) {
            case Tag::BUFFER: return alignof(BufferProxy<>);
            case Tag::BYTE_BUFFER: return alignof(BufferProxy<>);
            case Tag::ACCEL: return alignof(handle_ty);
            case Tag::TEXTURE: return alignof(TextureProxy);
            case Tag::BINDLESS_ARRAY: return alignof(BindlessArrayProxy);
            default:
                return 0;
        }
    }
    // return alignment of type on device memory
    static constexpr size_t alignment(const Type *type) {
        auto ret = alignment(type->tag());
        return ret == 0 ? type->alignment() : ret;
    }
    // return the size of max member recursive
    static size_t max_member_size(const Type *type) {
        auto ret = type->max_member_size();
        return ret == 0 ? sizeof(handle_ty) : ret;
    }

private:
    /** @brief Physical device representation */
    VkPhysicalDevice physicalDevice_;
    /** @brief Logical device representation (application's view of the device) */
    VkDevice logicalDevice_;
    /** @brief Properties of the physical device including limits that the application can check against */
    VkPhysicalDeviceProperties m_deviceProperties{};
    /** @brief Features of the physical device that an application can use to check if a feature is supported */
    VkPhysicalDeviceFeatures m_deviceFeatures{};
    /** @brief Features that have been enabled for use on the physical device */
    VkPhysicalDeviceFeatures m_enabledFeatures{};
    /** @brief Memory types and heaps of the physical device */
    VkPhysicalDeviceMemoryProperties m_deviceMemoryProperties{};
    /** @brief Queue family properties of the physical device */
    std::vector<VkQueueFamilyProperties> m_queueFamilyProperties;
    /** @brief List of extensions supported by the device */
    std::vector<std::string> m_supportedExtensions;
    /** @brief Default command pool for the graphics queue family index */
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

public:
    explicit VulkanDevice(FileManager *file_manager, const ocarina::InstanceCreation &instance_creation);
    ~VulkanDevice();
    void init_hardware_info();

    [[nodiscard]] handle_ty create_buffer(size_t size, const string &desc) noexcept override;
    void destroy_buffer(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_texture(uint3 res, PixelStorage pixel_storage,
                                           uint level_num,
                                           const string &desc) noexcept override;
    void destroy_texture(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_shader(const Function &function) noexcept override;
    void destroy_shader(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_accel() noexcept override;
    void destroy_accel(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_stream() noexcept override;
    void destroy_stream(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_mesh(const MeshParams &params) noexcept override;
    void destroy_mesh(handle_ty handle) noexcept override;
    [[nodiscard]] handle_ty create_bindless_array() noexcept override;
    void destroy_bindless_array(handle_ty handle) noexcept override;
    void register_shared_buffer(void *&shared_handle, ocarina::uint &gl_handle) noexcept override;
    void register_shared_tex(void *&shared_handle, ocarina::uint &gl_handle) noexcept override;
    void mapping_shared_buffer(void *&shared_handle,handle_ty &handle) noexcept override;
    void mapping_shared_tex(void *&shared_handle, handle_ty &handle) noexcept override;
    void unmapping_shared(void *&shared_handle) noexcept override;
    void unregister_shared(void *&shared_handle) noexcept override;
    void init_rtx() noexcept override {  }
    [[nodiscard]] CommandVisitor *command_visitor() noexcept override;
    void shutdown();

    OC_MAKE_MEMBER_GETTER_SETTER(logicalDevice, );

    OC_MAKE_MEMBER_GETTER_SETTER(physicalDevice, );

private:
    void init_vulkan();
    void create_logical_device();

    VulkanInstance m_instance;
    VulkanSwapchain m_swapChain;
};
}// namespace ocarina
