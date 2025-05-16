//
// Created by Zero on 06/08/2022.
//

#pragma once

#include "vulkan_renderpass.h"
#include "vulkan_device.h"
#include "util.h"
#include "vulkan_driver.h"
#include "rhi/pipeline_state.h"
#include "vulkan_pipeline.h"

namespace ocarina {

VulkanRenderPass::VulkanRenderPass(VulkanDevice *device, const RenderPassCreation &render_pass_creation) : RenderPass(render_pass_creation), device_(device) {

}

VulkanRenderPass::~VulkanRenderPass() {
    if (render_pass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(device_->logicalDevice(), render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderPass::begin_render_pass() {
    // Implementation for beginning the render pass
    VulkanDriver& driver = VulkanDriver::instance();
    VkCommandBuffer current_buffer = driver.get_current_command_buffer();

    VkRenderPassBeginInfo renderPassBeginInfo{};
    renderPassBeginInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassBeginInfo.renderPass = render_pass_;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = size_.x;
    renderPassBeginInfo.renderArea.extent.height = size_.y;
    renderPassBeginInfo.clearValueCount = is_use_swapchain_framebuffer() ? 2 : render_target_count_ + 1;
    renderPassBeginInfo.pClearValues = clear_values;

    vkCmdBeginRenderPass(current_buffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport = {
        .x = viewport_.x,
        .y = viewport_.y,
        .width = viewport_.z,
        .height = viewport_.w,
        .minDepth = 0.0f,
        .maxDepth = 1.0f};
    vkCmdSetViewport(current_buffer, 0, 1, &viewport);

    VkRect2D scissor;
    scissor.offset = {scissor_.x, scissor_.y};
    scissor.extent = {(uint32_t)scissor_.z, (uint32_t)scissor_.w};
    vkCmdSetScissor(current_buffer, 0, 1, &scissor);
}

void VulkanRenderPass::end_render_pass() {
    // Implementation for ending the render pass    

    vkCmdEndRenderPass(VulkanDriver::instance().get_current_command_buffer());
}

void VulkanRenderPass::draw_items() {
    VulkanDriver& driver = VulkanDriver::instance();
    for (const auto& item : draw_call_items_) {
        VulkanPipeline pipeline = driver.get_pipeline(*item.pipeline_state);
        driver.bind_pipeline(pipeline);
    }
}

void VulkanRenderPass::setup_render_pass() {
    if (render_pass_ != VK_NULL_HANDLE) {
        return;
    }

    VkDevice device = device_->logicalDevice();

    if (render_target_count_ == 0)
    {
        //swap chain frame buffer as target
        VulkanSwapchain *swapChain = device_->get_swapchain();
        size_ = swapChain->resolution();
        scissor_ = {0, 0, (int)size_.x, (int)size_.y};
        viewport_ = {0, 0, (float)size_.x, (float)size_.y};
        std::array<VkAttachmentDescription, 2> attachments = {};
        // Color attachment
        attachments[0].format = swapChain->color_format();
        attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        // Depth attachment
        attachments[1].format = swapChain->depth_format();
        attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        attachments[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorReference = {};
        colorReference.attachment = 0;
        colorReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthReference = {};
        depthReference.attachment = 1;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpassDescription = {};
        subpassDescription.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassDescription.colorAttachmentCount = 1;
        subpassDescription.pColorAttachments = &colorReference;
        subpassDescription.pDepthStencilAttachment = &depthReference;
        subpassDescription.inputAttachmentCount = 0;
        subpassDescription.pInputAttachments = nullptr;
        subpassDescription.preserveAttachmentCount = 0;
        subpassDescription.pPreserveAttachments = nullptr;
        subpassDescription.pResolveAttachments = nullptr;

        // Subpass dependencies for layout transitions
        std::array<VkSubpassDependency, 2> dependencies;

        dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[0].dstSubpass = 0;
        dependencies[0].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        dependencies[0].dependencyFlags = 0;

        dependencies[1].srcSubpass = VK_SUBPASS_EXTERNAL;
        dependencies[1].dstSubpass = 0;
        dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependencies[1].srcAccessMask = 0;
        dependencies[1].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        dependencies[1].dependencyFlags = 0;

        VkRenderPassCreateInfo renderPassInfo = {};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpassDescription;
        renderPassInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
        renderPassInfo.pDependencies = dependencies.data();

        VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassInfo, nullptr, &render_pass_));
    }
}

}// namespace ocarina