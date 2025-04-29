//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "index_buffer.h"
#include "device.h"

namespace ocarina {
IndexBuffer::~IndexBuffer() {
    release_index_buffer(device_, this);
}

IndexBuffer *IndexBuffer::create_index_buffer(Device::Impl *device, void *initial_data, uint32_t bytes) {
    IndexBuffer *buffer = device->create_index_buffer(initial_data, bytes);
    buffer->device_ = device;
    return buffer;
}

void IndexBuffer::release_index_buffer(Device::Impl *device, IndexBuffer *index_buffer) {
    device->destroy_buffer(index_buffer->buffer_handle_);
}

}// namespace ocarina