//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"
#include "dsl/rtx_type.h"
#include "core/image_base.h"
#include "core/concepts.h"
#include "core/thread_pool.h"
#include "params.h"
#include "graphics_descriptions.h"
#include "device.h"

namespace ocarina {

using Vector2 = Vector<float, 2>;
using Vector3 = Vector<float, 3>;
using Vector4 = Vector<float, 4>;


struct VertexStream {
    VertexAttributeType::Enum type;
    uint32_t count = 0;  // number of elements in the stream
    uint32_t offset = 0;  // offset in bytes
    uint32_t stride = 0;  // stride in bytes
    void *data = nullptr;  // pointer to the data
    handle_ty buffer_handle = InvalidUI64;  // handle to the buffer

    //get the size of the stream in bytes
    uint32_t get_size() const {
        return count * stride;
    }
};

class VertexBuffer {
public:
    VertexBuffer() = default;
    ~VertexBuffer();

    static VertexBuffer *create_vertex_buffer(Device::Impl *device);
    static void release_vertex_buffer(Device::Impl *device, VertexBuffer *vertex_buffer);

    void add_vertex_stream(VertexAttributeType::Enum type, uint32_t count, uint32_t stride, const void *data);

    //OC_MAKE_MEMBER_GETTER(buffer_handle, );
    OC_MAKE_MEMBER_GETTER(vertex_count, ); 
    OC_MAKE_MEMBER_GETTER(device, );

    Vector3* get_positions() {
        return static_cast<Vector3 *>(vertex_streams_[(uint8_t)VertexAttributeType::Enum::Position].data);
    }

    Vector2* get_uvs() {
        return static_cast<Vector2 *>(vertex_streams_[(uint8_t)VertexAttributeType::Enum::TexCoord0].data);
    }

    Vector3* get_normals() {
        return static_cast<Vector3 *>(vertex_streams_[(uint8_t)VertexAttributeType::Enum::Normal].data);
    }

    VertexStream* get_vertex_stream(VertexAttributeType::Enum attribute_type) {
        if (attribute_type >= VertexAttributeType::Enum::Count) {
            return nullptr;
        }
        return &vertex_streams_[(uint8_t)attribute_type];
    }

protected:

    virtual void upload_data(VertexAttributeType::Enum type) = 0;

    uint32_t vertex_count_ = 0;
    //handle_ty buffer_handle_ = InvalidUI32;
    Device::Impl *device_ = nullptr;

    //std::vector<Vector3> positions_;
    //std::vector<Vector3> normals_;
    //std::vector<Vector2> uvs_;
    //std::vector<Vector4> tangents_;
    //std::vector<Vector4> colors_;

    VertexStream vertex_streams_[(uint8_t)VertexAttributeType::Enum::Count];
};

}// namespace ocarina