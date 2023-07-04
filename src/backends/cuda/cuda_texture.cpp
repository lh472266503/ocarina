//
// Created by Zero on 06/08/2022.
//

#include "cuda_texture.h"
#include "util.h"
#include "cuda_device.h"

namespace ocarina {

CUDATexture::CUDATexture(CUDADevice *device, uint3 res, PixelStorage pixel_storage, uint level_num)
    : _device(device), _res(res), level_num(level_num) {
    _data.pixel_storage = pixel_storage;
    init();
}

void CUDATexture::init() {
    CUDA_ARRAY3D_DESCRIPTOR array_desc{};
    array_desc.Width = _res.x;
    array_desc.Height = _res.y;
    array_desc.Depth = _res.z;
    switch (_data.pixel_storage) {
        case PixelStorage::BYTE1:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 1;
            break;
        case PixelStorage::BYTE2:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 2;
            break;
        case PixelStorage::BYTE4:
            array_desc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            array_desc.NumChannels = 4;
            break;
        case PixelStorage::FLOAT1:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 1;
            break;
        case PixelStorage::FLOAT2:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 2;
            break;
        case PixelStorage::FLOAT4:
            array_desc.Format = CU_AD_FORMAT_FLOAT;
            array_desc.NumChannels = 4;
            break;
        default: OC_ASSERT(0); break;
    }

    OC_CU_CHECK(cuArray3DCreate(&_array_handle, &array_desc));

    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = _array_handle;
    res_desc.flags = 0;
    CUDA_TEXTURE_DESC tex_desc{};
    tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_CLAMP;
    tex_desc.maxAnisotropy = 2;
    tex_desc.maxMipmapLevelClamp = 9;
    tex_desc.filterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    OC_CU_CHECK(cuSurfObjectCreate(&_data.surface, &res_desc));
    OC_CU_CHECK(cuTexObjectCreate(&_data.texture, &res_desc, &tex_desc, nullptr));
}
CUDATexture::~CUDATexture() {
    OC_CU_CHECK(cuArrayDestroy(_array_handle));
    OC_CU_CHECK(cuTexObjectDestroy(_data.texture));
    OC_CU_CHECK(cuSurfObjectDestroy(_data.surface));
}
size_t CUDATexture::data_size() const noexcept { return CUDADevice::size(Type::Tag::TEXTURE); }
size_t CUDATexture::data_alignment() const noexcept { return CUDADevice::alignment(Type::Tag::TEXTURE); }
size_t CUDATexture::max_member_size() const noexcept { return sizeof(handle_ty); }

}// namespace ocarina