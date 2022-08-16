//
// Created by Zero on 06/08/2022.
//

#include "cuda_image.h"
#include "util.h"

namespace ocarina {

CUDAImage::CUDAImage(CUDADevice *device, uint2 res, PixelStorage pixel_storage)
    : _device(device), _res(res) {
    _image_data.pixel_storage = pixel_storage;
    init();
}

void CUDAImage::init() {
    CUDA_ARRAY_DESCRIPTOR array_desc{};
    array_desc.Width = _res.x;
    array_desc.Height = _res.y;
    switch (_image_data.pixel_storage) {
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

    OC_CU_CHECK(cuArrayCreate(&_array_handle, &array_desc));

    CUDA_RESOURCE_DESC res_desc{};
    res_desc.resType = CU_RESOURCE_TYPE_ARRAY;
    res_desc.res.array.hArray = _array_handle;
    res_desc.flags = 0;
    CUDA_TEXTURE_DESC tex_desc{};
    tex_desc.addressMode[0] = CU_TR_ADDRESS_MODE_WRAP;
    tex_desc.addressMode[1] = CU_TR_ADDRESS_MODE_WRAP;
    tex_desc.addressMode[2] = CU_TR_ADDRESS_MODE_WRAP;
    tex_desc.maxAnisotropy = 2;
    tex_desc.maxMipmapLevelClamp = 9;
    tex_desc.filterMode = CU_TR_FILTER_MODE_POINT;
    tex_desc.flags = CU_TRSF_NORMALIZED_COORDINATES;
    OC_CU_CHECK(cuSurfObjectCreate(&_image_data.surface, &res_desc));
    OC_CU_CHECK(cuTexObjectCreate(&_image_data.texture, &res_desc, &tex_desc, nullptr));
}
CUDAImage::~CUDAImage() {
    OC_CU_CHECK(cuArrayDestroy(_array_handle));
    OC_CU_CHECK(cuTexObjectDestroy(_image_data.texture));
    OC_CU_CHECK(cuSurfObjectDestroy(_image_data.surface));
}

}// namespace ocarina