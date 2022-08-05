//
// Created by Zero on 06/08/2022.
//

#include "cuda_texture.h"
#include "util.h"

namespace ocarina {

CUDATexture::CUDATexture(CUDADevice *device, uint2 res, PixelStorage pixel_storage)
    : _device(device), _res(res), _pixel_storage(pixel_storage) {
    CUDA_ARRAY_DESCRIPTOR array_desc{};
    array_desc.Width = _res.x;
    array_desc.Height = _res.y;
    switch (_pixel_storage) {
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
}

}// namespace ocarina