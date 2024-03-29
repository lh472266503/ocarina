//
// Created by Zero on 2024/3/22.
//

#include "widgets.h"

namespace ocarina {

template<typename TDialog>
bool file_dialog_common(const FileDialogFilterVec &filters, fs::path &path, DWORD options, const CLSID clsid) {
    TDialog *pDialog;
    if (FAILED(CoCreateInstance(clsid, NULL, CLSCTX_ALL, IID_PPV_ARGS(&pDialog)))) {
        OC_WARNING("file_dialog failure");
        return false;
    }

    if (IShellItem * pShellItem;
        SUCCEEDED(SHCreateItemFromParsingName(path.parent_path().c_str(), NULL, IID_IShellItem,
                                              reinterpret_cast<void **>(&pShellItem)))) {
        pDialog->SetFolder(pShellItem);
        pShellItem->Release();
    }

    pDialog->SetOptions(options | FOS_FORCEFILESYSTEM);
    if (pDialog->Show(nullptr) == S_OK) {
        if (IShellItem * pItem;
            pDialog->GetResult(&pItem) == S_OK) {
            pItem->Release();
            PWSTR pathStr;
            if (pItem->GetDisplayName(SIGDN_FILESYSPATH, &pathStr) == S_OK) {
                path = pathStr;
                CoTaskMemFree(pathStr);
                return true;
            }
        }
    }
    return false;
}

bool Widgets::open_file_dialog(std::filesystem::path &path, const ocarina::FileDialogFilterVec &filters) noexcept {
    return file_dialog_common<IFileOpenDialog>(filters, path,
                                               FOS_FILEMUSTEXIST, CLSID_FileOpenDialog);
};

bool Widgets::slider_floatN(const std::string &label, float *val, ocarina::uint size, float min, float max) noexcept {
    switch (size) {
        case 1:
            return slider_float(label, val, min, max);
        case 2:
            return slider_float2(label, reinterpret_cast<float2 *>(val), min, max);
        case 3:
            return slider_float3(label, reinterpret_cast<float3 *>(val), min, max);
        case 4:
            return slider_float4(label, reinterpret_cast<float4 *>(val), min, max);
        default:
            OC_ERROR("error");
            break;
    }
    return false;
}

bool Widgets::colorN_edit(const std::string &label, float *val, ocarina::uint size) noexcept {
    switch (size) {
        case 3:
            return color_edit(label, reinterpret_cast<float3 *>(val));
        case 4:
            return color_edit(label, reinterpret_cast<float4 *>(val));
        default:
            OC_ERROR("error");
            return false;
    }
}

bool Widgets::input_floatN(const std::string &label, float *val, ocarina::uint size) noexcept {
    switch (size) {
        case 1:
            return input_float(label, val);
        case 2:
            return input_float2(label, reinterpret_cast<float2 *>(val));
        case 3:
            return input_float3(label, reinterpret_cast<float3 *>(val));
        case 4:
            return input_float4(label, reinterpret_cast<float4 *>(val));
        default:
            OC_ERROR("error");
            break;
    }
    return false;
}
}// namespace ocarina