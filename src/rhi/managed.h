//
// Created by Zero on 24/10/2022.
//

#pragma once

#include "core/stl.h"
#include "resources/buffer.h"

namespace ocarina {

template<typename T>
class Managed : public Buffer<T>, public ocarina::vector<T> {
public:
    using host_ty = ocarina::vector<T>;
    using device_ty = Buffer<T>;
    using element_ty = T;

public:
    Managed() = default;
    Managed(Device::Impl *device, size_t size)
        : device_ty(device, size) {
        host_ty::reserve(size);
    }

    // Move constructor
    Managed(Managed &&other) noexcept
        : device_ty(std::move(other)),
          host_ty(std::move(other)) {}

    // Move assignment
    Managed &operator=(Managed &&other) noexcept {
        device_ty::operator=(std::move(other));
        host_ty::operator=(std::move(other));
        return *this;
    }

    void set_host(host_ty &&val) noexcept {
        host() = std::move(val);
    }

    [[nodiscard]] BufferUploadCommand *upload() const noexcept {
        return device_ty::upload(host_ty::data());
    }

    [[nodiscard]] BufferDownloadCommand *download() noexcept {
        return device_ty::download(host_ty::data());
    }

    void upload_immediately() const noexcept {
        device_ty::upload_immediately(host_ty::data());
    }

    void download_immediately() noexcept {
        device_ty::download_immediately(host_ty::data());
    }

    [[nodiscard]] BufferUploadCommand *upload_sync() const noexcept {
        return device_ty::upload_sync(host_ty::data());
    }

    [[nodiscard]] BufferDownloadCommand *download_sync() noexcept {
        return device_ty::download_sync(host_ty::data());
    }

    [[nodiscard]] device_ty &device() noexcept { return *this; }
    [[nodiscard]] const device_ty &device() const noexcept { return *this; }
    [[nodiscard]] host_ty &host() noexcept { return *this; }
    [[nodiscard]] const host_ty &host() const noexcept { return *this; }
};

}// namespace ocarina