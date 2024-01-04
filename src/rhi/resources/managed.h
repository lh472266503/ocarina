//
// Created by Zero on 24/10/2022.
//

#pragma once

#include "core/stl.h"
#include "buffer.h"
#include "texture.h"
#include "util/image_io.h"

namespace ocarina {

template<typename T>
class Array;

template<typename T>
class Managed : public Buffer<T>, public ocarina::vector<T> {
public:
    using host_ty = ocarina::vector<T>;
    using device_ty = Buffer<T>;
    using element_ty = T;

public:
    Managed() = default;
    Managed(Device::Impl *device, size_t size, const string &desc = "")
        : device_ty(device, size, desc) {
        host_ty::reserve(size);
    }

    explicit Managed(size_t size) {
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

    [[nodiscard]] device_ty &device_buffer() noexcept { return *this; }
    [[nodiscard]] const device_ty &device_buffer() const noexcept { return *this; }
    [[nodiscard]] host_ty &host_buffer() noexcept { return *this; }
    [[nodiscard]] const host_ty &host_buffer() const noexcept { return *this; }
    void set_host(host_ty val) noexcept { host_buffer() = std::move(val); }
    [[nodiscard]] const T *operator->() const { return host_ty::data(); }
    [[nodiscard]] T *operator->() { return host_ty::data(); }

    template<typename Index>
    requires concepts::integral<Index>
    [[nodiscard]] auto operator[](Index &&i) { return host_ty::operator[](OC_FORWARD(i)); }

    template<typename V>
    requires concepts::iterable<V>
    void append(V &&v) {
        host_ty::insert(host_ty::cend(), v.cbegin(), v.cend());
    }

    void reset_device_buffer_immediately(Device &d, const string &desc = "", size_t num = 0) {
        num = num == 0 ? host_ty ::size() : num;
        if (num == 0) {
            return;
        }
        device_buffer() = d.template create_buffer<T>(num, desc);
    }

    /**
     * reset host and device memory
     * @param d device
     * @param num number of element
     */
    void reset_all(Device &d, size_t num, const string &desc = "") {
        reset_device_buffer_immediately(d, desc, num);
        host_buffer().resize(num);
    }

    void clear_all() {
        host_ty::clear();
        device_ty::destroy();
    }

    [[nodiscard]] BufferUploadCommand *upload(bool async = true) const noexcept {
        return device_ty::upload(host_ty::data(), async);
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
};

//todo
class ManagedTexture : public Texture, public ImageIO {
public:
    using host_ty = ocarina::ImageIO;
    using device_ty = ocarina::Texture;

public:
    ManagedTexture() = default;

    void upload_immediately() const noexcept {

    }

    void download_immediately() noexcept {

    }
};

}// namespace ocarina