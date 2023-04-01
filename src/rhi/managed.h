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

    [[nodiscard]] device_ty &device() noexcept { return *this; }
    [[nodiscard]] const device_ty &device() const noexcept { return *this; }
    [[nodiscard]] host_ty &host() noexcept { return *this; }
    [[nodiscard]] const host_ty &host() const noexcept { return *this; }
    void set_host(host_ty val) noexcept { host() = std::move(val); }
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

    void reset_device_buffer(Device &d, size_t num = 0) {
        num = num == 0 ? host_ty ::size() : num;
        if (num == 0) {
            return;
        }
        device() = d.create_buffer<T>(num);
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
};

template<typename T>
class ManagedWrapper : public Managed<T> {
public:
    using Super = Managed<T>;

private:
    uint _id{InvalidUI32};
    ResourceArray *_resource_array{};

public:
    ManagedWrapper() = default;
    void init(ResourceArray &resource_array) noexcept { _resource_array = &resource_array; }
    explicit ManagedWrapper(ResourceArray &resource_array) : _resource_array(&resource_array) {}
    void register_self() noexcept {
        _id = _resource_array->emplace(Super::device());
    }

    template<typename Index>
    requires concepts::all_integral<expr_value_t<Index>>
        OC_NODISCARD auto read(Index &&index) const noexcept {
        return _resource_array->buffer<T>(_id).read(OC_FORWARD(index));
    }

    template<typename Index, typename Val>
    requires concepts::integral<expr_value_t<Index>> && concepts::is_same_v<T, expr_value_t<Val>>
    void write(Index &&index, Val &&elm) {
        _resource_array->buffer<T>(_id).write(OC_FORWARD(index), OC_FORWARD(elm));
    }
};

}// namespace ocarina