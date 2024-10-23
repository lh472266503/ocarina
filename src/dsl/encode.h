//
// Created by zhu on 2023/4/21.
//

#pragma once

#include "dsl/type_trait.h"
#include "dynamic_array.h"

namespace ocarina {

template<typename T>
class RegistrableManaged;

using encoded_ty = float;

template<typename U = encoded_ty>
requires(sizeof(U) == sizeof(float))
struct DataAccessor {
    mutable Uint offset;
    const RegistrableManaged<U> &datas;

    template<typename T>
    [[nodiscard]] DynamicArray<T> load_dynamic_array(uint size) const noexcept {
        auto ret = datas.template load_dynamic_array<T>(size, offset);
        offset += size * static_cast<uint>(sizeof(T));
        return ret;
    }

    template<typename Target>
    OC_NODISCARD auto byte_read() const noexcept {
        auto ret = datas.template byte_read<Target>(offset);
        offset += static_cast<uint>(sizeof(Target));
        return ret;
    }
};

namespace detail {
template<typename T = encoded_ty>
class encodable_impl {
public:
    /// for host
    virtual void encode(RegistrableManaged<T> &data) const noexcept {}
    virtual void update(RegistrableManaged<T> &data) const noexcept {}
    virtual void update() const noexcept {}
    /// for device
    virtual void decode(const DataAccessor<T> *da) const noexcept {}
    virtual void decode() const noexcept {}
    [[nodiscard]] virtual uint element_num() const noexcept { return 0; }
    [[nodiscard]] virtual bool has_device_value() const noexcept { return true; }
    virtual void reset_device_value() const noexcept {}
    virtual ~encodable_impl() = default;
};
}// namespace detail

template<typename T = encoded_ty>
class Encodable : public detail::encodable_impl<T> {};

template<typename value_ty, typename T = encoded_ty>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct EncodedData final : public Encodable<T> {
private:
    using host_ty = std::variant<value_ty, std::function<value_ty()>>;
    host_ty _host_value{};
    optional<dsl_t<value_ty>> _device_value{};
    /// origin index in buffer
    mutable uint _offset{InvalidUI32};
    mutable RegistrableManaged<T> *_data{nullptr};

public:
    explicit EncodedData(value_ty val = value_ty{}) : _host_value(std::move(val)) {}
    EncodedData &operator=(const value_ty &val) {
        _host_value = val;
        return *this;
    }
    EncodedData &operator=(const std::function<value_ty()> &val) {
        _host_value = val;
        return *this;
    }
    [[nodiscard]] bool has_device_value() const noexcept override { return _device_value.has_value(); }
    void reset_device_value() const noexcept override {
        (const_cast<decltype(_device_value) &>(_device_value)).reset();
    }
    [[nodiscard]] value_ty hv() const noexcept {
        if (_host_value.index() == 0) {
            return std::get<0>(_host_value);
        } else {
            return std::get<1>(_host_value)();
        }
    }
    [[nodiscard]] value_ty &hv() noexcept {
        OC_ASSERT(_host_value.index() == 0);
        return std::get<0>(_host_value);
    }
    [[nodiscard]] const dsl_t<value_ty> &dv() const noexcept {
        OC_ASSERT(has_device_value());
        return *_device_value;
    }
    
    [[nodiscard]] dsl_t<value_ty> operator*() const noexcept {
        if (has_device_value()) {
            return dv();
        } else {
            return dsl_t<value_ty>(hv());
        }
    }

    [[nodiscard]] bool has_encoded() const noexcept { return _offset != InvalidUI32; }
    void invalidation() const noexcept { _offset = InvalidUI32; }

    void init_encode(RegistrableManaged<T> &data) const noexcept {
        OC_ASSERT(!has_encoded());
        _offset = data.host_buffer().size();
        _data = addressof(data);
        if constexpr (is_scalar_v<value_ty>) {
            data.push_back(bit_cast<T>(hv()));
        } else if constexpr (is_vector_v<value_ty>) {
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                data.push_back(bit_cast<T>(hv()[i]));
            }
        } else if constexpr (is_matrix_v<value_ty>) {
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    data.push_back(bit_cast<T>(hv()[i][j]));
                }
            }
        } else if constexpr (is_std_vector_v<value_ty>) {
            for (int i = 0; i < hv().size(); ++i) {
                data.push_back(bit_cast<T>(hv()[i]));
            }
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    void update(RegistrableManaged<T> &data) const noexcept override {
        OC_ASSERT(has_encoded());
        if constexpr (is_scalar_v<value_ty>) {
            data.host_buffer()[_offset] = bit_cast<T>(hv());
        } else if constexpr (is_vector_v<value_ty>) {
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                data.host_buffer()[_offset + i] = bit_cast<T>(hv()[i]);
            }
        } else if constexpr (is_matrix_v<value_ty>) {
            uint count = 0;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    data.host_buffer()[_offset + count] = bit_cast<T>(hv()[i][j]);
                    ++count;
                }
            }
        } else if constexpr (is_std_vector_v<value_ty>) {
            for (int i = 0; i < hv().size(); ++i) {
                data.host_buffer()[_offset + i] = bit_cast<T>(hv()[i]);
            }
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    void update() const noexcept override {
        update(*_data);
    }

    void encode(RegistrableManaged<T> &data) const noexcept override {
        if (has_encoded()) {
            update(data);
        } else {
            init_encode(data);
        }
    }

    [[nodiscard]] uint element_num() const noexcept override {
        if constexpr (is_scalar_v<value_ty>) {
            static_assert(sizeof(value_ty) <= sizeof(float));
            return 1;
        } else if constexpr (is_vector_v<value_ty>) {
            return vector_dimension_v<value_ty>;
        } else if constexpr (is_matrix_v<value_ty>) {
            return sqr(matrix_dimension_v<value_ty>);
        } else if constexpr (is_std_vector_v<value_ty>) {
            return hv().size();
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    [[nodiscard]] auto _decode(const DynamicArray<T> &array) const noexcept {
        if constexpr (is_scalar_v<value_ty>) {
            return as<value_ty>(array[0]);
        } else if constexpr (is_vector_v<value_ty>) {
            Var<value_ty> ret;
            using element_ty = vector_element_t<value_ty>;
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                ret[i] = as<element_ty>(array[i]);
            }
            return ret;
        } else if constexpr (is_matrix_v<value_ty>) {
            Var<value_ty> ret;
            uint cursor = 0u;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    ret[i][j] = as<float>(array[cursor]);
                    ++cursor;
                }
            }
            return ret;
        } else if constexpr (is_std_vector_v<value_ty>) {
            using element_ty = value_ty::value_type;
            DynamicArray<element_ty> ret{hv().size()};
            for (int i = 0; i < hv().size(); ++i) {
                ret[i] = as<element_ty>(array[i]);
            }
            return ret;
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    void decode(const DataAccessor<T> *da) const noexcept override {
        const DynamicArray<T> array = da->template load_dynamic_array<T>(element_num());
        const_cast<decltype(_device_value) *>(&_device_value)->emplace(_decode(array));
    }

    void decode() const noexcept override {
        if (_data == nullptr) {
            return;
        }
        DataAccessor<T> da{_offset * sizeof(T), *_data};
        decode(&da);
    }
};

namespace detail {
OC_MAKE_AUTO_MEMBER_FUNC(encode)
OC_MAKE_AUTO_MEMBER_FUNC(update)
OC_MAKE_AUTO_MEMBER_FUNC(decode)
OC_MAKE_AUTO_MEMBER_FUNC(reset_device_value)
OC_MAKE_AUTO_MEMBER_FUNC(has_device_value)
OC_MAKE_AUTO_MEMBER_FUNC(element_num)
}// namespace detail

#define OC_ENCODE_ELEMENT(name) ocarina::detail::encode(name, datas);
#define OC_UPDATE_ELEMENT(name) ocarina::detail::update(name, datas);
#define OC_DECODE_ELEMENT(name) ocarina::detail::decode(name, da);
#define OC_INVALIDATE_ELEMENT(name) ocarina::detail::reset_device_value(name);
#define OC_VALID_ELEMENT(name) &&ocarina::detail::has_device_value(name)
#define OC_SIZE_ELEMENT(name) +ocarina::detail::element_num(name)

#define OC_ENCODABLE_FUNC(Super, ...)                                            \
    [[nodiscard]] uint element_num() const noexcept override {                   \
        return Super::element_num() MAP(OC_SIZE_ELEMENT, __VA_ARGS__);           \
    }                                                                            \
    void encode(RegistrableManaged<encoded_ty> &datas) const noexcept override { \
        Super::encode(datas);                                                    \
        MAP(OC_ENCODE_ELEMENT, __VA_ARGS__)                                      \
    }                                                                            \
    void update(RegistrableManaged<encoded_ty> &datas) const noexcept override { \
        Super::update(datas);                                                    \
        MAP(OC_UPDATE_ELEMENT, __VA_ARGS__)                                      \
    }                                                                            \
    void decode(const DataAccessor<encoded_ty> *da) const noexcept override {    \
        Super::decode(da);                                                       \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                                      \
    }                                                                            \
    void reset_device_value() const noexcept override {                          \
        Super::reset_device_value();                                             \
        MAP(OC_INVALIDATE_ELEMENT, __VA_ARGS__)                                  \
    }                                                                            \
    [[nodiscard]] bool has_device_value() const noexcept override {              \
        return Super::has_device_value() MAP(OC_VALID_ELEMENT, __VA_ARGS__);     \
    }

}// namespace ocarina