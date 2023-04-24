//
// Created by zhu on 2023/4/21.
//

#pragma once

#include "type_trait.h"
#include "rhi/common.h"

namespace ocarina {

template<typename U = float>
requires(sizeof(U) == sizeof(float))
struct DataAccessor {
    mutable Uint offset;
    ManagedWrapper<U> &datas;

    template<typename T>
    [[nodiscard]] Array<T> read_dynamic_array(uint size) const noexcept {
        auto ret = datas.template read_dynamic_array<T>(size, offset);
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

template<typename T = float>
class ISerializable {
public:
    /// for host
    virtual void encode(ManagedWrapper<T> &data) const noexcept = 0;
    /// for device
    virtual void decode(const DataAccessor<T> *da) const noexcept = 0;
    [[nodiscard]] virtual uint size() const noexcept = 0;
};

template<typename value_ty, typename T = float>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct SharedData : public ISerializable<T> {
public:
    value_ty _host_value{};
    optional<dsl_t<value_ty>> _device_value{};

public:
    explicit SharedData(value_ty val = {}) : _host_value(std::move(val)) {}
    SharedData &operator=(const value_ty &val) {
        _host_value = val;
        return *this;
    }
    [[nodiscard]] auto &hv() const noexcept { return _host_value; }
    [[nodiscard]] auto &hv() noexcept { return _host_value; }
    [[nodiscard]] const dsl_t<value_ty> &dv() const noexcept { return *_device_value; }
    [[nodiscard]] const dsl_t<value_ty> &operator*() const noexcept { return *_device_value; }
    void encode(ManagedWrapper<T> &data) const noexcept override {
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

    [[nodiscard]] uint size() const noexcept override {
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

    [[nodiscard]] auto _decode(const Array<T> &array) const noexcept {
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
            Array<element_ty> ret{hv().size()};
            for (int i = 0; i < hv().size(); ++i) {
                ret[i] = as<element_ty>(array[i]);
            }
            return ret;
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    void decode(const DataAccessor<T> *da) const noexcept override {
        const Array<T> array = da->template read_dynamic_array<T>(size());
        *(const_cast<decltype(_device_value) *>(&_device_value)) = _decode(array);
    }
};

#define OC_ENCODE_ELEMENT(name) \
    name.encode(datas);

#define OC_DECODE_ELEMENT(name) name.decode(da);

#define OC_ENCODE_DECODE(type, ...)                                     \
    mutable uint _size{0u};                                             \
                                                                        \
public:                                                                 \
    [[nodiscard]] uint size() const noexcept override { return _size; } \
    void encode(ManagedWrapper<type> &datas) const noexcept override {  \
        uint offset = datas.host().size();                              \
        MAP(OC_ENCODE_ELEMENT, __VA_ARGS__)                             \
        _size = datas.host().size() - offset;                           \
    }                                                                   \
    void decode(const DataAccessor<type> *da) const noexcept override { \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                             \
    }

}// namespace ocarina