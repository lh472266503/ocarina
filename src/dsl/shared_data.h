//
// Created by zhu on 2023/4/21.
//

#pragma once

#include "type_trait.h"
#include "rhi/common.h"

namespace ocarina {

template<typename U = float>
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

template<typename value_ty>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct SharedData {
public:
    value_ty _host_value{};
    optional<dsl_t<value_ty>> _device_value{};
    uint _offset{};

public:
    explicit SharedData(value_ty val = {}) : _host_value(std::move(val)) {}
    SharedData &operator=(const value_ty &val) {
        _host_value = val;
        return *this;
    }
    [[nodiscard]] auto &hv() const noexcept { return _host_value; }
    [[nodiscard]] auto &hv() noexcept { return _host_value; }
    [[nodiscard]] const dsl_t<value_ty> &dv() const noexcept { return *_device_value; }

    template<typename T>
    void encode(vector<T> &data) noexcept {
        if constexpr (is_scalar_v<value_ty>) {
            data.push_back(bit_cast<T>(_host_value));
        } else if constexpr (is_vector_v<value_ty>) {
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                data.push_back(bit_cast<T>(_host_value[i]));
            }
        } else if constexpr (is_matrix_v<value_ty>) {
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    data.push_back(bit_cast<T>(_host_value[i][j]));
                }
            }
        } else if constexpr (is_std_vector_v<value_ty>) {
            for (int i = 0; i < _host_value.size(); ++i) {
                data.push_back(bit_cast<T>(_host_value[i]));
            }
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    [[nodiscard]] uint size() noexcept {
        if constexpr (is_scalar_v<value_ty>) {
            static_assert(sizeof(value_ty) <= sizeof(float));
            return 1;
        } else if constexpr (is_vector_v<value_ty>) {
            return vector_dimension_v<value_ty>;
        } else if constexpr (is_matrix_v<value_ty>) {
            return sqr(matrix_dimension_v<value_ty>);
        } else if constexpr (is_std_vector_v<value_ty>) {
            return _host_value.size();
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    template<typename T>
    [[nodiscard]] auto _decode(const Array<T> &array) noexcept {
        _offset = 0;
        if constexpr (is_scalar_v<value_ty>) {
            return as<value_ty>(array[_offset]);
        } else if constexpr (is_vector_v<value_ty>) {
            Var<value_ty> ret;
            using element_ty = vector_element_t<value_ty>;
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                ret[i] = as<element_ty>(array[_offset + i]);
            }
            return ret;
        } else if constexpr (is_matrix_v<value_ty>) {
            Var<value_ty> ret;
            uint cursor = 0u;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    ret[i][j] = as<float>(array[cursor + _offset]);
                    ++cursor;
                }
            }
            return ret;
        } else if constexpr (is_std_vector_v<value_ty>) {
            using element_ty = value_ty::value_type;
            Array<element_ty> ret{_host_value.size()};
            for (int i = 0; i < _host_value.size(); ++i) {
                ret[i] = array[i + _offset];
            }
            return ret;
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    template<typename T>
    void decode(const Array<T> &array) noexcept {
        _device_value = _decode(array);
    }

    template<typename T>
    void decode(const DataAccessor<T> *da) noexcept {
        const Array<T> array = da->read_dynamic_array<T>(size());
        _device_value = _decode(array);
    }
};

#define OC_ENCODE_ELEMENT(name)           \
    name._offset = datas.size() - offset; \
    name.encode(datas);

#define OC_DECODE_ELEMENT(name) name.decode(da);

#define OC_ENCODE_DECODE(...)                                            \
    uint _data_size{0u};                                                 \
                                                                         \
public:                                                                  \
    void encode(vector<float> &datas) noexcept {                         \
        uint offset = datas.size();                                      \
        MAP(OC_ENCODE_ELEMENT, __VA_ARGS__)                              \
        _data_size = datas.size() - offset;                              \
    }                                                                    \
    void decode(const DataAccessor<float> *da) noexcept {                \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                              \
    }

}// namespace ocarina