//
// Created by zhu on 2023/4/21.
//

#pragma once

#include "dsl/type_trait.h"
#include "dynamic_array.h"

namespace ocarina {

template<typename T>
class RegistrableManaged;

using encoded_ty = uint;

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

template<typename T = encoded_ty>
class Encodable {
public:
    /// for host
    virtual void encode(RegistrableManaged<T> &data) const noexcept {}
    virtual void update(RegistrableManaged<T> &data) const noexcept {}
    virtual void update() const noexcept {}
    virtual void invalidate() const noexcept {}
    /// for device
    virtual void decode(const DataAccessor<T> *da) const noexcept {}
    virtual void decode() const noexcept {}
    [[nodiscard]] virtual uint encoded_size() const noexcept { return 0; }
    [[nodiscard]] virtual bool has_device_value() const noexcept { return true; }
    virtual void reset_device_value() const noexcept {}
    virtual ~Encodable() = default;
};

template<typename value_ty, typename T = encoded_ty>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct EncodedData final : public Encodable<T> {
private:
    using host_ty = std::variant<value_ty, std::function<value_ty()>>;
    host_ty host_value_{};
    optional<dsl_t<value_ty>> device_value_{};
    /// origin index in buffer
    mutable uint offset_{InvalidUI32};
    mutable RegistrableManaged<T> *data_{nullptr};

public:
    explicit EncodedData(value_ty val = value_ty{}) : host_value_(std::move(val)) {}
    EncodedData &operator=(const value_ty &val) {
        host_value_ = val;
        return *this;
    }
    EncodedData &operator=(const std::function<value_ty()> &val) {
        host_value_ = val;
        return *this;
    }
    [[nodiscard]] bool has_device_value() const noexcept override { return device_value_.has_value(); }
    void reset_device_value() const noexcept override {
        (const_cast<decltype(device_value_) &>(device_value_)).reset();
    }
    [[nodiscard]] value_ty hv() const noexcept {
        if (host_value_.index() == 0) {
            return std::get<0>(host_value_);
        } else {
            return std::get<1>(host_value_)();
        }
    }
    [[nodiscard]] value_ty &hv() noexcept {
        OC_ASSERT(host_value_.index() == 0);
        return std::get<0>(host_value_);
    }
    [[nodiscard]] const dsl_t<value_ty> &dv() const noexcept {
        OC_ASSERT(has_device_value());
        return *device_value_;
    }

    [[nodiscard]] dsl_t<value_ty> operator*() const noexcept {
        if (has_device_value()) {
            return dv();
        } else {
            return dsl_t<value_ty>(hv());
        }
    }

    [[nodiscard]] bool has_encoded() const noexcept { return offset_ != InvalidUI32; }
    void invalidate() const noexcept override { offset_ = InvalidUI32; }

    void init_encode(RegistrableManaged<T> &data) const noexcept {
        OC_ASSERT(!has_encoded());
        offset_ = data.host_buffer().size();
        data_ = addressof(data);
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
            data.host_buffer()[offset_] = bit_cast<T>(hv());
        } else if constexpr (is_vector_v<value_ty>) {
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                data.host_buffer()[offset_ + i] = bit_cast<T>(hv()[i]);
            }
        } else if constexpr (is_matrix_v<value_ty>) {
            uint count = 0;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    data.host_buffer()[offset_ + count] = bit_cast<T>(hv()[i][j]);
                    ++count;
                }
            }
        } else if constexpr (is_std_vector_v<value_ty>) {
            for (int i = 0; i < hv().size(); ++i) {
                data.host_buffer()[offset_ + i] = bit_cast<T>(hv()[i]);
            }
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }

    void update() const noexcept override {
        update(*data_);
    }

    void encode(RegistrableManaged<T> &data) const noexcept override {
        if (has_encoded()) {
            update(data);
        } else {
            init_encode(data);
        }
    }

    [[nodiscard]] uint encoded_size() const noexcept override {
        if constexpr (is_scalar_v<value_ty>) {
            static_assert(sizeof(value_ty) <= sizeof(float));
            return sizeof(value_ty);
        } else if constexpr (is_vector_v<value_ty>) {
            return sizeof(typename value_ty::scalar_type) * value_ty::dimension;
        } else if constexpr (is_matrix_v<value_ty>) {
            return sizeof(typename value_ty::scalar_type) * value_ty::ElementNum;
        } else if constexpr (is_std_vector_v<value_ty>) {
            return hv().size() * sizeof(typename value_ty::value_type);
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
            for (int i = 0; i < value_ty::ColNum; ++i) {
                for (int j = 0; j < value_ty::RowNum; ++j) {
                    ret[i][j] = as<float>(array[cursor]);
                    ++cursor;
                }
            }
            return ret;
        } else if constexpr (is_std_vector_v<value_ty>) {
            using element_ty = typename value_ty::value_type;
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
        const DynamicArray<T> array = da->template load_dynamic_array<T>(encoded_size() / sizeof(float));
        const_cast<decltype(device_value_) *>(&device_value_)->emplace(_decode(array));
    }

    void decode() const noexcept override {
        if (data_ == nullptr) {
            return;
        }
        DataAccessor<T> da{offset_ * sizeof(T), *data_};
        decode(&da);
    }
};

namespace detail {
OC_MAKE_AUTO_MEMBER_FUNC(encode)
OC_MAKE_AUTO_MEMBER_FUNC(invalidate)
OC_MAKE_AUTO_MEMBER_FUNC(update)
OC_MAKE_AUTO_MEMBER_FUNC(decode)
OC_MAKE_AUTO_MEMBER_FUNC(reset_device_value)
OC_MAKE_AUTO_MEMBER_FUNC(has_device_value)
OC_MAKE_AUTO_MEMBER_FUNC(encoded_size)
}// namespace detail

#define OC_ENCODE_ELEMENT(name) ocarina::detail::encode(name, datas);
#define OC_UPDATE_ELEMENT(name) ocarina::detail::update(name, datas);
#define OC_INVALIDATE_ELEMENT(name) ocarina::detail::invalidate(name);
#define OC_DECODE_ELEMENT(name) ocarina::detail::decode(name, da);
#define OC_RESET_DEVICE_ELEMENT(name) ocarina::detail::reset_device_value(name);
#define OC_VALID_ELEMENT(name) &&ocarina::detail::has_device_value(name)
#define OC_SIZE_ELEMENT(name) +ocarina::detail::encoded_size(name)

#define OC_ENCODABLE_FUNC(Super, ...)                                            \
    [[nodiscard]] uint encoded_size() const noexcept override {                  \
        return Super::encoded_size() MAP(OC_SIZE_ELEMENT, __VA_ARGS__);          \
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
    void invalidate() const noexcept override {                                  \
        Super::invalidate();                                                     \
        MAP(OC_INVALIDATE_ELEMENT, __VA_ARGS__)                                  \
    }                                                                            \
    void reset_device_value() const noexcept override {                          \
        Super::reset_device_value();                                             \
        MAP(OC_RESET_DEVICE_ELEMENT, __VA_ARGS__)                                \
    }                                                                            \
    [[nodiscard]] bool has_device_value() const noexcept override {              \
        return Super::has_device_value() MAP(OC_VALID_ELEMENT, __VA_ARGS__);     \
    }

}// namespace ocarina