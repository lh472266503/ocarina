//
// Created by zhu on 2023/4/21.
//

#pragma once

#include "dsl/type_trait.h"
#include "dynamic_array.h"

namespace ocarina {

template<typename T>
class RegistrableManaged;

using buffer_ty = uint;

namespace detail {
template<typename U = buffer_ty>
requires(sizeof(U) == sizeof(float))
struct data_accessor_impl {
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
}// namespace detail

using DataAccessor = detail::data_accessor_impl<>;

enum EncodeType {
    Original,
    Uint8,
};

class Encodable {
public:
    /// for host
    virtual void encode(RegistrableManaged<buffer_ty> &data) const noexcept {}
    virtual void update(RegistrableManaged<buffer_ty> &data) const noexcept {}
    virtual void update() const noexcept {}
    virtual void invalidate() const noexcept {}
    /// for device
    virtual void decode(const DataAccessor *da) const noexcept {}
    virtual void decode() const noexcept {}
    [[nodiscard]] virtual uint encoded_size() const noexcept { return 0; }
    [[nodiscard]] virtual bool has_device_value() const noexcept { return true; }
    [[nodiscard]] virtual uint alignment() const noexcept { return 1; }

    /**
     * calculate the offset of current data and store
     * @param prev_size
     * @return the size of current add previous data
     */
    virtual uint cal_offset(uint prev_size) const noexcept { return prev_size; }
    virtual void reset_device_value() const noexcept {}
    virtual ~Encodable() = default;
};

template<typename value_ty, typename T = buffer_ty>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct EncodedData final : public Encodable {
public:
    using host_ty = std::variant<value_ty, std::function<value_ty()>>;
    using buffer_type = T;
    static constexpr size_t max_alignment = alignof(uint);

private:
    host_ty host_value_{};
    optional<dsl_t<value_ty>> device_value_{};
    EncodeType encode_type_{Original};

    /// origin offset in buffer
    mutable uint offset_{InvalidUI32};
    mutable RegistrableManaged<T> *data_{nullptr};

public:
    explicit EncodedData(value_ty val = value_ty{}, EncodeType et = Original)
        : host_value_(std::move(val)), encode_type_(et) {}
    EncodedData &operator=(const value_ty &val) {
        host_value_ = val;
        return *this;
    }
    EncodedData &operator=(const std::function<value_ty()> &val) {
        host_value_ = val;
        return *this;
    }
    OC_MAKE_MEMBER_GETTER_SETTER(encode_type, )
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
    [[nodiscard]] uint alignment() const noexcept override {
        switch (encode_type_) {
            case Original:
                return 4;
            case Uint8: {
                return 1;
            }
        }
        OC_ASSERT(false);
        return 1;
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
        offset_ = data.host_buffer().size() * sizeof(buffer_ty);
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
            data.host_buffer()[offset_ / 4] = bit_cast<T>(hv());
        } else if constexpr (is_vector_v<value_ty>) {
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                data.host_buffer()[offset_ / 4 + i] = bit_cast<T>(hv()[i]);
            }
        } else if constexpr (is_matrix_v<value_ty>) {
            uint count = 0;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    data.host_buffer()[offset_ / 4 + count] = bit_cast<T>(hv()[i][j]);
                    ++count;
                }
            }
        } else if constexpr (is_std_vector_v<value_ty>) {
            for (int i = 0; i < hv().size(); ++i) {
                data.host_buffer()[offset_ / 4 + i] = bit_cast<T>(hv()[i]);
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

    [[nodiscard]] uint cal_offset(ocarina::uint prev_size) const noexcept override {
        offset_ = mem_offset(prev_size, alignment());
        uint ret = offset_ + encoded_size();
        return ret;
    }

    [[nodiscard]] uint encoded_size() const noexcept override {
        switch (encode_type_) {
            case Original: {
                if constexpr (is_scalar_v<value_ty>) {
                    static_assert(sizeof(value_ty) <= sizeof(float));
                    return sizeof(value_ty);
                } else if constexpr (is_vector_v<value_ty>) {
                    return sizeof(typename value_ty::scalar_type) * value_ty::dimension;
                } else if constexpr (is_matrix_v<value_ty>) {
                    return sizeof(typename value_ty::scalar_type) * value_ty::element_num;
                } else if constexpr (is_std_vector_v<value_ty>) {
                    return hv().size() * sizeof(typename value_ty::value_type);
                } else {
                    static_assert(always_false_v<value_ty>);
                }
            }
            case Uint8: {
                if constexpr (is_scalar_v<value_ty>) {
                    static_assert(sizeof(value_ty) <= sizeof(float));
                    return sizeof(uint8_t);
                } else if constexpr (is_vector_v<value_ty>) {
                    return sizeof(uint8_t) * value_ty::dimension;
                } else if constexpr (is_matrix_v<value_ty>) {
                    return sizeof(uint8_t) * value_ty::element_num;
                } else if constexpr (is_std_vector_v<value_ty>) {
                    return hv().size() * sizeof(uint8_t);
                } else {
                    static_assert(always_false_v<value_ty>);
                }
            }
            default: break;
        }
        OC_ASSERT(0);
        return 0;
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
            for (int i = 0; i < value_ty::col_num; ++i) {
                for (int j = 0; j < value_ty::row_num; ++j) {
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

    void decode(const DataAccessor *da) const noexcept override {
        const DynamicArray<T> array = da->template load_dynamic_array<T>(encoded_size() / sizeof(T));
        const_cast<decltype(device_value_) *>(&device_value_)->emplace(_decode(array));
    }

    void decode() const noexcept override {
        if (data_ == nullptr) {
            return;
        }
        DataAccessor da{offset_, *data_};
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
OC_MAKE_AUTO_MEMBER_FUNC(cal_offset)
OC_MAKE_AUTO_MEMBER_FUNC(alignment)
}// namespace detail

#define OC_ENCODE_ELEMENT(name) ocarina::detail::encode(name, datas);
#define OC_UPDATE_ELEMENT(name) ocarina::detail::update(name, datas);
#define OC_INVALIDATE_ELEMENT(name) ocarina::detail::invalidate(name);
#define OC_DECODE_ELEMENT(name) ocarina::detail::decode(name, da);
#define OC_RESET_DEVICE_ELEMENT(name) ocarina::detail::reset_device_value(name);
#define OC_VALID_ELEMENT(name) &&ocarina::detail::has_device_value(name)
#define OC_SIZE_ELEMENT(name) +ocarina::detail::encoded_size(name)
#define OC_CAL_OFFSET(name) ret = ocarina::detail::cal_offset(name, ret);
#define OC_ALIGNMENT(name) ret = ocarina::max(ret, ocarina::detail::alignment(name));

#define OC_ENCODABLE_FUNC(Super, ...)                                           \
    [[nodiscard]] uint encoded_size() const noexcept override {                 \
        return Super::encoded_size() MAP(OC_SIZE_ELEMENT, __VA_ARGS__);         \
    }                                                                           \
    void encode(RegistrableManaged<buffer_ty> &datas) const noexcept override { \
        Super::encode(datas);                                                   \
        MAP(OC_ENCODE_ELEMENT, __VA_ARGS__)                                     \
    }                                                                           \
    void update(RegistrableManaged<buffer_ty> &datas) const noexcept override { \
        Super::update(datas);                                                   \
        MAP(OC_UPDATE_ELEMENT, __VA_ARGS__)                                     \
    }                                                                           \
    void decode(const DataAccessor *da) const noexcept override {               \
        Super::decode(da);                                                      \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                                     \
    }                                                                           \
    void invalidate() const noexcept override {                                 \
        Super::invalidate();                                                    \
        MAP(OC_INVALIDATE_ELEMENT, __VA_ARGS__)                                 \
    }                                                                           \
    void reset_device_value() const noexcept override {                         \
        Super::reset_device_value();                                            \
        MAP(OC_RESET_DEVICE_ELEMENT, __VA_ARGS__)                               \
    }                                                                           \
    [[nodiscard]] bool has_device_value() const noexcept override {             \
        return Super::has_device_value() MAP(OC_VALID_ELEMENT, __VA_ARGS__);    \
    }                                                                           \
    uint cal_offset(uint prev_size) const noexcept override {                   \
        uint ret = Super::cal_offset(prev_size);                                \
        MAP(OC_CAL_OFFSET, __VA_ARGS__)                                         \
        return ret;                                                             \
    }                                                                           \
    [[nodiscard]] uint alignment() const noexcept override {                    \
        uint ret = Super::alignment();                                          \
        MAP(OC_ALIGNMENT, __VA_ARGS__)                                          \
        return ret;                                                             \
    }
}// namespace ocarina