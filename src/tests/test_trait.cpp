//
// Created by Zero on 21/09/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "dsl/polymorphic.h"

using namespace ocarina;

namespace ocarina {
namespace detail {

template<typename value_ty>
requires(is_std_vector_v<value_ty> && is_scalar_v<typename value_ty::value_type>) || is_basic_v<value_ty>
struct SharedData {
public:
    value_ty _host_value{};
    optional<dsl_t<value_ty>> _device_value{};
    uint offset{};

public:
    SharedData(value_ty val) : _host_value(val) {}

    [[nodiscard]] value_ty &hv() const noexcept { return _host_value; }
    [[nodiscard]] dsl_t<value_ty> &dv() const noexcept { return *_device_value; }

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
    [[nodiscard]] auto decode(const Array<T> &array, uint offset) noexcept {
        if constexpr (is_scalar_v<value_ty>) {
            return as<value_ty>(array[offset]);
        } else if constexpr (is_vector_v<value_ty>) {
            Var<value_ty> ret;
            using element_ty = vector_element_t<value_ty>;
            for (int i = 0; i < vector_dimension_v<value_ty>; ++i) {
                ret[i] = as<element_ty>(array[offset + i]);
            }
            return ret;
        } else if constexpr (is_matrix_v<value_ty>) {
            Var<value_ty> ret;
            uint cursor = 0u;
            for (int i = 0; i < matrix_dimension_v<value_ty>; ++i) {
                for (int j = 0; j < matrix_dimension_v<value_ty>; ++j) {
                    ret[i][j] = as<float>(array[cursor + offset]);
                    ++cursor;
                }
            }
            return ret;
        } else if constexpr (is_std_vector_v<value_ty>) {
            using element_ty = value_ty::value_type;
            Array<element_ty> ret;
            for (int i = 0; i < _host_value.size(); ++i) {
                ret[i] = array[i + offset];
            }
            return ret;
        } else {
            static_assert(always_false_v<value_ty>);
        }
    }
};

}
}// namespace ocarina::detail

#define OC_SERIALIZE_MEMBER(type, name) \
    type name{};                        \
    optional<Var<type>> _device_##name; \
    uchar _offset_##name;

#define OC_ENCODE_ELEMENT(name)             \
    _offset_##name = datas.size() - offset; \
    detail::encode(datas, name);

#define OC_DECODE_ELEMENT(name) _device_##name = detail::decode<decltype(name)>(values, _offset_##name);

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
        Array<float> values = da->read_dynamic_array<float>(_data_size); \
        MAP(OC_DECODE_ELEMENT, __VA_ARGS__)                              \
    }

struct Test {
    OC_SERIALIZE_MEMBER(float, a);
    OC_SERIALIZE_MEMBER(float, b);

    OC_ENCODE_DECODE(a, b)
};

int main() {

    //    Test t;

    //    Float a{nullptr};
    //    Float a{};

    //    cout << is_vector_v<float2>;
    //    cout << (!is_dsl_v<Float3>) && is_vector_v<Float3> ;

    return 0;

    //    cout << typeid(scalar_t<int3>).name() << endl;
    //    cout << typeid(scalar_t<float>).name() << endl;
    //    cout << typeid(scalar_t<float4x4>).name() << endl;
    //    cout << typeid(scalar_t<Float3>).name() << endl;
    //    cout << typeid(scalar_t<Float>).name() << endl;
    //    cout << typeid(scalar_t<Float4x4>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(vec_t<float3, 2>).name() << endl;
    //    cout << typeid(vec_t<float, 2>).name() << endl;
    //    cout << typeid(vec_t<float4x4, 2>).name() << endl;
    //    cout << typeid(vec_t<Float3, 2>).name() << endl;
    //    cout << typeid(vec_t<Float, 2>).name() << endl;
    //    cout << typeid(vec_t<Float4x4, 2>).name() << endl;
    //
    //    cout << endl;
    //
    //    cout << typeid(matrix_t<float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<float, 2>).name() << endl;
    //    cout << typeid(matrix_t<float4x4, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float3, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float, 2>).name() << endl;
    //    cout << typeid(matrix_t<Float4x4, 2>).name() << endl;
    //
    //    cout << typeid(boolean_t<Float>).name() << endl;
    //    cout << typeid(boolean_t<float>).name() << endl;

    return 0;
}