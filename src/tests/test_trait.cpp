//
// Created by Zero on 21/09/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "dsl/polymorphic.h"

using namespace ocarina;



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

    Test t;

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