//
// Created by Zero on 21/09/2022.
//

#include "core/stl.h"
#include "dsl/common.h"

using namespace ocarina;



int main() {
    //    cout << is_vector_v<float2>;
    //    cout << (!is_dsl_v<Float3>) && is_vector_v<Float3> ;

    cout << typeid(scalar_t<int3>).name() << endl;
    cout << typeid(scalar_t<float>).name() << endl;
    cout << typeid(scalar_t<float4x4>).name() << endl;
    cout << typeid(scalar_t<Float3>).name() << endl;
    cout << typeid(scalar_t<Float>).name() << endl;
    cout << typeid(scalar_t<Float4x4>).name() << endl;

    cout << endl;

    cout << typeid(vec_t<float3, 2>).name() << endl;
    cout << typeid(vec_t<float, 2>).name() << endl;
    cout << typeid(vec_t<float4x4, 2>).name() << endl;
    cout << typeid(vec_t<Float3, 2>).name() << endl;
    cout << typeid(vec_t<Float, 2>).name() << endl;
    cout << typeid(vec_t<Float4x4, 2>).name() << endl;

    cout << endl;

    cout << typeid(matrix_t<float3, 2>).name() << endl;
    cout << typeid(matrix_t<float, 2>).name() << endl;
    cout << typeid(matrix_t<float4x4, 2>).name() << endl;
    cout << typeid(matrix_t<Float3, 2>).name() << endl;
    cout << typeid(matrix_t<Float, 2>).name() << endl;
    cout << typeid(matrix_t<Float4x4, 2>).name() << endl;

    cout << typeid(boolean_t<Float>).name() << endl;
    cout << typeid(boolean_t<float>).name() << endl;

    return 0;
}