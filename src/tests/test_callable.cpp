//
// Created by Zero on 16/05/2022.
//

#include "dsl/common.h"
#include "core/concepts.h"
#include "dsl/operators.h"
#include "dsl/func.h"
#include "ast/expression.h"
#include <iostream>
#include <rhi/context.h>
#include "rhi/device.h"
#include "generator/cpp_codegen.h"
#include "core/platform.h"
#include "dsl/syntax_sugar.h"
#include "core/util.h"
#include "rt/hit.h"
#include "rhi/stream.h"

using std::cout;
using std::endl;
using namespace ocarina;

Var<int> add(Var<int> a, Var<int> b) {
    return a + b;
}

template<typename T>
auto func(T a, T b) {
    Var<int3> arr;
    arr[1] = b;
    a = arr[1] + 1;
    $if(a + b > 0) {
        a = (a + 9) * b;
    }
    $elif(a > 0) {
        a += 3;
    };
    $for(v, 9) {
        a = a + v;
    };
    $switch(a) {
        $case(1) {
            $comment(1111)
                $break;
        };
        $case(2) {
            $comment(2222)
                $break;
        };
    };
    $while(a > 10) {
        a -= 1;
    };
    return a + b;
}


int main(int argc, char *argv[]) {
    Callable callable = func<Var<int>>;

    Callable add = [&](Var<int> a, Var<int> b, Var<int>) {
        a = a + b;
        print("{}, {}---",a, 1.f);
        return a;
    };

    Callable c1 = [&](Var<int> a, Var<int> b) {
        $for(v, 9) {
            a = a + v;
        };
        Var<Hit> hit;
        hit->init();
        $if(hit->is_miss()){
            $comment(miss)

        };
        Var<float4x4> m4;
        Var<ocarina::tuple<int, float>> tp;
        Var vec = m4.get<3>();
        m4.get<3>() = vec;
        Var<int[6]> arr;
        arr[3] = 0;

        return add(a, a + 7, 1);
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    Device device = context.create_device("cuda");

    Stream stream = device.create_stream();

    CppCodegen codegen;
    decltype(auto) f = callable.function();
    codegen.emit(f);
    cout << codegen.scratch().c_str();


    return 0;
}