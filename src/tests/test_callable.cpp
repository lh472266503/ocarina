//
// Created by Zero on 16/05/2022.
//

#include "dsl/common.h"
#include "core/concepts.h"
#include "dsl/operators.h"
#include "dsl/func.h"
#include "ast/expression.h"
#include <iostream>
#include <runtime/context.h>
#include "runtime/device.h"
#include "generator/cpp_codegen.h"
#include "core/platform.h"
#include "dsl/syntax_sugar.h"
#include "core/util.h"
#include "rt/hit.h"

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
//    T d = a + b;
//    T c = a + b * a + 1.5f;
    return a + b;
    $if(a + b > 0) {
        a = a + 9;
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

    Callable add = [&](Var<int> a, Var<int> b ,Var<int>) {
        a = a + b;
        return a;
    };

    Callable c1 = [&](Var<int> a, Var<int> b) {
        a += 1;
        b += 1;
//        Var<Hit> hit;
        Var<float4x4> m4;
        Var<ocarina::tuple<int, float>> tp;
        auto g0 = tp.get<0>();
        Var vec = m4.get<3>();
        m4.get<3>() = vec;
        Var<int[6]> arr;
        arr[3] = 0;
//        hit.bary.x = 0;
        add(a , a + 7 , 1);
        return a;
//        return a + b;
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    //    context.init_device("cuda");

    CppCodegen codegen;
    decltype(auto) f = c1.function();
    codegen.emit(f);
    cout << codegen.scratch().c_str();

    //    Device *device = context.device();

    return 0;
}