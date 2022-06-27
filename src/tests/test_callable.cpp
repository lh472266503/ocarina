//
// Created by Zero on 16/05/2022.
//

#include "dsl/common.h"
#include "core/concepts.h"
//#include "core/util.h"
#include "dsl/operators.h"
#include "dsl/func.h"
#include "ast/expression.h"
#include <iostream>
#include <runtime/context.h>
#include "runtime/device.h"
#include "generator/cpp_codegen.h"
#include "core/platform.h"
#include "dsl/syntax_sugar.h"
#include "dsl/syntax.h"
#include "core/util.h"

using std::cout;
using std::endl;
using namespace ocarina;

template<typename T>
auto func(T a, T b) {
    //    T ret = (a + b) * b;

    //    for_range(a,[&](auto x) {
    //    $comment(---)
    //    });

//    ::ocarina::range(a, 9 + b, 3 + a + b) / [&](auto v) noexcept {
//        v += 1;
//    };
//    Var<std::array<int, 5>> arr;
    a = (a + a * b);
//    a = arr[0];
    return a + a;

    $for(v, b, 9) {
        a += v;
    };
//    return a * 5.1f;
    //    Var cond(1);
    //    auto fun = [&]() {
    //        a += 1;
    //        return a < 15;
    //    };
    //    while_(a < 5, [&] {
    //        a += 1;
    //        $comment(sddsfdfsa)
    //    });
    //
    //    $while(fun()){
    //        $comment(sddsfd-- -- -fsa)};
    //
    //    $switch(a) {
    //        $case(1) {
    //            $comment(daf)
    //                $break;
    //        };
    //        $case(2){
    //            $comment(9089)};
    //        $default{
    //            $comment(default_)};
    //    };
    //
    switch_(a, [&] {
        case_(2, [&] {
            $comment(adsfdsf)
        });
        default_([&] {
            $comment(90890887879)
        });
    });
//
//    switch_(a)
//        .case_(2, [&] {
//            $comment(adsfdsf)
//            break_();
//        })
//        .case_(3, [&]{
//            $comment(456747567)
//        })
//        .default_([&]{
//            $comment(-------)
//        });
    //
        $if(1) {
            $comment(adsfadsf)
                a = b;
        }
        $elif(a > b) {
            a = b;
        }
        $else {
            a = 2;
        };
    //
    //    if_(cond, [&] {
    //        comment("this is comment");
    //        a = 1;
    //    }).elif_(cond, [&] {
    //          a = b;
    //      }).else_([&] {
    //        a = b;
    //    });

//    return a;
}

int main(int argc, char *argv[]) {

    Callable callable = func<Var<int>>;
    fs::path path(argv[0]);
    Context context(path.parent_path());

    context.init_device("cuda");

    CppCodegen codegen;
    decltype(auto) f = callable.function();
    codegen.emit(f);
    cout << codegen.scratch().c_str();

    //    Device *device = context.device();

    return 0;
}