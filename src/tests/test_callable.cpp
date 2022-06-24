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
#include "compile/cpp_codegen.h"
#include "core/platform.h"
#include "dsl/syntax_sugar.h"
#include "dsl/syntax.h"

using std::cout;
using std::endl;
using namespace ocarina;

Var<int> func(Var<int> a, Var<int> b) {

    Var cond(1);

    $switch(a) {
        $case(1) {
            $comment(daf)
            $break;
        };
        $case(2) {
            $comment(9089)
        };
        $default {
            $comment(default_)
        };
    };

    switch_(a, [&] {
        case_(2, [&] {
            $comment(adsfdsf)
        });
    });

//    $if(1) {
//        $comment(adsfadsf)
//        a = b;
//    };

//    $elif(cond) {
//        a = b;
//    }
//    $else {
//        a = 2;
//    };
//
//    if_(cond, [&] {
//        comment("this is comment");
//        a = 1;
//    }).elif_(cond, [&] {
//          a = b;
//      }).else_([&] {
//        a = b;
//    });

    return a;
}

int main(int argc, char *argv[]) {
    Callable callable = func;
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