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

    auto fun = [&]() {
        a += 1;
        return a < 15;
    };
    //    while_(a < 5, [&] {
    //        a += 1;
    //        $comment(sddsfdfsa)
    //    });

    $while(fun()){
        $comment(sddsfd-- -- -fsa)};

    //    $switch(a) {
    //        $case(1) {
    //            $comment(daf)
    //            $break;
    //        };
    //        $case(2) {
    //            $comment(9089)
    //        };
    //        $default {
    //            $comment(default_)
//        };
//    };
//
//    switch_(a, [&] {
//        case_(2, [&] {
//            $comment(adsfdsf)
//        });
//        default_([&]{
//            $comment(90890887879)
//        });
//    });
//
//    $if(1) {
//        $comment(adsfadsf)
//        a = b;
//    }
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

class Te {
public:
    Te() {
        cout << "ctor te" << endl;
    }

    ~Te() {
        std::cout << "destructor Te" << endl;
    }
};

int count() {
    cout << "count" << endl;
    return 3;
}

int step() {
    cout << "step" << endl;
    return 1;
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