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
#include "ast/type.h"
#include <runtime/context.h>
#include "runtime/device.h"
#include "compile/cpp_codegen.h"
#include "core/platform.h"


using std::cout;
using std::endl;
using namespace ocarina;

Var<int> func(Var<int> a, Var<float> b) {
    Var c = a + b;
    return c + c;
}

int main(int argc, char *argv[]) {
    Callable callable = func;
    ocarina::tuple<int> a(1);
    ocarina::tuple<int, int> b(3, 9);
    fs::path path(argv[0]);
    Context context(path.parent_path());

    context.init_device("cuda");

//    auto t = detail::tuple_append(b, 10);

    CppCodegen codegen;
    auto f = callable.function();
    codegen.emit(f);
    cout << codegen.scratch().c_str();



//    Device *device = context.device();

    return 0;
}