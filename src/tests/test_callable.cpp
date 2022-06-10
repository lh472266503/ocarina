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


using std::cout;
using std::endl;
using namespace ocarina;

Var<int> func(Var<int> a, Var<int> b) {
    return (a + b) * a;
}

int main(int argc, char *argv[]) {
    Callable callable = func;
    fs::path path(argv[0]);
    Context context(path.parent_path());

    context.init_device("cuda");

    CppCodegen codegen;

//    Device *device = context.device();

    return 0;
}