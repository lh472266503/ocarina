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

using std::cout;
using std::endl;
using namespace ocarina;

template<typename T>
auto func(T a, T b) {
    Var<std::array<int ,6>> arr;
    arr[1] =  b;
    a = arr[1] + 1;
    T d = a + b;
    T c = a + b * a + 1.5f;
    return c + d;
}

int main(int argc, char *argv[]) {

    Callable callable = func<Var<int>>;
    fs::path path(argv[0]);
    Context context(path.parent_path());

//    context.init_device("cuda");

    CppCodegen codegen;
    decltype(auto) f = callable.function();
    codegen.emit(f);
    cout << codegen.scratch().c_str();

    //    Device *device = context.device();

    return 0;
}