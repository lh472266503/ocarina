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
#include "runtime/device.h"


using std::cout;
using std::endl;
using namespace katana;

Var<int> func(Var<int> a, Var<int> b) {
    return (a + b);
}

int main() {
    Callable callable = func;

    return 0;
}