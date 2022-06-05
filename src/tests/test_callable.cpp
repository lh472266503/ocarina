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

using std::cout;
using std::endl;
using namespace katana;

class ttt {
    ~ttt() {}
};



Var<int> func(Var<int> a, Var<int> b) {
    return a + b;
}

int main() {
    Callable callable = func;

    LiteralExpr::value_type a = false;

    a = 1.f;
//    auto cb = [](int, int) -> float { return 0.f; };
//    cout << typeid(decltype(callable)).name();

    return 0;
}