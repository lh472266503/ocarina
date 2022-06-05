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



Var<int> func(Var<int> &a, Var<int> b) {
    Var<int> c ;
    return (a + b) * c;
}

int main() {
    Callable callable = func;

    cout << typeid(decltype(callable)).name();

    return 0;
}