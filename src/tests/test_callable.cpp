//
// Created by Zero on 16/05/2022.
//

#include "dsl/common.h"
#include "core/concepts.h"
//#include "core/util.h"
#include "dsl/operators.h"
#include "dsl/func.h"
#include <iostream>



using std::cout;
using std::endl;
using namespace katana;

class ttt {
~ttt() {}
};

int main() {

    Callable callable = [&](Var<int> a, Var<int &> b)->Var<int> {
        return a + b;
    };

    auto cb = [](int, int) -> float {return 0.f;};
    cout << typeid(decltype(callable)).name();

    return 0;
}