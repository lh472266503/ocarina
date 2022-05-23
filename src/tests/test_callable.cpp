//
// Created by Zero on 16/05/2022.
//

#include "dsl/common.h"
#include "core/concepts.h"
//#include "core/util.h"
#include "dsl/operators.h"
#include <iostream>



using std::cout;
using std::endl;
using namespace katana;

class ttt {
~ttt() {}
};

int main() {

    Callable<int(int, int)> callable = [&](Var<int> a, Var<int> b) {
        return a + b;
    };


    return 0;
}