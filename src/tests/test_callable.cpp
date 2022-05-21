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

//    auto callable = [](Var<int> a, Var<int> b) {
//        return a + b;
//    };

cout << std::is_trivially_destructible_v<ttt>;

    return 0;
}