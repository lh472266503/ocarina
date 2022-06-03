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

// 基本模板
template<typename T, typename = void>
struct is_smart_pointer : std::false_type {
};

// 特化模板: 通过判断 T::-> 存在否和 T::get() 存在否来确定 T 是否一个智能指针
template<typename T>
struct is_smart_pointer<T,
                        std::void_t<decltype(std::declval<T>().operator->()),
                                    decltype(std::declval<T>().get())>> : std::true_type {
};

Var<int> func(Var<int> a, Var<int> b) {
    return a + b;
}

int main() {
    Callable callable = func;

    cout << is_smart_pointer<katana::shared_ptr<int>>::value;
    cout << typeid(std::void_t<decltype(std::declval<katana::shared_ptr<int>>().operator->()),
        decltype(std::declval<katana::shared_ptr<int>>().get())>).name();
//    auto cb = [](int, int) -> float { return 0.f; };
//    cout << typeid(decltype(callable)).name();

    return 0;
}