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

// ����ģ��
template<typename T, typename = void>
struct is_smart_pointer : std::false_type {
};

// �ػ�ģ��: ͨ���ж� T::-> ���ڷ�� T::get() ���ڷ���ȷ�� T �Ƿ�һ������ָ��
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

    auto cb = [](int, int) -> float { return 0.f; };
    cout << typeid(decltype(callable)).name();

    return 0;
}