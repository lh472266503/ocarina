//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"
//#include "e"

using namespace ocarina;
using std::cout;
using std::endl;

void func(ocarina::string_view &str) {
    str = str.substr(2);
}

struct aaaa {

    bool m;
    double a;
    bool n;
};

int main() {
    using namespace std::string_view_literals;
    auto str = "123456789"sv;

    decltype(auto) tr = TypeRegistry::instance();
    //    tr.parse_type(detail::TypeDesc<float3>::description());
    auto lst = string_split(",adf,fad,gre,ger,", ',');
    //        tr.parse_type(detail::TypeDesc<bool3>::description());
    using Test = ocarina::tuple<float3x3, int, bool, uint>;
//        using Test = float3x3;
    using Test2 = std::array<Test, 5>;
    tr.parse_type(detail::TypeDesc<Test2>::description());
//        cout << detail::TypeDesc<Test2>::description() << endl;
//    for (auto iter = tr._type_set.begin(); iter != tr._type_set.end(); ++iter) {
//        cout << (*iter)->description() << endl;
//    }
//    auto tp = std::tuple(1, 1.f, false);
cout << typeid(linear_layout_t<array<array<array<float, 2>, 2>, 2>>).name() << endl;
//cout << typeid(linear_layout_t<tuple<int, tuple<bool, float>>>).name();
//    cout << typeid(to_tuple_t<int>).name() << endl;
    return 0;
}