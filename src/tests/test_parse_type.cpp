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
    using Test = ocarina::tuple<uint, uint, float2>;
//        using Test = float3x3;
    using Test2 = ocarina::tuple<float4, float4>;
    tr.parse_type(detail::TypeDesc<Test2>::description());
//    tr.parse_type(detail::TypeDesc<Test>::description());

    return 0;
}