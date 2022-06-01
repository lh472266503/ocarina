//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"

using namespace katana;
using std::cout;
using std::endl;

void func(katana::string_view &str) {
    str = str.substr(2);
}

int main() {
    using namespace std::string_view_literals;
    auto str = "123456789"sv;

    decltype(auto) tr = TypeRegistry::instance();
    //    tr.parse_type(detail::TypeDesc<float3>::description());
    auto lst = string_split(",adf,fad,gre,ger,", ',');
    //    tr.parse_type(detail::TypeDesc<bool3>::description());
    using Test = std::tuple<float, int, uint>;
    tr.parse_type(detail::TypeDesc<Test>::description());
    for (auto iter = tr._type_set.begin(); iter != tr._type_set.end(); ++iter) {
        cout << (*iter)->description() << endl;
    }
    return 0;
}