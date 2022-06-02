//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"
//#include "e"

using namespace katana;
using std::cout;
using std::endl;

void func(katana::string_view &str) {
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
    //    tr.parse_type(detail::TypeDesc<bool3>::description());
    using Test = katana::tuple<int,bool, bool>;
//    using Test = aaaa;
//    tr.parse_type(detail::TypeDesc<Test>::description());
//    cout << detail::TypeDesc<Test>::description() << endl;
cout << sizeof (Test) << endl ;
cout << alignof(Test) << endl;
//    for (auto iter = tr._type_set.begin(); iter != tr._type_set.end(); ++iter) {
//        cout << (*iter)->description() << endl;
//    }
    return 0;
}