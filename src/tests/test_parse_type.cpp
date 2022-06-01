//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"

using namespace katana;
using std::cout;
using std::endl;
int main() {

    decltype(auto) tr = TypeRegistry::instance();
//    using test_type = std:
    tr.parse_type(detail::TypeDesc<float3>::description());
    auto lst = string_split(",adf,fad,gre,ger,", ',');
    tr.parse_type(detail::TypeDesc<int3>::description());
    for (auto iter = tr._type_set.begin(); iter != tr._type_set.end(); ++iter) {
        cout << (*iter)->description() << endl;
    }
//    cout << tr._types.size() << endl;
    return 0;
}