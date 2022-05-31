//
// Created by Zero on 30/05/2022.
//

#include "ast/type_registry.h"
#include "core/stl.h"

using namespace katana;

int main() {

    decltype(auto) tr = TypeRegistry::instance();
    tr.parse_type(detail::TypeDesc<float3>::description());

    return 0;
}