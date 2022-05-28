//
// Created by Zero on 30/04/2022.
//

#include "type_registry.h"

namespace katana {
TypeRegistry &TypeRegistry::instance() noexcept {
    static TypeRegistry type_registry;
    return type_registry;
}
const Type *TypeRegistry::from(katana::string_view desc) noexcept {
    return nullptr;
}
}