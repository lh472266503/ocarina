//
// Created by Zero on 02/05/2022.
//

#pragma once

#include "core/stl.h"
#include "computable.h"
#include "core/basic_types.h"

namespace katana {

template<typename T>
class Var : public detail::Computable<T> {
};

}// namespace katana