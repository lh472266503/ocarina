//
// Created by Zero on 03/05/2022.
//

#pragma once

#include "core/stl.h"
#include "type.h"
#include "core/concepts.h"
#include "variable.h"

namespace sycamore::ast {

class FunctionBuilder : public sycamore::enable_shared_from_this<FunctionBuilder>,
                        public concepts::Noncopyable {

};

}// namespace sycamore::ast