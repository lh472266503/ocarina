//
// Created by Zero on 08/06/2022.
//

#pragma once

#include "codegen.h"
#include "ast/expression.h"
#include "ast/statement.h"

namespace ocarina {

class CppCodegen : public Codegen, public ExprVisitor, public StmtVisitor {

};

}// namespace ocarina