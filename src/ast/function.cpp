//
// Created by Zero on 30/04/2022.
//

#include "function.h"
#include "function_builder.h"

namespace ocarina {

class Function::Impl : public concepts::Noncopyable {
private:
    const Type *_ret{nullptr};
    ocarina::vector<ocarina::unique_ptr<Expression>> _expressions;
    ocarina::vector<ocarina::unique_ptr<Statement>> _statements;
    ocarina::vector<Variable> _arguments;
    ocarina::vector<Usage> _variable_usages;
    mutable uint64_t _hash{0};
    mutable bool _hash_computed{false};
    Tag _tag{Tag::CALLABLE};

private:
    [[nodiscard]] const RefExpr *_ref(Variable variable) noexcept {
        return create_expression<RefExpr>(variable);
    }
    [[nodiscard]] uint64_t _compute_hash() const noexcept {
        return 6545689;
    }

public:
    [[nodiscard]] uint next_variable_uid() noexcept {
        auto ret = _variable_usages.size();
        _variable_usages.push_back(Usage::NONE);
        return ret;
    }
    explicit Impl(Tag tag = Tag::CALLABLE) : _tag(tag) {}
    template<typename Expr, typename... Args>
    [[nodiscard]] const Expr *create_expression(Args &&...args) {
        auto expr = ocarina::make_unique<Expr>(std::forward<Args>(args)...);
        auto ret = expr.get();
        _expressions.push_back(std::move(expr));
        return ret;
    }
    template<typename Stmt, typename... Args>
    const Stmt *_create_statement(Args &&...args) {
        auto stmt = ocarina::make_unique<Stmt>(std::forward<Args>(args)...);
        auto ret = stmt.get();
        _statements.push_back(std::move(stmt));
        return ret;
    }
    void mark_variable_usage(uint uid, Usage usage) noexcept {
        _variable_usages[uid] = usage;
    }
    [[nodiscard]] Tag tag() const noexcept { return _tag; }
    [[nodiscard]] bool is_callable() const noexcept { return _tag == Tag::CALLABLE; }
    [[nodiscard]] bool is_kernel() const noexcept { return _tag == Tag::KERNEL; }
    [[nodiscard]] uint64_t hash() const noexcept {
        if (!_hash_computed) {
            _hash = _compute_hash();
            _hash_computed = true;
        }
        return _hash;
    }
    [[nodiscard]] const RefExpr *argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::LOCAL, next_variable_uid());
        return _ref(variable);
    }
    [[nodiscard]] const RefExpr *reference_argument(const Type *type) noexcept {
        Variable variable(type, Variable::Tag::REFERENCE, next_variable_uid());
        return _ref(variable);
    }
    void return_(const Expression *expression) noexcept {
        if (expression) {
            _ret = expression->type();
        }
        _create_statement<ReturnStmt>(expression);
    }
    void assign(const Expression *lhs, const Expression *rhs) noexcept {
        _create_statement<AssignStmt>(lhs, rhs);
    }
};

ocarina::vector<Function *> Function::_function_stack() noexcept {
    static thread_local ocarina::vector<Function *> ret;
    return ret;
}

void Function::return_(const Expression *expression) noexcept {
}

Function::Function(Function::Tag tag) noexcept
    : _impl(ocarina::make_shared<Impl>(tag)) {
}

const RefExpr *Function::argument(const Type *type) noexcept {
    return _impl->argument(type);
}

const RefExpr *Function::reference_argument(const Type *type) noexcept {
    return _impl->reference_argument(type);
}

const Expression *Function::local(const Type *type) noexcept {
    return _impl->create_expression<RefExpr>(Variable(type, Variable::Tag::LOCAL,
                                                       _impl->next_variable_uid()));
}

const LiteralExpr *Function::literal(const Type *type, LiteralExpr::value_type value) noexcept {
    return _impl->create_expression<LiteralExpr>(type, value);
}

const BinaryExpr *Function::binary(const Type *type, BinaryOp op, const Expression *lhs, const Expression *rhs) noexcept {
    return _impl->create_expression<BinaryExpr>(type, op, lhs, rhs);
}

const UnaryExpr *Function::unary(const Type *type, UnaryOp op, const Expression *expression) noexcept {
    return _impl->create_expression<UnaryExpr>(type, op, expression);
}

void Function::mark_variable_usage(uint uid, Usage usage) noexcept {
    _impl->mark_variable_usage(uid, usage);
}

ocarina::span<const Variable> Function::arguments() const noexcept {
    return _builder->arguments();
}

const Type *Function::return_type() const noexcept {
    return _builder->return_type();
}

Function::Tag Function::tag() const noexcept {
    return _builder->tag();
}

bool Function::is_callable() const noexcept {
    return _builder->is_callable();
}

bool Function::is_kernel() const noexcept {
    return _builder->is_kernel();
}
void Function::assign(const Expression *lhs, const Expression *rhs) noexcept {
    _impl->assign(lhs, rhs);
}
uint64_t Function::hash() const noexcept {
    return _impl->hash();
}
}// namespace ocarina