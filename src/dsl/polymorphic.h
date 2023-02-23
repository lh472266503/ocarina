//
// Created by Zero on 06/11/2022.
//

#pragma once

#include "type_trait.h"
#include "syntax.h"
#include "core/stl.h"

namespace ocarina {

template<typename T>
class Polymorphic : public vector<T> {
public:
    using Super = vector<T>;

protected:
    struct {
        map<string, uint> _type_to_index;

//        [[nodiscard]] uint obtain_index(T t) const noexcept {
//            auto cname = typeid(*t).name();
//        }
    } _type_mgr;

public:
    template<typename Arg>
    void push_back(Arg &&arg) {
        OC_FORWARD(arg)->set_type_index(1u);
        Super::push_back(OC_FORWARD(arg));
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void dispatch(Index &&index, const std::function<void(const T &)> &func) const noexcept {
        if (Super::empty()) [[unlikely]] { OC_ERROR_FORMAT("{} lst is empty", typeid(*this).name()); }
        comment(typeid(*this).name());
        if (Super::size() == 1) {
            comment(typeid(*Super::at(0u)).name());
            func(Super::at(0u));
            return;
        }
        switch_(OC_FORWARD(index), [&] {
            for (int i = 0; i < Super::size(); ++i) {
                comment(typeid(*Super::at(i)).name());
                case_(i, [&] {func(Super::at(i));break_(); });
            }
            default_([&] {unreachable();break_(); });
        });
    }

    template<typename Func>
    void for_each(Func &&func) const noexcept {
        for (int i = 0; i < Super::size(); ++i) {
            func(Super::at(i));
        }
    }

    template<typename Func>
    void for_each(Func &&func) noexcept {
        for (int i = 0; i < Super::size(); ++i) {
            func(Super::at(i));
        }
    }
};

}// namespace ocarina