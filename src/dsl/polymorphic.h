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
        map<string, uint> type_to_index;
        vector<T> lst;
        uint type_num{0u};

        void set_type_index(T t) noexcept {
            t->set_type_index(obtain_index(t));
        }

        void clear() noexcept {
            type_num = 0u;
            type_to_index.clear();
            lst.clear();
        }

        [[nodiscard]] uint obtain_index(T t) noexcept {
            auto cname = typeid(*t).name();
            if (auto iter = type_to_index.find(cname); iter == type_to_index.cend()) {
                type_to_index[cname] = type_num++;
                lst.push_back(t);
            }
            return type_to_index.at(cname);
        }

        [[nodiscard]] bool empty() const noexcept {
            return lst.empty();
        }

        [[nodiscard]] auto size() const noexcept {
            return lst.size();
        }

    } _type_mgr;

public:
    void push_back(T arg) noexcept {
        _type_mgr.set_type_index(arg);
        Super::push_back(arg);
    }

    void clear() {
        Super ::clear();
        _type_mgr.clear();
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void dispatch_instance(Index &&index, const std::function<void(const T &)> &func) const noexcept {
        if (Super::empty()) [[unlikely]] { OC_ERROR_FORMAT("{} lst is empty", typeid(*this).name()); }
        comment("dispatch_instance");
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

    template<typename Index>
    requires is_integral_expr_v<Index>
    void dispatch_type(Index &&index, const std::function<void(const T &)> &func) const noexcept {
        auto lst = _type_mgr.lst;
        if (lst.empty()) [[unlikely]] { OC_ERROR_FORMAT("{} type lst is empty", typeid(*this).name()); }
        comment("dispatch_type");
        comment(typeid(*this).name());
        if (_type_mgr.size() == 1) {
            comment(typeid(*lst.at(0u)).name());
            func(lst.at(0u));
            return;
        }
        switch_(OC_FORWARD(index), [&] {
            for (int i = 0; i < lst.size(); ++i) {
                comment(typeid(*lst.at(i)).name());
                case_(i, [&] {func(*lst.at(i));break_(); });
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