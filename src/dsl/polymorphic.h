//
// Created by Zero on 06/11/2022.
//

#pragma once

#include "type_trait.h"
#include "syntax.h"
#include "core/stl.h"
#include "rhi/managed.h"

namespace ocarina {

template<typename T, typename U = float>
class Polymorphic : public vector<T> {
public:
    using Super = vector<T>;
    using data_type = ManagedWrapper<U>;

protected:
    struct Index {
        // index of type
        uint type_index;
        // Index in a list of the same type
        uint index;
    };

    struct {
        map<uint64_t, Index> inst_to_index;

        map<uint64_t, uint> type_to_index;
        // Used to store a representative of each type
        vector<T> representatives;
        // Each type has a managed used to store data
        vector<ManagedWrapper<U>> datas;

        void set_type_index(T t) noexcept {
            t->set_type_index(obtain_index(t));
            
        }

        void clear() noexcept {
            type_to_index.clear();
            representatives.clear();
        }

        [[nodiscard]] uint obtain_index(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            if (auto iter = type_to_index.find(hash_code); iter == type_to_index.cend()) {
                type_to_index[hash_code] = representatives.size();
                representatives.push_back(t);
            }
            return type_to_index.at(hash_code);
        }

        [[nodiscard]] bool empty() const noexcept { return representatives.empty(); }
        [[nodiscard]] auto size() const noexcept { return representatives.size(); }
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

    void add_type_data(data_type data) noexcept { _type_mgr.datas.push_back(move(data)); }
    [[nodiscard]] data_type &type_datas() noexcept { return _type_mgr.datas; }
    [[nodiscard]] const data_type &type_datas() const noexcept { return _type_mgr.datas; }
    [[nodiscard]] size_t instance_num() const noexcept { return Super::size(); }
    [[nodiscard]] size_t type_num() const noexcept { return _type_mgr.size(); }

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
    void for_each_instance(Func &&func) const noexcept {
        for (int i = 0; i < Super::size(); ++i) {
            func(Super::at(i));
        }
    }

    template<typename Func>
    void for_each_instance(Func &&func) noexcept {
        for (int i = 0; i < Super::size(); ++i) {
            func(Super::at(i));
        }
    }

    template<typename Func>
    void for_each_representative(Func &&func) const noexcept {
        for (const auto &elm : _type_mgr.representatives) {
            func(elm);
        }
    }

    template<typename Func>
    void for_each_representative(Func &&func) noexcept {
        for (auto &elm : _type_mgr.representatives) {
            func(elm);
        }
    }
};

}// namespace ocarina