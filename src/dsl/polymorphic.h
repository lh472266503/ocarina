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
requires std::is_pointer_v<std::remove_cvref_t<T>>
class Polymorphic : public vector<T> {
public:
    using Super = vector<T>;
    using data_type = ManagedWrapper<U>;

protected:
    struct Object {
    public:
        // Index in a list of the same type
        uint data_index{};
#ifndef NDEBUG
        string class_name;
#endif
    };

    struct TypeData {
        // index of type
        uint type_index{};
        // used to store current type data
        data_type datas;
#ifndef NDEBUG
        string class_name;
#endif
    };

    struct {
        map<uint64_t, uint> type_counter;
        map<uint64_t, Object> all_object;
        map<uint64_t, TypeData> all_type;
        // Used to store a representative of each type
        vector<T> representatives;

        void add_object(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            if (auto iter = all_type.find(hash_code); iter == all_type.cend()) {
                all_type[hash_code] = TypeData();
                all_type[hash_code].type_index = representatives.size();
                representatives.push_back(t);
                type_counter[hash_code] = 0;
            }
#ifndef NDEBUG
            all_object.insert(make_pair(reinterpret_cast<uint64_t>(t), Object{type_counter[hash_code]++, typeid(*t).name()}));
            all_type[hash_code].class_name = typeid(*t).name();
#else
            all_object.insert(make_pair(reinterpret_cast<uint64_t>(t), Object{data_index}));
#endif
        }

        void clear() noexcept {
            all_type.clear();
            all_object.clear();
            type_counter.clear();
            representatives.clear();
        }

        [[nodiscard]] bool empty() const noexcept { return representatives.empty(); }
        [[nodiscard]] auto size() const noexcept { return representatives.size(); }
    } _type_mgr;

public:
    void push_back(T arg) noexcept {
        _type_mgr.add_object(arg);
        Super::push_back(arg);
    }

    void clear() {
        Super ::clear();
        _type_mgr.clear();
    }

    [[nodiscard]] size_t instance_num() const noexcept { return Super::size(); }
    [[nodiscard]] size_t type_num() const noexcept { return _type_mgr.size(); }
    [[nodiscard]] uint type_index(const std::remove_pointer_t<T> *object) const noexcept {
        return _type_mgr.all_type.at(object->type_hash()).type_index;
    }

    void prepare(ResourceArray &resource_array) noexcept {
        for (TypeData &type_data : _type_mgr.all_type) {
            type_data.datas.init(resource_array);
        }
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