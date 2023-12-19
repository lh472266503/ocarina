//
// Created by Zero on 06/11/2022.
//

#pragma once

#include "type_trait.h"
#include "stmt_builder.h"
#include "core/stl.h"
#include "core/util.h"
#include "serialize.h"
#include "registrable.h"

namespace ocarina {

enum PolymorphicMode {
    EInstance = 0,
    EType = 1
};

template<EPort p>
[[nodiscard]] inline oc_uint<p> encode_id(oc_uint<p> inst_id, oc_uint<p> type_id) noexcept {
    inst_id = inst_id << 8;
    return inst_id | type_id;
}

template<EPort p>
[[nodiscard]] inline pair<oc_uint<p>, oc_uint<p>> decode_id(oc_uint<p> id) noexcept {
    oc_uint<p> inst_id = (0xffffff00 & id) >> 8;
    oc_uint<p> type_id = 0x000000ff & id;
    return std::make_pair(inst_id, type_id);
}

template<typename T>
class PolyEvaluator : public vector<UP<T>> {
public:
    using Super = vector<T>;
    using element_ty = T;

protected:
    Uint _tag{};
    ocarina::unordered_map<uint64_t, uint> _tags;

public:
    template<typename Derive>
    requires std::is_base_of_v<element_ty, Derive>
    Derive *link(uint64_t hash, UP<Derive> elm) noexcept {
        auto [iter, first] = _tags.try_emplace(hash, Super::size());
        _tag = _tags[hash];
        uint index = _tags[hash];
        Derive *ret = nullptr;
        if (first) {
            ret = elm.get();
            Super::push_back(ocarina::move(elm));
        } else {
            ret = dynamic_cast<Derive *>(Super::at(index));
            *ret = *elm.get();
        }
        OC_ASSERT(ret != nullptr);
        return ret;
    }

    template<typename Derive>
    requires std::is_base_of_v<element_ty, Derive>
    Derive *link(UP<Derive> elm) noexcept {
        return link(elm->type_hash(), elm);
    }

    template<typename Func>
    void dispatch(Func &&func) noexcept {
        comment(ocarina::format("PolyEvaluator dispatch {}", typeid(T).name()));
        if (Super::size() == 1) {
            comment(typeid(*Super::at(0u)).name());
            func(raw_ptr(Super::at(0u)));
        }
        switch_(_tag, [&] {
            for (int i = 0; i < Super::size(); ++i) {
                comment(typeid(*Super::at(i)).name());
                case_(i, [&] {func(raw_ptr(Super::at(i)));break_(); });
            }
            default_([&] { unreachable();break_(); });
        });
    }
};

template<typename T, typename U = float>
requires is_ptr_v<T> && std::is_base_of_v<Serializable<U>, ptr_t<T>>
class Polymorphic : public vector<T> {
public:
    using Super = vector<T>;
    using data_type = U;
    using ptr_type = ptr_t<T>;
    using datas_type = RegistrableManaged<U>;

protected:
    struct TypeData {
        string class_name;
        datas_type data_set;
        vector<ptr_type *> objects;
    };

    struct {
        map<uint64_t, TypeData> type_map;

        void add_object(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            if (auto iter = type_map.find(hash_code); iter == type_map.cend()) {
                type_map[hash_code] = TypeData();
                type_map[hash_code].class_name = typeid(*t).name();
            }
            type_map[hash_code].objects.push_back(raw_ptr(t));
        }

        [[nodiscard]] uint type_index(uint64_t hash_code) const noexcept {
            uint cursor = 0;
            for (auto iter = type_map.cbegin(); iter != type_map.cend(); ++iter, ++cursor) {
                if (hash_code == iter->first) {
                    return cursor;
                }
            }
            OC_ASSERT(false);
            return InvalidUI32;
        }

        [[nodiscard]] uint64_t type_hash(uint type_index) const noexcept {
            uint cursor = 0;
            for (auto iter = type_map.cbegin(); iter != type_map.cend(); ++iter, ++cursor) {
                if (cursor == type_index) {
                    return iter->first;
                }
            }
            OC_ASSERT(false);
            return InvalidUI64;
        }

        template<typename Func>
        void for_each_type(Func &&func) const noexcept {
            if constexpr (std::invocable<Func, TypeData, uint>) {
                uint cursor = 0u;
                for (auto iter = type_map.cbegin(); iter != type_map.cend(); ++iter, ++cursor) {
                    func(iter->second, cursor);
                }
            } else {
                for (const auto &it : type_map) {
                    func(it.second);
                }
            }
        }

        template<typename Func>
        void for_each_type(Func &&func) noexcept {
            if constexpr (std::invocable<Func, TypeData, uint>) {
                uint cursor = 0u;
                for (auto iter = type_map.begin(); iter != type_map.end(); ++iter, ++cursor) {
                    func(iter->second, cursor);
                }
            } else {
                for (auto &it : type_map) {
                    func(it.second);
                }
            }
        }

        template<typename Func>
        void for_each_representative(Func &&func) const noexcept {
            if constexpr (std::invocable<Func, ptr_type *, uint>) {
                uint cursor = 0u;
                for (auto iter = type_map.cbegin(); iter != type_map.cend(); ++iter, ++cursor) {
                    func(iter->second.objects[0], cursor);
                }
            } else {
                for (const auto &it : type_map) {
                    func(it.second.objects[0]);
                }
            }
        }

        template<typename Func>
        void for_each_representative(Func &&func) noexcept {
            if constexpr (std::invocable<Func, ptr_type *, uint>) {
                uint cursor = 0u;
                for (auto iter = type_map.begin(); iter != type_map.end(); ++iter, ++cursor) {
                    func(iter->second.objects[0], cursor);
                }
            } else {
                for (const auto &it : type_map) {
                    func(it.second.objects[0]);
                }
            }
        }

        void erase(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            if (auto iter = type_map.find(hash_code); iter == type_map.cend()) {
                OC_ASSERT(false);
            }
            auto &lst = type_map[hash_code].objects;

            erase_if(lst, [&](auto elm) {
                return elm == raw_ptr(t);
            });

            if (lst.size() == 0) {
                type_map.erase(hash_code);
            }
        }

        void clear() noexcept {
            type_map.clear();
        }

        [[nodiscard]] bool empty() const noexcept { return type_map.empty(); }
        [[nodiscard]] auto size() const noexcept { return type_map.size(); }
    } _type_mgr;
    PolymorphicMode _mode{EInstance};

public:
    explicit Polymorphic(PolymorphicMode mode = EInstance) : _mode(mode) {}
    void push_back(T arg) noexcept {
        _type_mgr.add_object(arg);
        Super::push_back(arg);
    }

    Super::iterator erase(Super::iterator iter) noexcept {
        _type_mgr.erase(*iter);
        return Super::erase(iter);
    }

    void clear() {
        Super ::clear();
        _type_mgr.clear();
    }

    [[nodiscard]] size_t all_instance_num() const noexcept { return Super::size(); }
    [[nodiscard]] uint instance_num(const ptr_type *object) const noexcept {
        uint64_t hash_code = object->type_hash();
        return _type_mgr.type_map.at(hash_code).objects.size();
    }
    [[nodiscard]] size_t type_num() const noexcept { return _type_mgr.size(); }
    [[nodiscard]] size_t instance_num(uint type_idx) const noexcept {
        uint64_t hash_code = _type_mgr.type_hash(type_idx);
        return _type_mgr.type_map.at(hash_code).objects.size();
    }
    [[nodiscard]] uint type_index(const ptr_type *object) const noexcept {
        uint64_t hash_code = object->type_hash();
        return _type_mgr.type_index(hash_code);
    }
    [[nodiscard]] uint data_index(const ptr_type *object) const noexcept {
        uint64_t hash_code = object->type_hash();
        return ocarina::get_index(_type_mgr.type_map.at(hash_code).objects, [&](auto obj) {
            return object == raw_ptr(obj);
        });
    }
    [[nodiscard]] DataAccessor<U> data_accessor(const ptr_type *object,
                                                const Uint &data_index) noexcept {
        return {data_index * object->element_num() * uint(sizeof(U)), get_datas(object)};
    }
    [[nodiscard]] DataAccessor<U> data_accessor(const ptr_type *object,
                                                const Uint &data_index) const noexcept {
        return {data_index * object->element_num() * uint(sizeof(U)), get_datas(object)};
    }
    [[nodiscard]] datas_type &get_datas(const ptr_type *object) noexcept {
        return _type_mgr.type_map.at(object->type_hash()).data_set;
    }
    [[nodiscard]] const datas_type &get_datas(const ptr_type *object) const noexcept {
        return _type_mgr.type_map.at(object->type_hash()).data_set;
    }

    template<typename Func>
    [[nodiscard]] auto find_if(Func &&func) const noexcept {
        return std::find_if(Super::cbegin(), Super::cend(), OC_FORWARD(func));
    }

    template<typename Func>
    [[nodiscard]] auto find_if(Func &&func) noexcept {
        return std::find_if(Super::begin(), Super::end(), OC_FORWARD(func));
    }

    /**
     * update data to managed memory
     * tips: Called on the host side code
     */
    void update(const ptr_type *object) noexcept {
        // todo
    }

    /**
     * update data to managed memory
     * tips: Called on the host side code
     */
    void update() noexcept {
        // todo
    }

    void set_datas(const ptr_type *object, datas_type &&datas) noexcept {
        _type_mgr.type_map.at(object->type_hash()).data_set = ocarina::move(datas);
    }
    void set_mode(PolymorphicMode mode) noexcept { _mode = mode; }
    [[nodiscard]] PolymorphicMode mode() const noexcept { return _mode; }
    [[nodiscard]] uint encode_id(uint id, const ptr_type *object) const noexcept {
        switch (_mode) {
            case EInstance:
                return ocarina::encode_id<H>(id, type_index(object));
            case EType:
                return ocarina::encode_id<H>(data_index(object), type_index(object));
        }
        OC_ASSERT(false);
        return InvalidUI32;
    }

    void prepare(ResourceArray &resource_array, Device &device) noexcept {
        switch (_mode) {
            case EInstance: break;
            case EType: {
                _type_mgr.for_each_type([&](TypeData &type_data) {
                    type_data.data_set.set_resource_array(resource_array);
                    type_data.data_set.reserve(type_data.objects.size() * type_data.objects[0]->element_num());
                    for (ptr_type *object : type_data.objects) {
                        object->encode(type_data.data_set);
                    }
                    type_data.data_set.reset_device_buffer_immediately(device);
                    type_data.data_set.register_self();
                    if (!type_data.data_set.empty()) {
                        type_data.data_set.upload_immediately();
                    }
                });
                break;
            }
            default: OC_ASSERT(false);
        }
    }

    template<typename ObjectID, typename Func>
    requires is_integral_expr_v<ObjectID>
    void dispatch(ObjectID &&object_id, const Func &func) const noexcept {
        auto [inst_id, type_id] = ocarina::decode_id<D>(OC_FORWARD(object_id));
        dispatch(type_id, inst_id, func);
    }

    template<typename TypeID, typename InstanceID, typename Func>
    requires is_all_integral_expr_v<TypeID, InstanceID>
    void dispatch(TypeID &&type_id, InstanceID &&inst_id, const Func &func) const noexcept {
        switch (_mode) {
            case EInstance: {
                dispatch_instance(OC_FORWARD(inst_id), [&](auto object) {
                    func(object);
                });
                break;
            }
            case EType: {
                dispatch_representative(OC_FORWARD(type_id), [&](auto object) {
                    DataAccessor<U> da = data_accessor(object, OC_FORWARD(inst_id));
                    object->decode(&da);
                    func(object);
                    object->reset_device_value();
                });
                break;
            }
            default: OC_ASSERT(false);
        }
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void dispatch_instance(Index &&index, const std::function<void(const ptr_type *)> &func) const noexcept {
        if (Super::empty()) [[unlikely]] {
            return;
        }
        comment("dispatch_instance");
        comment(typeid(*this).name());
        if (Super::size() == 1) {
            comment(typeid(*Super::at(0u)).name());
            func(raw_ptr(Super::at(0u)));
            return;
        }
        switch_(OC_FORWARD(index), [&] {
            for (int i = 0; i < Super::size(); ++i) {
                comment(typeid(*Super::at(i)).name());
                case_(i, [&] {func(raw_ptr(Super::at(i)));break_(); });
            }
            default_([&] {unreachable();break_(); });
        });
    }

    template<typename Index>
    requires is_integral_expr_v<Index>
    void dispatch_representative(Index &&index, const std::function<void(const ptr_type *)> &func) const noexcept {
        if (_type_mgr.empty()) [[unlikely]] {
            return;
        }
        comment("dispatch_type");
        comment(typeid(*this).name());
        if (_type_mgr.size() == 1) {
            ptr_type *elm = _type_mgr.type_map.begin()->second.objects[0];
            comment(typeid(*elm).name());
            func(elm);
            return;
        }
        switch_(OC_FORWARD(index), [&] {
            _type_mgr.for_each_representative([&](ptr_type *elm, uint i) {
                comment(typeid(*elm).name());
                case_(i, [&] {
                    func(elm);
                    break_();
                });
            });
            default_([&] {unreachable();break_(); });
        });
    }

    template<typename Func>
    void for_each_instance(Func &&func) const noexcept {
        if constexpr (std::invocable<Func, T, uint>) {
            for (uint i = 0; i < Super::size(); ++i) {
                func(Super::at(i), i);
            }
        } else {
            for (uint i = 0; i < Super::size(); ++i) {
                func(Super::at(i));
            }
        }
    }

    template<typename Func>
    [[nodiscard]] uint get_index(Func &&func) const noexcept {
        return ocarina::get_index(*this, OC_FORWARD(func));
    }

    template<typename Func>
    void for_each_instance(Func &&func) noexcept {
        if constexpr (std::invocable<Func, T, uint>) {
            for (uint i = 0; i < Super::size(); ++i) {
                func(Super::at(i), i);
            }
        } else {
            for (uint i = 0; i < Super::size(); ++i) {
                func(Super::at(i));
            }
        }
    }

    template<typename Func>
    void for_each_representative(Func &&func) const noexcept {
        _type_mgr.for_each_representative(OC_FORWARD(func));
    }

    template<typename Func>
    void for_each_representative(Func &&func) noexcept {
        _type_mgr.for_each_representative(OC_FORWARD(func));
    }
};

}// namespace ocarina