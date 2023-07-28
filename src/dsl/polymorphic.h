//
// Created by Zero on 06/11/2022.
//

#pragma once

#include "type_trait.h"
#include "syntax.h"
#include "core/stl.h"
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

template<typename T, typename U = float>
requires is_ptr_v<T> && std::is_base_of_v<Serializable<U>, ptr_t<T>>
class Polymorphic : public vector<T> {
public:
    using Super = vector<T>;
    using data_type = U;
    using ptr_type = ptr_t<T>;
    using datas_type = RegistrableManaged<U>;

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
        datas_type datas;
        uint counter = 0;
#ifndef NDEBUG
        string class_name;
#endif
    };

    struct {
        map<uint64_t, Object> all_object;
        map<uint64_t, TypeData> all_type;
        // Used to store a representative of each type
        vector<ptr_type *> representatives;

        void add_object(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            if (auto iter = all_type.find(hash_code); iter == all_type.cend()) {
                all_type[hash_code] = TypeData();
                all_type[hash_code].type_index = representatives.size();
                representatives.push_back(raw_ptr(t));
            }
#ifndef NDEBUG
            all_object.insert(make_pair(reinterpret_cast<uint64_t>(raw_ptr(t)), Object{all_type[hash_code].counter++, typeid(*t).name()}));
            all_type[hash_code].class_name = typeid(*t).name();
#else
            all_object.insert(make_pair(reinterpret_cast<uint64_t>(raw_ptr(t)), Object{all_type[hash_code].counter++}));
#endif
        }

        void erase(T t) noexcept {
            uint64_t hash_code = t->type_hash();
            all_type[hash_code].counter--;
            all_object.erase(reinterpret_cast<uint64_t>(raw_ptr(t)));
            if (all_type[hash_code].counter == 0) {
                all_type.erase(hash_code);
            }
        }

        void clear() noexcept {
            all_type.clear();
            all_object.clear();
            representatives.clear();
        }

        [[nodiscard]] bool empty() const noexcept { return representatives.empty(); }
        [[nodiscard]] auto size() const noexcept { return representatives.size(); }
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
        return _type_mgr.all_type.at(object->type_hash()).counter;
    }
    [[nodiscard]] size_t type_num() const noexcept { return _type_mgr.size(); }
    [[nodiscard]] size_t instance_num(uint type_id) const noexcept {
        return instance_num(_type_mgr.representatives.at(type_id));
    }
    [[nodiscard]] uint type_index(const ptr_type *object) const noexcept {
        return _type_mgr.all_type.at(object->type_hash()).type_index;
    }
    [[nodiscard]] uint data_index(const ptr_type *object) const noexcept {
        return _type_mgr.all_object.at(reinterpret_cast<uint64_t>(object)).data_index;
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
        return _type_mgr.all_type.at(object->type_hash()).datas;
    }
    [[nodiscard]] const datas_type &get_datas(const ptr_type *object) const noexcept {
        return _type_mgr.all_type.at(object->type_hash()).datas;
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
        _type_mgr.all_type.at(object->type_hash()).datas = ocarina::move(datas);
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
                for_each_representative([&](auto object) {
                    RegistrableManaged<U> data_set{resource_array};
                    set_datas(raw_ptr(object), ocarina::move(data_set));
                });
                for_each_instance([&](auto object) {
                    object->encode(get_datas(raw_ptr(object)));
                });
                for_each_representative([&](auto object) {
                    datas_type &data_set = get_datas(raw_ptr(object));
                    if (data_set.empty()) {
                        return;
                    }
                    data_set.reset_device_buffer_immediately(device);
                    data_set.register_self();
                    data_set.upload_immediately();
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
        if (Super::empty()) [[unlikely]] { OC_ERROR_FORMAT("{} lst is empty", typeid(*this).name()); }
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
        auto lst = _type_mgr.representatives;
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
                case_(i, [&] {func(lst.at(i));break_(); });
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
    [[nodiscard]] uint get_index(Func &&func) const noexcept {
        for (uint i = 0; i < Super::size(); ++i) {
            if (func(Super::at(i))) {
                return i;
            }
        }
        return InvalidUI32;
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