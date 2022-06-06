//
// Created by Zero on 30/04/2022.
//

#pragma once

#include "header.h"
#include <functional>
#include <deque>
#include <queue>
#include <variant>
#include <span>
#include <vector>
#include <map>
#include <set>
#include <array>
#include <iostream>
#include <unordered_map>
#include <memory>
#include <string>
#include <unordered_set>
#include <string_view>
#include <filesystem>
#include <EASTL/tuple.h>

namespace katana {

// io
using std::cout;
using std::endl;

// ptr
using std::const_pointer_cast;
using std::dynamic_pointer_cast;
using std::enable_shared_from_this;
using std::function;
using std::make_shared;
using std::make_unique;
using std::reinterpret_pointer_cast;
using std::shared_ptr;
using std::static_pointer_cast;
using std::unique_ptr;
using std::weak_ptr;

// string
using std::string;
using std::string_view;

// range and container
using std::span;
using std::vector;
using std::allocator;
using std::deque;
using std::list;
using std::map;
using std::set;
using std::optional;
using std::queue;
using std::unordered_map;
using std::array;
using std::unordered_set;


// tuple
using eastl::tuple;
using eastl::tuple_size;
using eastl::tuple_size_v;
using eastl::tuple_element;
using eastl::tuple_element_t;
using eastl::get;

// sequence
using std::make_index_sequence;
using std::make_integer_sequence;
using std::index_sequence;
using std::integer_sequence;
using std::index_sequence_for;

// other
using std::monostate;
using std::variant;
using std::visit;
namespace fs = std::filesystem;

}// namespace katana