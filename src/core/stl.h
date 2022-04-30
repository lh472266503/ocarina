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
#include <unordered_map>

namespace sycamore {

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

using string = std::string;
using std::string_view;

using std::span;
using std::vector;

using std::deque;
using std::list;
using std::map;
using std::monostate;
using std::optional;
using std::queue;
using std::unordered_map;
using std::variant;

}// namespace sycamore