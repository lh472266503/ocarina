#include <tuple>
#include <iostream>
#include <array>
#include <utility>
#include "ast/op.h"
#include "core/concepts.h"
#include "core/basic_traits.h"
#include "typeinfo"
#include "core/hash.h"
#include "core/string_util.h"
#include "ast/type_registry.h"
#include "dsl/common.h"
#include "rt/hit.h"

using namespace katana;
using namespace std;

template<typename T, size_t N, size_t... i>
auto array_to_tuple(std::array<T, N> arr, std::index_sequence<i...>) {
    //    ((cout << arr[i]),...);
    auto ret = std::tuple<remove_cvref_t<decltype(arr[i])>...>(arr[i]...);
    return ret;
}

int main() {
    using namespace katana;
    std::array<float, 2> arr = {1, 2};

    std::tuple<float, float> tp = array_to_tuple(arr, std::make_index_sequence<2>());
    //    std::tuple<float, float> tp = std::tuple<float, float>(1,5);
    //    cout << typeid(tp).name() << endl;
    //    cout << typeid(struct_member_tuple<std::array<float, 2>>::type).name();
//    cout << detail::TypeDesc<decltype(tp)>::description() << endl;
    cout << detail::TypeDesc<Hit>::description() << endl;
    //    cout << typeid(katana::tuple_join_t<tuple<int, float, int>, tuple<int, float, uint>, int>).name() << endl;
    //    cout << typeid(canonical_layout<float2x2>::type).name() << endl;
}