
#pragma once

namespace ocarina {
template<typename... Ts>
struct always_false {
    static constexpr bool value = false;
};

template<typename... Ts>
static constexpr bool always_false_v = always_false<Ts...>::value;

template<size_t... Ints>
struct index_sequence {};

template<size_t N, size_t... Ints>
struct make_index_sequence_helper : make_index_sequence_helper<N - 1, N - 1, Ints...> {
};

template<size_t... Ints>
struct make_index_sequence_helper<0, Ints...> {
    using type = index_sequence<Ints...>;
};

template<size_t N>
using make_index_sequence = typename make_index_sequence_helper<N>::type;

template<bool B, typename T = void>
struct enable_if {};

template<typename T>
struct enable_if<true, T> {
    using type = T;
};

template<bool B, typename T = void>
using enable_if_t = typename enable_if<B, T>::type;

namespace detail {
template<typename T, size_t N>
struct VectorStorage {
    static_assert(always_false_v<T>, "Invalid vector storage");
};

template<typename T>
struct alignas(sizeof(T) * 2) VectorStorage<T, 2> {
    T x{}, y{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s} {}
    __device__ constexpr VectorStorage(T x, T y) noexcept : x{x}, y{y} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 3> {
    T x{}, y{}, z{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s}, z{s} {}
    __device__ constexpr VectorStorage(T x, T y, T z) noexcept : x{x}, y{y}, z{z} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]}, z{ptr[2]} {}
};

template<typename T>
struct alignas(sizeof(T) * 4) VectorStorage<T, 4> {
    T x{}, y{}, z{}, w{};
    __device__ constexpr VectorStorage(T s = T{}) noexcept : x{s}, y{s}, z{s}, w{s} {}
    __device__ constexpr VectorStorage(T x, T y, T z, T w) noexcept : x{x}, y{y}, z{z}, w{w} {}
    __device__ constexpr VectorStorage(const T *ptr) noexcept : x{ptr[0]}, y{ptr[1]}, z{ptr[2]}, w{ptr[3]} {}
};
}// namespace detail

namespace detail {

}// namespace detail

template<typename T, size_t N>
struct Vector : public detail::VectorStorage<T, N> {
    using detail::VectorStorage<T, N>::VectorStorage;

private:
    template<typename U, size_t NN, size_t... i>
    static Vector<T, N> construct_helper(Vector<U, NN> v,
                                         ocarina::index_sequence<i...>) {
        return Vector<T, N>(static_cast<T>(v[i])...);
    }

public:
    template<typename U>
    explicit constexpr Vector(U s) noexcept : Vector(static_cast<T>(s)) {}

    template<typename U, size_t NN, ocarina::enable_if_t<NN >= N, int> = 0>
    explicit constexpr Vector(Vector<U, NN> v)
        : Vector{construct_helper(v, ocarina::make_index_sequence<M>())} {}

    __device__ constexpr T &operator[](size_t index) noexcept { return (&(this->x))[index]; }
    __device__ constexpr const T &operator[](size_t index) const noexcept { return (&(this->x))[index]; }
};

using uint = unsigned int;
using uint64t = unsigned long long;
using uchar = unsigned char;
using ushort = unsigned short;

#define OC_MAKE_VECTOR_TYPES(T) \
    using T##2 = Vector<T, 2>;  \
    using T##3 = Vector<T, 3>;  \
    using T##4 = Vector<T, 4>;

OC_MAKE_VECTOR_TYPES(bool)
OC_MAKE_VECTOR_TYPES(float)
OC_MAKE_VECTOR_TYPES(int)
OC_MAKE_VECTOR_TYPES(char)
OC_MAKE_VECTOR_TYPES(short)
OC_MAKE_VECTOR_TYPES(ushort)
OC_MAKE_VECTOR_TYPES(uchar)
OC_MAKE_VECTOR_TYPES(uint)
OC_MAKE_VECTOR_TYPES(uint64t)

#undef OC_MAKE_VECTOR_TYPES

}// namespace ocarina

using oc_int = int;
using oc_uint = unsigned int;
using oc_float = float;
using oc_bool = bool;
using oc_uchar = unsigned char;
using oc_ushort = unsigned short;
using oc_uint64t = unsigned long long;

#define OC_MAKE_VECTOR_N(type, dim) using type##dim = ocarina::Vector<type, dim>;

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator+(const ocarina::Vector<T, N> v) noexcept {
    return v;
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator-(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{-v.x, -v.y};
    } else if constexpr (N == 3) {
        return R{-v.x, -v.y, -v.z};
    } else {
        return R{-v.x, -v.y, -v.z, -v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto operator!(const ocarina::Vector<T, N> v) noexcept {
    if constexpr (N == 2u) {
        return ocarina::Vector<bool, 2>{!v.x, !v.y};
    } else if constexpr (N == 3u) {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z};
    } else {
        return ocarina::Vector<bool, 3>{!v.x, !v.y, !v.z, !v.w};
    }
}

template<typename T, size_t N>
[[nodiscard]] __device__ constexpr auto
operator~(const ocarina::Vector<T, N> v) noexcept {
    using R = ocarina::Vector<T, N>;
    if constexpr (N == 2) {
        return R{~v.x, ~v.y};
    } else if constexpr (N == 3) {
        return R{~v.x, ~v.y, ~v.z};
    } else {
        return R{~v.x, ~v.y, ~v.z, ~v.w};
    }
}

#define OC_MAKE_VECTOR(type)  \
    OC_MAKE_VECTOR_N(type, 2) \
    OC_MAKE_VECTOR_N(type, 3) \
    OC_MAKE_VECTOR_N(type, 4)

OC_MAKE_VECTOR(oc_int)
OC_MAKE_VECTOR(oc_uint)
OC_MAKE_VECTOR(oc_float)
OC_MAKE_VECTOR(oc_bool)
OC_MAKE_VECTOR(oc_uchar)
OC_MAKE_VECTOR(oc_ushort)
OC_MAKE_VECTOR(oc_uint64t)

#define OC_MAKE_VECTOR_BINARY_OPERATOR(op, ...)                          \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<U, N> rhs) noexcept { \
        using ret_type = decltype(T{} + U{});                            \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<ret_type, 2>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<ret_type, 3>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<ret_type, 4>{                         \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, U rhs) noexcept {             \
        return lhs op ocarina::Vector<U, N>{rhs};                        \
    }                                                                    \
    template<typename T, typename U, size_t N>                           \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(T lhs, ocarina::Vector<U, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }

OC_MAKE_VECTOR_BINARY_OPERATOR(+, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(-, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(*, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(/, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(%, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(>>, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(<<, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(|, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(&, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_BINARY_OPERATOR(^, ocarina::is_all_integral_v<T, U>)

#define OC_MAKE_VECTOR_ASSIGN_OPERATOR(op, ...)                           \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
        ocarina::Vector<T, N> &lhs, ocarina::Vector<U, N> rhs) noexcept { \
        lhs.x op rhs.x;                                                   \
        lhs.y op rhs.y;                                                   \
        if constexpr (N >= 3) { lhs.z op rhs.z; }                         \
        if constexpr (N == 4) { lhs.w op rhs.w; }                         \
        return (lhs);                                                     \
    }                                                                     \
    template<typename T, typename U, size_t N>                            \
    __device__ constexpr decltype(auto) operator op(                      \
        ocarina::Vector<T, N> &lhs, U rhs) noexcept {                     \
        return (lhs op ocarina::Vector<U, N>{rhs});                       \
    }

OC_MAKE_VECTOR_ASSIGN_OPERATOR(+=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(-=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(*=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(/=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(%=, ocarina::is_all_number_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(<<=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(>>=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(|=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(&=, ocarina::is_all_integral_v<T, U>)
OC_MAKE_VECTOR_ASSIGN_OPERATOR(^=, ocarina::is_all_integral_v<T, U>)

#define OC_MAKE_VECTOR_LOGIC_OPERATOR(op, ...)                           \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(                                                         \
        ocarina::Vector<T, N> lhs, ocarina::Vector<T, N> rhs) noexcept { \
        if constexpr (N == 2) {                                          \
            return ocarina::Vector<bool, 2>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y};                                         \
        } else if constexpr (N == 3) {                                   \
            return ocarina::Vector<bool, 3>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z};                                         \
        } else {                                                         \
            return ocarina::Vector<bool, 4>{                             \
                lhs.x op rhs.x,                                          \
                lhs.y op rhs.y,                                          \
                lhs.z op rhs.z,                                          \
                lhs.w op rhs.w};                                         \
        }                                                                \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(ocarina::Vector<T, N> lhs, T rhs) noexcept {             \
        return lhs op ocarina::Vector<T, N>{rhs};                        \
    }                                                                    \
    template<typename T, size_t N>                                       \
    [[nodiscard]] __device__ constexpr auto                              \
    operator op(T lhs, ocarina::Vector<T, N> rhs) noexcept {             \
        return ocarina::Vector<T, N>{lhs} op rhs;                        \
    }
OC_MAKE_VECTOR_LOGIC_OPERATOR(||, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(&&, ocarina::is_all_boolean_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(==, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(!=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(<=, ocarina::is_all_number_v<T>)
OC_MAKE_VECTOR_LOGIC_OPERATOR(>=, ocarina::is_all_number_v<T>)

template<typename T, oc_uint N>
class oc_array {
private:
    T _data[N];

public:
    template<typename... Elem>
    __device__ constexpr oc_array(Elem... elem) noexcept : _data{elem...} {}
    __device__ constexpr oc_array(oc_array &&) noexcept = default;
    __device__ constexpr oc_array(const oc_array &) noexcept = default;
    __device__ constexpr oc_array &operator=(oc_array &&) noexcept = default;
    __device__ constexpr oc_array &operator=(const oc_array &) noexcept = default;
    __device__ constexpr T *data() noexcept { return &_data[0]; }
    __device__ constexpr const T *data() const noexcept { return &_data[0]; }
    [[nodiscard]] __device__ T &operator[](size_t i) noexcept { return _data[i]; }
    [[nodiscard]] __device__ T operator[](size_t i) const noexcept { return _data[i]; }
};

namespace ocarina {

template<size_t N, size_t M = N>
struct Matrix {
public:
    static constexpr auto RowNum = M;
    static constexpr auto ColNum = N;
    static constexpr auto ElementNum = M * N;
    using scalar_type = float;
    using vector_type = Vector<scalar_type, M>;
    using array_t = oc_array<vector_type, N>;

private:
    oc_array<vector_type, N> cols_{};

public:
    template<size_t... i>
    [[nodiscard]] static constexpr array_t diagonal_helper(index_sequence<i...>, scalar_type s) noexcept {
        array_t ret{};
        if constexpr (M == N) {
            ((ret[i][i] = s), ...);
        }
        return ret;
    }
    constexpr Matrix(scalar_type s = 1) noexcept
        : cols_(diagonal_helper(make_index_sequence<N>(), s)) {
    }

    template<typename... Args, enable_if_t<sizeof...(Args) == N, int> = 0>
    constexpr Matrix(Args... args) noexcept : cols_(oc_array<vector_type, N>{args...}) {}

    template<size_t NN, size_t MM, size_t ...i>
    [[nodiscard]] static constexpr auto construct_helper(Matrix<NN, MM> mat, index_sequence<i...>) {
        return oc_array<Vector<float, M>, N>{Vector<float, M>{mat[i]}...};
    }

    template<size_t NN, size_t MM, enable_if_t<(NN >= N && MM >= M), int> = 0>
    explicit constexpr Matrix(Matrix<NN, MM> mat) noexcept
        : cols_{construct_helper(mat, make_index_sequence<N>)} {}

    template<size_t... i>
    [[nodiscard]] static constexpr array_t construct_helper(ocarina::index_sequence<i...>,
                                                            const oc_array<scalar_type, ElementNum> &arr) {
        return array_t{vector_type{&(arr.data()[i * M])}...};
    }

    template<typename... Args, enable_if_t<sizeof...(Args) == ElementNum, int> = 0>
    explicit constexpr Matrix(Args &&...args) noexcept
        : cols_(construct_helper(ocarina::make_index_sequence<N>(),
                                 oc_array<scalar_type, ElementNum>{static_cast<scalar_type>(args)...})) {}

    [[nodiscard]] constexpr vector_type &operator[](size_t i) noexcept { return cols_[i]; }
    [[nodiscard]] constexpr const vector_type &operator[](size_t i) const noexcept { return cols_[i]; }
};

#define OC_MAKE_MATRIX(N, M) using float##N##x##M = Matrix<N, M>;

OC_MAKE_MATRIX(2, 2)
OC_MAKE_MATRIX(2, 3)
OC_MAKE_MATRIX(2, 4)
OC_MAKE_MATRIX(3, 2)
OC_MAKE_MATRIX(3, 3)
OC_MAKE_MATRIX(3, 4)
OC_MAKE_MATRIX(4, 2)
OC_MAKE_MATRIX(4, 3)
OC_MAKE_MATRIX(4, 4)

#undef OC_MAKE_MATRIX

}// namespace ocarina

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto negate_matrix_impl(const ocarina::Matrix<N, M> &m, ocarina::index_sequence<i...>) {
    return ocarina::Matrix<N, M>{(-m[i])...};
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator-(ocarina::Matrix<N, M> m) {
    return ocarina::negate_matrix_impl(m, ocarina::make_index_sequence<N>());
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto multiply_impl(const ocarina::Matrix<N, M> &m, float s, ocarina::index_sequence<i...>) {
    return ocarina::Matrix<N, M>((m[i] * s)...);
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, M> m, float s) {
    return ocarina::multiply_impl(m, s, ocarina::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(float s, ocarina::Matrix<N, M> m) {
    return m * s;
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator/(ocarina::Matrix<N, M> m, float s) {
    return m * (1.0f / s);
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto multiply_impl(const ocarina::Matrix<N, M> &m, const ocarina::Vector<float, N> &v,
                                        ocarina::index_sequence<i...>) noexcept {
    return ((v[i] * m[i]) + ...);
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, M> m, ocarina::Vector<float, N> v) noexcept {
    return ocarina::multiply_impl(m, v, ocarina::make_index_sequence<N>());
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto multiply_matrices_impl(const ocarina::Matrix<N, M> &lhs, const ocarina::Matrix<M, N> &rhs,
                                                 ocarina::index_sequence<i...>) noexcept {
    return ocarina::Matrix<M, M>{(lhs * rhs[i])...};
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator*(ocarina::Matrix<N, M> lhs, ocarina::Matrix<M, N> rhs) noexcept {
    return ocarina::multiply_matrices_impl(lhs, rhs, ocarina::make_index_sequence<M>());
}

namespace ocarina {
template<size_t N, size_t M, size_t... i>
__device__ constexpr auto add_matrices_impl(const ocarina::Matrix<N, M> &lhs, const ocarina::Matrix<N, M> &rhs,
                                            ocarina::index_sequence<i...>) noexcept {
    return ocarina::Matrix<N, M>{(lhs[i] + rhs[i])...};
}
}// namespace ocarina

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator+(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return ocarina::add_matrices_impl(lhs, rhs, ocarina::make_index_sequence<N>());
}

template<size_t N, size_t M>
[[nodiscard]] __device__ constexpr auto operator-(ocarina::Matrix<N, M> lhs, ocarina::Matrix<N, M> rhs) noexcept {
    return lhs + (-rhs);
}

#define OC_MAKE_MATRIX(N, M) using oc_float##N##x##M = ocarina::Matrix<N, M>;

OC_MAKE_MATRIX(2, 2)
OC_MAKE_MATRIX(2, 3)
OC_MAKE_MATRIX(2, 4)
OC_MAKE_MATRIX(3, 2)
OC_MAKE_MATRIX(3, 3)
OC_MAKE_MATRIX(3, 4)
OC_MAKE_MATRIX(4, 2)
OC_MAKE_MATRIX(4, 3)
OC_MAKE_MATRIX(4, 4)

#undef OC_MAKE_MATRIX
