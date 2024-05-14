[[nodiscard]] constexpr auto xx_() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] constexpr auto xy_() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] constexpr auto yx_() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] constexpr auto yy_() const noexcept { return Vector<T, 2>{y, y}; }

[[nodiscard]] constexpr auto xxx_() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] constexpr auto xxy_() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] constexpr auto xyx_() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] constexpr auto xyy_() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] constexpr auto yxx_() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] constexpr auto yxy_() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] constexpr auto yyx_() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] constexpr auto yyy_() const noexcept { return Vector<T, 3>{y, y, y}; }

[[nodiscard]] constexpr auto xxxx_() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] constexpr auto xxxy_() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] constexpr auto xxyx_() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] constexpr auto xxyy_() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] constexpr auto xyxx_() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] constexpr auto xyxy_() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] constexpr auto xyyx_() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] constexpr auto xyyy_() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] constexpr auto yxxx_() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] constexpr auto yxxy_() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] constexpr auto yxyx_() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] constexpr auto yxyy_() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] constexpr auto yyxx_() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] constexpr auto yyxy_() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] constexpr auto yyyx_() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] constexpr auto yyyy_() const noexcept { return Vector<T, 4>{y, y, y, y}; }
