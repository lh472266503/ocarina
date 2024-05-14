[[nodiscard]] constexpr auto xx_() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] constexpr auto xy_() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] constexpr auto xz_() const noexcept { return Vector<T, 2>{x, z}; }
[[nodiscard]] constexpr auto yx_() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] constexpr auto yy_() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] constexpr auto yz_() const noexcept { return Vector<T, 2>{y, z}; }
[[nodiscard]] constexpr auto zx_() const noexcept { return Vector<T, 2>{z, x}; }
[[nodiscard]] constexpr auto zy_() const noexcept { return Vector<T, 2>{z, y}; }
[[nodiscard]] constexpr auto zz_() const noexcept { return Vector<T, 2>{z, z}; }

[[nodiscard]] constexpr auto xxx_() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] constexpr auto xxy_() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] constexpr auto xxz_() const noexcept { return Vector<T, 3>{x, x, z}; }
[[nodiscard]] constexpr auto xyx_() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] constexpr auto xyy_() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] constexpr auto xyz_() const noexcept { return Vector<T, 3>{x, y, z}; }
[[nodiscard]] constexpr auto xzx_() const noexcept { return Vector<T, 3>{x, z, x}; }
[[nodiscard]] constexpr auto xzy_() const noexcept { return Vector<T, 3>{x, z, y}; }
[[nodiscard]] constexpr auto xzz_() const noexcept { return Vector<T, 3>{x, z, z}; }
[[nodiscard]] constexpr auto yxx_() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] constexpr auto yxy_() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] constexpr auto yxz_() const noexcept { return Vector<T, 3>{y, x, z}; }
[[nodiscard]] constexpr auto yyx_() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] constexpr auto yyy_() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] constexpr auto yyz_() const noexcept { return Vector<T, 3>{y, y, z}; }
[[nodiscard]] constexpr auto yzx_() const noexcept { return Vector<T, 3>{y, z, x}; }
[[nodiscard]] constexpr auto yzy_() const noexcept { return Vector<T, 3>{y, z, y}; }
[[nodiscard]] constexpr auto yzz_() const noexcept { return Vector<T, 3>{y, z, z}; }
[[nodiscard]] constexpr auto zxx_() const noexcept { return Vector<T, 3>{z, x, x}; }
[[nodiscard]] constexpr auto zxy_() const noexcept { return Vector<T, 3>{z, x, y}; }
[[nodiscard]] constexpr auto zxz_() const noexcept { return Vector<T, 3>{z, x, z}; }
[[nodiscard]] constexpr auto zyx_() const noexcept { return Vector<T, 3>{z, y, x}; }
[[nodiscard]] constexpr auto zyy_() const noexcept { return Vector<T, 3>{z, y, y}; }
[[nodiscard]] constexpr auto zyz_() const noexcept { return Vector<T, 3>{z, y, z}; }
[[nodiscard]] constexpr auto zzx_() const noexcept { return Vector<T, 3>{z, z, x}; }
[[nodiscard]] constexpr auto zzy_() const noexcept { return Vector<T, 3>{z, z, y}; }
[[nodiscard]] constexpr auto zzz_() const noexcept { return Vector<T, 3>{z, z, z}; }

[[nodiscard]] constexpr auto xxxx_() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] constexpr auto xxxy_() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] constexpr auto xxxz_() const noexcept { return Vector<T, 4>{x, x, x, z}; }
[[nodiscard]] constexpr auto xxyx_() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] constexpr auto xxyy_() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] constexpr auto xxyz_() const noexcept { return Vector<T, 4>{x, x, y, z}; }
[[nodiscard]] constexpr auto xxzx_() const noexcept { return Vector<T, 4>{x, x, z, x}; }
[[nodiscard]] constexpr auto xxzy_() const noexcept { return Vector<T, 4>{x, x, z, y}; }
[[nodiscard]] constexpr auto xxzz_() const noexcept { return Vector<T, 4>{x, x, z, z}; }
[[nodiscard]] constexpr auto xyxx_() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] constexpr auto xyxy_() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] constexpr auto xyxz_() const noexcept { return Vector<T, 4>{x, y, x, z}; }
[[nodiscard]] constexpr auto xyyx_() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] constexpr auto xyyy_() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] constexpr auto xyyz_() const noexcept { return Vector<T, 4>{x, y, y, z}; }
[[nodiscard]] constexpr auto xyzx_() const noexcept { return Vector<T, 4>{x, y, z, x}; }
[[nodiscard]] constexpr auto xyzy_() const noexcept { return Vector<T, 4>{x, y, z, y}; }
[[nodiscard]] constexpr auto xyzz_() const noexcept { return Vector<T, 4>{x, y, z, z}; }
[[nodiscard]] constexpr auto xzxx_() const noexcept { return Vector<T, 4>{x, z, x, x}; }
[[nodiscard]] constexpr auto xzxy_() const noexcept { return Vector<T, 4>{x, z, x, y}; }
[[nodiscard]] constexpr auto xzxz_() const noexcept { return Vector<T, 4>{x, z, x, z}; }
[[nodiscard]] constexpr auto xzyx_() const noexcept { return Vector<T, 4>{x, z, y, x}; }
[[nodiscard]] constexpr auto xzyy_() const noexcept { return Vector<T, 4>{x, z, y, y}; }
[[nodiscard]] constexpr auto xzyz_() const noexcept { return Vector<T, 4>{x, z, y, z}; }
[[nodiscard]] constexpr auto xzzx_() const noexcept { return Vector<T, 4>{x, z, z, x}; }
[[nodiscard]] constexpr auto xzzy_() const noexcept { return Vector<T, 4>{x, z, z, y}; }
[[nodiscard]] constexpr auto xzzz_() const noexcept { return Vector<T, 4>{x, z, z, z}; }
[[nodiscard]] constexpr auto yxxx_() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] constexpr auto yxxy_() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] constexpr auto yxxz_() const noexcept { return Vector<T, 4>{y, x, x, z}; }
[[nodiscard]] constexpr auto yxyx_() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] constexpr auto yxyy_() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] constexpr auto yxyz_() const noexcept { return Vector<T, 4>{y, x, y, z}; }
[[nodiscard]] constexpr auto yxzx_() const noexcept { return Vector<T, 4>{y, x, z, x}; }
[[nodiscard]] constexpr auto yxzy_() const noexcept { return Vector<T, 4>{y, x, z, y}; }
[[nodiscard]] constexpr auto yxzz_() const noexcept { return Vector<T, 4>{y, x, z, z}; }
[[nodiscard]] constexpr auto yyxx_() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] constexpr auto yyxy_() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] constexpr auto yyxz_() const noexcept { return Vector<T, 4>{y, y, x, z}; }
[[nodiscard]] constexpr auto yyyx_() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] constexpr auto yyyy_() const noexcept { return Vector<T, 4>{y, y, y, y}; }
[[nodiscard]] constexpr auto yyyz_() const noexcept { return Vector<T, 4>{y, y, y, z}; }
[[nodiscard]] constexpr auto yyzx_() const noexcept { return Vector<T, 4>{y, y, z, x}; }
[[nodiscard]] constexpr auto yyzy_() const noexcept { return Vector<T, 4>{y, y, z, y}; }
[[nodiscard]] constexpr auto yyzz_() const noexcept { return Vector<T, 4>{y, y, z, z}; }
[[nodiscard]] constexpr auto yzxx_() const noexcept { return Vector<T, 4>{y, z, x, x}; }
[[nodiscard]] constexpr auto yzxy_() const noexcept { return Vector<T, 4>{y, z, x, y}; }
[[nodiscard]] constexpr auto yzxz_() const noexcept { return Vector<T, 4>{y, z, x, z}; }
[[nodiscard]] constexpr auto yzyx_() const noexcept { return Vector<T, 4>{y, z, y, x}; }
[[nodiscard]] constexpr auto yzyy_() const noexcept { return Vector<T, 4>{y, z, y, y}; }
[[nodiscard]] constexpr auto yzyz_() const noexcept { return Vector<T, 4>{y, z, y, z}; }
[[nodiscard]] constexpr auto yzzx_() const noexcept { return Vector<T, 4>{y, z, z, x}; }
[[nodiscard]] constexpr auto yzzy_() const noexcept { return Vector<T, 4>{y, z, z, y}; }
[[nodiscard]] constexpr auto yzzz_() const noexcept { return Vector<T, 4>{y, z, z, z}; }
[[nodiscard]] constexpr auto zxxx_() const noexcept { return Vector<T, 4>{z, x, x, x}; }
[[nodiscard]] constexpr auto zxxy_() const noexcept { return Vector<T, 4>{z, x, x, y}; }
[[nodiscard]] constexpr auto zxxz_() const noexcept { return Vector<T, 4>{z, x, x, z}; }
[[nodiscard]] constexpr auto zxyx_() const noexcept { return Vector<T, 4>{z, x, y, x}; }
[[nodiscard]] constexpr auto zxyy_() const noexcept { return Vector<T, 4>{z, x, y, y}; }
[[nodiscard]] constexpr auto zxyz_() const noexcept { return Vector<T, 4>{z, x, y, z}; }
[[nodiscard]] constexpr auto zxzx_() const noexcept { return Vector<T, 4>{z, x, z, x}; }
[[nodiscard]] constexpr auto zxzy_() const noexcept { return Vector<T, 4>{z, x, z, y}; }
[[nodiscard]] constexpr auto zxzz_() const noexcept { return Vector<T, 4>{z, x, z, z}; }
[[nodiscard]] constexpr auto zyxx_() const noexcept { return Vector<T, 4>{z, y, x, x}; }
[[nodiscard]] constexpr auto zyxy_() const noexcept { return Vector<T, 4>{z, y, x, y}; }
[[nodiscard]] constexpr auto zyxz_() const noexcept { return Vector<T, 4>{z, y, x, z}; }
[[nodiscard]] constexpr auto zyyx_() const noexcept { return Vector<T, 4>{z, y, y, x}; }
[[nodiscard]] constexpr auto zyyy_() const noexcept { return Vector<T, 4>{z, y, y, y}; }
[[nodiscard]] constexpr auto zyyz_() const noexcept { return Vector<T, 4>{z, y, y, z}; }
[[nodiscard]] constexpr auto zyzx_() const noexcept { return Vector<T, 4>{z, y, z, x}; }
[[nodiscard]] constexpr auto zyzy_() const noexcept { return Vector<T, 4>{z, y, z, y}; }
[[nodiscard]] constexpr auto zyzz_() const noexcept { return Vector<T, 4>{z, y, z, z}; }
[[nodiscard]] constexpr auto zzxx_() const noexcept { return Vector<T, 4>{z, z, x, x}; }
[[nodiscard]] constexpr auto zzxy_() const noexcept { return Vector<T, 4>{z, z, x, y}; }
[[nodiscard]] constexpr auto zzxz_() const noexcept { return Vector<T, 4>{z, z, x, z}; }
[[nodiscard]] constexpr auto zzyx_() const noexcept { return Vector<T, 4>{z, z, y, x}; }
[[nodiscard]] constexpr auto zzyy_() const noexcept { return Vector<T, 4>{z, z, y, y}; }
[[nodiscard]] constexpr auto zzyz_() const noexcept { return Vector<T, 4>{z, z, y, z}; }
[[nodiscard]] constexpr auto zzzx_() const noexcept { return Vector<T, 4>{z, z, z, x}; }
[[nodiscard]] constexpr auto zzzy_() const noexcept { return Vector<T, 4>{z, z, z, y}; }
[[nodiscard]] constexpr auto zzzz_() const noexcept { return Vector<T, 4>{z, z, z, z}; }
