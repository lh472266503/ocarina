[[nodiscard]] constexpr auto xx_() const noexcept { return Vector<T, 2>{x, x}; }
[[nodiscard]] constexpr auto xy_() const noexcept { return Vector<T, 2>{x, y}; }
[[nodiscard]] constexpr auto xz_() const noexcept { return Vector<T, 2>{x, z}; }
[[nodiscard]] constexpr auto xw_() const noexcept { return Vector<T, 2>{x, w}; }
[[nodiscard]] constexpr auto yx_() const noexcept { return Vector<T, 2>{y, x}; }
[[nodiscard]] constexpr auto yy_() const noexcept { return Vector<T, 2>{y, y}; }
[[nodiscard]] constexpr auto yz_() const noexcept { return Vector<T, 2>{y, z}; }
[[nodiscard]] constexpr auto yw_() const noexcept { return Vector<T, 2>{y, w}; }
[[nodiscard]] constexpr auto zx_() const noexcept { return Vector<T, 2>{z, x}; }
[[nodiscard]] constexpr auto zy_() const noexcept { return Vector<T, 2>{z, y}; }
[[nodiscard]] constexpr auto zz_() const noexcept { return Vector<T, 2>{z, z}; }
[[nodiscard]] constexpr auto zw_() const noexcept { return Vector<T, 2>{z, w}; }
[[nodiscard]] constexpr auto wx_() const noexcept { return Vector<T, 2>{w, x}; }
[[nodiscard]] constexpr auto wy_() const noexcept { return Vector<T, 2>{w, y}; }
[[nodiscard]] constexpr auto wz_() const noexcept { return Vector<T, 2>{w, z}; }
[[nodiscard]] constexpr auto ww_() const noexcept { return Vector<T, 2>{w, w}; }

[[nodiscard]] constexpr auto xxx_() const noexcept { return Vector<T, 3>{x, x, x}; }
[[nodiscard]] constexpr auto xxy_() const noexcept { return Vector<T, 3>{x, x, y}; }
[[nodiscard]] constexpr auto xxz_() const noexcept { return Vector<T, 3>{x, x, z}; }
[[nodiscard]] constexpr auto xxw_() const noexcept { return Vector<T, 3>{x, x, w}; }
[[nodiscard]] constexpr auto xyx_() const noexcept { return Vector<T, 3>{x, y, x}; }
[[nodiscard]] constexpr auto xyy_() const noexcept { return Vector<T, 3>{x, y, y}; }
[[nodiscard]] constexpr auto xyz_() const noexcept { return Vector<T, 3>{x, y, z}; }
[[nodiscard]] constexpr auto xyw_() const noexcept { return Vector<T, 3>{x, y, w}; }
[[nodiscard]] constexpr auto xzx_() const noexcept { return Vector<T, 3>{x, z, x}; }
[[nodiscard]] constexpr auto xzy_() const noexcept { return Vector<T, 3>{x, z, y}; }
[[nodiscard]] constexpr auto xzz_() const noexcept { return Vector<T, 3>{x, z, z}; }
[[nodiscard]] constexpr auto xzw_() const noexcept { return Vector<T, 3>{x, z, w}; }
[[nodiscard]] constexpr auto xwx_() const noexcept { return Vector<T, 3>{x, w, x}; }
[[nodiscard]] constexpr auto xwy_() const noexcept { return Vector<T, 3>{x, w, y}; }
[[nodiscard]] constexpr auto xwz_() const noexcept { return Vector<T, 3>{x, w, z}; }
[[nodiscard]] constexpr auto xww_() const noexcept { return Vector<T, 3>{x, w, w}; }
[[nodiscard]] constexpr auto yxx_() const noexcept { return Vector<T, 3>{y, x, x}; }
[[nodiscard]] constexpr auto yxy_() const noexcept { return Vector<T, 3>{y, x, y}; }
[[nodiscard]] constexpr auto yxz_() const noexcept { return Vector<T, 3>{y, x, z}; }
[[nodiscard]] constexpr auto yxw_() const noexcept { return Vector<T, 3>{y, x, w}; }
[[nodiscard]] constexpr auto yyx_() const noexcept { return Vector<T, 3>{y, y, x}; }
[[nodiscard]] constexpr auto yyy_() const noexcept { return Vector<T, 3>{y, y, y}; }
[[nodiscard]] constexpr auto yyz_() const noexcept { return Vector<T, 3>{y, y, z}; }
[[nodiscard]] constexpr auto yyw_() const noexcept { return Vector<T, 3>{y, y, w}; }
[[nodiscard]] constexpr auto yzx_() const noexcept { return Vector<T, 3>{y, z, x}; }
[[nodiscard]] constexpr auto yzy_() const noexcept { return Vector<T, 3>{y, z, y}; }
[[nodiscard]] constexpr auto yzz_() const noexcept { return Vector<T, 3>{y, z, z}; }
[[nodiscard]] constexpr auto yzw_() const noexcept { return Vector<T, 3>{y, z, w}; }
[[nodiscard]] constexpr auto ywx_() const noexcept { return Vector<T, 3>{y, w, x}; }
[[nodiscard]] constexpr auto ywy_() const noexcept { return Vector<T, 3>{y, w, y}; }
[[nodiscard]] constexpr auto ywz_() const noexcept { return Vector<T, 3>{y, w, z}; }
[[nodiscard]] constexpr auto yww_() const noexcept { return Vector<T, 3>{y, w, w}; }
[[nodiscard]] constexpr auto zxx_() const noexcept { return Vector<T, 3>{z, x, x}; }
[[nodiscard]] constexpr auto zxy_() const noexcept { return Vector<T, 3>{z, x, y}; }
[[nodiscard]] constexpr auto zxz_() const noexcept { return Vector<T, 3>{z, x, z}; }
[[nodiscard]] constexpr auto zxw_() const noexcept { return Vector<T, 3>{z, x, w}; }
[[nodiscard]] constexpr auto zyx_() const noexcept { return Vector<T, 3>{z, y, x}; }
[[nodiscard]] constexpr auto zyy_() const noexcept { return Vector<T, 3>{z, y, y}; }
[[nodiscard]] constexpr auto zyz_() const noexcept { return Vector<T, 3>{z, y, z}; }
[[nodiscard]] constexpr auto zyw_() const noexcept { return Vector<T, 3>{z, y, w}; }
[[nodiscard]] constexpr auto zzx_() const noexcept { return Vector<T, 3>{z, z, x}; }
[[nodiscard]] constexpr auto zzy_() const noexcept { return Vector<T, 3>{z, z, y}; }
[[nodiscard]] constexpr auto zzz_() const noexcept { return Vector<T, 3>{z, z, z}; }
[[nodiscard]] constexpr auto zzw_() const noexcept { return Vector<T, 3>{z, z, w}; }
[[nodiscard]] constexpr auto zwx_() const noexcept { return Vector<T, 3>{z, w, x}; }
[[nodiscard]] constexpr auto zwy_() const noexcept { return Vector<T, 3>{z, w, y}; }
[[nodiscard]] constexpr auto zwz_() const noexcept { return Vector<T, 3>{z, w, z}; }
[[nodiscard]] constexpr auto zww_() const noexcept { return Vector<T, 3>{z, w, w}; }
[[nodiscard]] constexpr auto wxx_() const noexcept { return Vector<T, 3>{w, x, x}; }
[[nodiscard]] constexpr auto wxy_() const noexcept { return Vector<T, 3>{w, x, y}; }
[[nodiscard]] constexpr auto wxz_() const noexcept { return Vector<T, 3>{w, x, z}; }
[[nodiscard]] constexpr auto wxw_() const noexcept { return Vector<T, 3>{w, x, w}; }
[[nodiscard]] constexpr auto wyx_() const noexcept { return Vector<T, 3>{w, y, x}; }
[[nodiscard]] constexpr auto wyy_() const noexcept { return Vector<T, 3>{w, y, y}; }
[[nodiscard]] constexpr auto wyz_() const noexcept { return Vector<T, 3>{w, y, z}; }
[[nodiscard]] constexpr auto wyw_() const noexcept { return Vector<T, 3>{w, y, w}; }
[[nodiscard]] constexpr auto wzx_() const noexcept { return Vector<T, 3>{w, z, x}; }
[[nodiscard]] constexpr auto wzy_() const noexcept { return Vector<T, 3>{w, z, y}; }
[[nodiscard]] constexpr auto wzz_() const noexcept { return Vector<T, 3>{w, z, z}; }
[[nodiscard]] constexpr auto wzw_() const noexcept { return Vector<T, 3>{w, z, w}; }
[[nodiscard]] constexpr auto wwx_() const noexcept { return Vector<T, 3>{w, w, x}; }
[[nodiscard]] constexpr auto wwy_() const noexcept { return Vector<T, 3>{w, w, y}; }
[[nodiscard]] constexpr auto wwz_() const noexcept { return Vector<T, 3>{w, w, z}; }
[[nodiscard]] constexpr auto www_() const noexcept { return Vector<T, 3>{w, w, w}; }

[[nodiscard]] constexpr auto xxxx_() const noexcept { return Vector<T, 4>{x, x, x, x}; }
[[nodiscard]] constexpr auto xxxy_() const noexcept { return Vector<T, 4>{x, x, x, y}; }
[[nodiscard]] constexpr auto xxxz_() const noexcept { return Vector<T, 4>{x, x, x, z}; }
[[nodiscard]] constexpr auto xxxw_() const noexcept { return Vector<T, 4>{x, x, x, w}; }
[[nodiscard]] constexpr auto xxyx_() const noexcept { return Vector<T, 4>{x, x, y, x}; }
[[nodiscard]] constexpr auto xxyy_() const noexcept { return Vector<T, 4>{x, x, y, y}; }
[[nodiscard]] constexpr auto xxyz_() const noexcept { return Vector<T, 4>{x, x, y, z}; }
[[nodiscard]] constexpr auto xxyw_() const noexcept { return Vector<T, 4>{x, x, y, w}; }
[[nodiscard]] constexpr auto xxzx_() const noexcept { return Vector<T, 4>{x, x, z, x}; }
[[nodiscard]] constexpr auto xxzy_() const noexcept { return Vector<T, 4>{x, x, z, y}; }
[[nodiscard]] constexpr auto xxzz_() const noexcept { return Vector<T, 4>{x, x, z, z}; }
[[nodiscard]] constexpr auto xxzw_() const noexcept { return Vector<T, 4>{x, x, z, w}; }
[[nodiscard]] constexpr auto xxwx_() const noexcept { return Vector<T, 4>{x, x, w, x}; }
[[nodiscard]] constexpr auto xxwy_() const noexcept { return Vector<T, 4>{x, x, w, y}; }
[[nodiscard]] constexpr auto xxwz_() const noexcept { return Vector<T, 4>{x, x, w, z}; }
[[nodiscard]] constexpr auto xxww_() const noexcept { return Vector<T, 4>{x, x, w, w}; }
[[nodiscard]] constexpr auto xyxx_() const noexcept { return Vector<T, 4>{x, y, x, x}; }
[[nodiscard]] constexpr auto xyxy_() const noexcept { return Vector<T, 4>{x, y, x, y}; }
[[nodiscard]] constexpr auto xyxz_() const noexcept { return Vector<T, 4>{x, y, x, z}; }
[[nodiscard]] constexpr auto xyxw_() const noexcept { return Vector<T, 4>{x, y, x, w}; }
[[nodiscard]] constexpr auto xyyx_() const noexcept { return Vector<T, 4>{x, y, y, x}; }
[[nodiscard]] constexpr auto xyyy_() const noexcept { return Vector<T, 4>{x, y, y, y}; }
[[nodiscard]] constexpr auto xyyz_() const noexcept { return Vector<T, 4>{x, y, y, z}; }
[[nodiscard]] constexpr auto xyyw_() const noexcept { return Vector<T, 4>{x, y, y, w}; }
[[nodiscard]] constexpr auto xyzx_() const noexcept { return Vector<T, 4>{x, y, z, x}; }
[[nodiscard]] constexpr auto xyzy_() const noexcept { return Vector<T, 4>{x, y, z, y}; }
[[nodiscard]] constexpr auto xyzz_() const noexcept { return Vector<T, 4>{x, y, z, z}; }
[[nodiscard]] constexpr auto xyzw_() const noexcept { return Vector<T, 4>{x, y, z, w}; }
[[nodiscard]] constexpr auto xywx_() const noexcept { return Vector<T, 4>{x, y, w, x}; }
[[nodiscard]] constexpr auto xywy_() const noexcept { return Vector<T, 4>{x, y, w, y}; }
[[nodiscard]] constexpr auto xywz_() const noexcept { return Vector<T, 4>{x, y, w, z}; }
[[nodiscard]] constexpr auto xyww_() const noexcept { return Vector<T, 4>{x, y, w, w}; }
[[nodiscard]] constexpr auto xzxx_() const noexcept { return Vector<T, 4>{x, z, x, x}; }
[[nodiscard]] constexpr auto xzxy_() const noexcept { return Vector<T, 4>{x, z, x, y}; }
[[nodiscard]] constexpr auto xzxz_() const noexcept { return Vector<T, 4>{x, z, x, z}; }
[[nodiscard]] constexpr auto xzxw_() const noexcept { return Vector<T, 4>{x, z, x, w}; }
[[nodiscard]] constexpr auto xzyx_() const noexcept { return Vector<T, 4>{x, z, y, x}; }
[[nodiscard]] constexpr auto xzyy_() const noexcept { return Vector<T, 4>{x, z, y, y}; }
[[nodiscard]] constexpr auto xzyz_() const noexcept { return Vector<T, 4>{x, z, y, z}; }
[[nodiscard]] constexpr auto xzyw_() const noexcept { return Vector<T, 4>{x, z, y, w}; }
[[nodiscard]] constexpr auto xzzx_() const noexcept { return Vector<T, 4>{x, z, z, x}; }
[[nodiscard]] constexpr auto xzzy_() const noexcept { return Vector<T, 4>{x, z, z, y}; }
[[nodiscard]] constexpr auto xzzz_() const noexcept { return Vector<T, 4>{x, z, z, z}; }
[[nodiscard]] constexpr auto xzzw_() const noexcept { return Vector<T, 4>{x, z, z, w}; }
[[nodiscard]] constexpr auto xzwx_() const noexcept { return Vector<T, 4>{x, z, w, x}; }
[[nodiscard]] constexpr auto xzwy_() const noexcept { return Vector<T, 4>{x, z, w, y}; }
[[nodiscard]] constexpr auto xzwz_() const noexcept { return Vector<T, 4>{x, z, w, z}; }
[[nodiscard]] constexpr auto xzww_() const noexcept { return Vector<T, 4>{x, z, w, w}; }
[[nodiscard]] constexpr auto xwxx_() const noexcept { return Vector<T, 4>{x, w, x, x}; }
[[nodiscard]] constexpr auto xwxy_() const noexcept { return Vector<T, 4>{x, w, x, y}; }
[[nodiscard]] constexpr auto xwxz_() const noexcept { return Vector<T, 4>{x, w, x, z}; }
[[nodiscard]] constexpr auto xwxw_() const noexcept { return Vector<T, 4>{x, w, x, w}; }
[[nodiscard]] constexpr auto xwyx_() const noexcept { return Vector<T, 4>{x, w, y, x}; }
[[nodiscard]] constexpr auto xwyy_() const noexcept { return Vector<T, 4>{x, w, y, y}; }
[[nodiscard]] constexpr auto xwyz_() const noexcept { return Vector<T, 4>{x, w, y, z}; }
[[nodiscard]] constexpr auto xwyw_() const noexcept { return Vector<T, 4>{x, w, y, w}; }
[[nodiscard]] constexpr auto xwzx_() const noexcept { return Vector<T, 4>{x, w, z, x}; }
[[nodiscard]] constexpr auto xwzy_() const noexcept { return Vector<T, 4>{x, w, z, y}; }
[[nodiscard]] constexpr auto xwzz_() const noexcept { return Vector<T, 4>{x, w, z, z}; }
[[nodiscard]] constexpr auto xwzw_() const noexcept { return Vector<T, 4>{x, w, z, w}; }
[[nodiscard]] constexpr auto xwwx_() const noexcept { return Vector<T, 4>{x, w, w, x}; }
[[nodiscard]] constexpr auto xwwy_() const noexcept { return Vector<T, 4>{x, w, w, y}; }
[[nodiscard]] constexpr auto xwwz_() const noexcept { return Vector<T, 4>{x, w, w, z}; }
[[nodiscard]] constexpr auto xwww_() const noexcept { return Vector<T, 4>{x, w, w, w}; }
[[nodiscard]] constexpr auto yxxx_() const noexcept { return Vector<T, 4>{y, x, x, x}; }
[[nodiscard]] constexpr auto yxxy_() const noexcept { return Vector<T, 4>{y, x, x, y}; }
[[nodiscard]] constexpr auto yxxz_() const noexcept { return Vector<T, 4>{y, x, x, z}; }
[[nodiscard]] constexpr auto yxxw_() const noexcept { return Vector<T, 4>{y, x, x, w}; }
[[nodiscard]] constexpr auto yxyx_() const noexcept { return Vector<T, 4>{y, x, y, x}; }
[[nodiscard]] constexpr auto yxyy_() const noexcept { return Vector<T, 4>{y, x, y, y}; }
[[nodiscard]] constexpr auto yxyz_() const noexcept { return Vector<T, 4>{y, x, y, z}; }
[[nodiscard]] constexpr auto yxyw_() const noexcept { return Vector<T, 4>{y, x, y, w}; }
[[nodiscard]] constexpr auto yxzx_() const noexcept { return Vector<T, 4>{y, x, z, x}; }
[[nodiscard]] constexpr auto yxzy_() const noexcept { return Vector<T, 4>{y, x, z, y}; }
[[nodiscard]] constexpr auto yxzz_() const noexcept { return Vector<T, 4>{y, x, z, z}; }
[[nodiscard]] constexpr auto yxzw_() const noexcept { return Vector<T, 4>{y, x, z, w}; }
[[nodiscard]] constexpr auto yxwx_() const noexcept { return Vector<T, 4>{y, x, w, x}; }
[[nodiscard]] constexpr auto yxwy_() const noexcept { return Vector<T, 4>{y, x, w, y}; }
[[nodiscard]] constexpr auto yxwz_() const noexcept { return Vector<T, 4>{y, x, w, z}; }
[[nodiscard]] constexpr auto yxww_() const noexcept { return Vector<T, 4>{y, x, w, w}; }
[[nodiscard]] constexpr auto yyxx_() const noexcept { return Vector<T, 4>{y, y, x, x}; }
[[nodiscard]] constexpr auto yyxy_() const noexcept { return Vector<T, 4>{y, y, x, y}; }
[[nodiscard]] constexpr auto yyxz_() const noexcept { return Vector<T, 4>{y, y, x, z}; }
[[nodiscard]] constexpr auto yyxw_() const noexcept { return Vector<T, 4>{y, y, x, w}; }
[[nodiscard]] constexpr auto yyyx_() const noexcept { return Vector<T, 4>{y, y, y, x}; }
[[nodiscard]] constexpr auto yyyy_() const noexcept { return Vector<T, 4>{y, y, y, y}; }
[[nodiscard]] constexpr auto yyyz_() const noexcept { return Vector<T, 4>{y, y, y, z}; }
[[nodiscard]] constexpr auto yyyw_() const noexcept { return Vector<T, 4>{y, y, y, w}; }
[[nodiscard]] constexpr auto yyzx_() const noexcept { return Vector<T, 4>{y, y, z, x}; }
[[nodiscard]] constexpr auto yyzy_() const noexcept { return Vector<T, 4>{y, y, z, y}; }
[[nodiscard]] constexpr auto yyzz_() const noexcept { return Vector<T, 4>{y, y, z, z}; }
[[nodiscard]] constexpr auto yyzw_() const noexcept { return Vector<T, 4>{y, y, z, w}; }
[[nodiscard]] constexpr auto yywx_() const noexcept { return Vector<T, 4>{y, y, w, x}; }
[[nodiscard]] constexpr auto yywy_() const noexcept { return Vector<T, 4>{y, y, w, y}; }
[[nodiscard]] constexpr auto yywz_() const noexcept { return Vector<T, 4>{y, y, w, z}; }
[[nodiscard]] constexpr auto yyww_() const noexcept { return Vector<T, 4>{y, y, w, w}; }
[[nodiscard]] constexpr auto yzxx_() const noexcept { return Vector<T, 4>{y, z, x, x}; }
[[nodiscard]] constexpr auto yzxy_() const noexcept { return Vector<T, 4>{y, z, x, y}; }
[[nodiscard]] constexpr auto yzxz_() const noexcept { return Vector<T, 4>{y, z, x, z}; }
[[nodiscard]] constexpr auto yzxw_() const noexcept { return Vector<T, 4>{y, z, x, w}; }
[[nodiscard]] constexpr auto yzyx_() const noexcept { return Vector<T, 4>{y, z, y, x}; }
[[nodiscard]] constexpr auto yzyy_() const noexcept { return Vector<T, 4>{y, z, y, y}; }
[[nodiscard]] constexpr auto yzyz_() const noexcept { return Vector<T, 4>{y, z, y, z}; }
[[nodiscard]] constexpr auto yzyw_() const noexcept { return Vector<T, 4>{y, z, y, w}; }
[[nodiscard]] constexpr auto yzzx_() const noexcept { return Vector<T, 4>{y, z, z, x}; }
[[nodiscard]] constexpr auto yzzy_() const noexcept { return Vector<T, 4>{y, z, z, y}; }
[[nodiscard]] constexpr auto yzzz_() const noexcept { return Vector<T, 4>{y, z, z, z}; }
[[nodiscard]] constexpr auto yzzw_() const noexcept { return Vector<T, 4>{y, z, z, w}; }
[[nodiscard]] constexpr auto yzwx_() const noexcept { return Vector<T, 4>{y, z, w, x}; }
[[nodiscard]] constexpr auto yzwy_() const noexcept { return Vector<T, 4>{y, z, w, y}; }
[[nodiscard]] constexpr auto yzwz_() const noexcept { return Vector<T, 4>{y, z, w, z}; }
[[nodiscard]] constexpr auto yzww_() const noexcept { return Vector<T, 4>{y, z, w, w}; }
[[nodiscard]] constexpr auto ywxx_() const noexcept { return Vector<T, 4>{y, w, x, x}; }
[[nodiscard]] constexpr auto ywxy_() const noexcept { return Vector<T, 4>{y, w, x, y}; }
[[nodiscard]] constexpr auto ywxz_() const noexcept { return Vector<T, 4>{y, w, x, z}; }
[[nodiscard]] constexpr auto ywxw_() const noexcept { return Vector<T, 4>{y, w, x, w}; }
[[nodiscard]] constexpr auto ywyx_() const noexcept { return Vector<T, 4>{y, w, y, x}; }
[[nodiscard]] constexpr auto ywyy_() const noexcept { return Vector<T, 4>{y, w, y, y}; }
[[nodiscard]] constexpr auto ywyz_() const noexcept { return Vector<T, 4>{y, w, y, z}; }
[[nodiscard]] constexpr auto ywyw_() const noexcept { return Vector<T, 4>{y, w, y, w}; }
[[nodiscard]] constexpr auto ywzx_() const noexcept { return Vector<T, 4>{y, w, z, x}; }
[[nodiscard]] constexpr auto ywzy_() const noexcept { return Vector<T, 4>{y, w, z, y}; }
[[nodiscard]] constexpr auto ywzz_() const noexcept { return Vector<T, 4>{y, w, z, z}; }
[[nodiscard]] constexpr auto ywzw_() const noexcept { return Vector<T, 4>{y, w, z, w}; }
[[nodiscard]] constexpr auto ywwx_() const noexcept { return Vector<T, 4>{y, w, w, x}; }
[[nodiscard]] constexpr auto ywwy_() const noexcept { return Vector<T, 4>{y, w, w, y}; }
[[nodiscard]] constexpr auto ywwz_() const noexcept { return Vector<T, 4>{y, w, w, z}; }
[[nodiscard]] constexpr auto ywww_() const noexcept { return Vector<T, 4>{y, w, w, w}; }
[[nodiscard]] constexpr auto zxxx_() const noexcept { return Vector<T, 4>{z, x, x, x}; }
[[nodiscard]] constexpr auto zxxy_() const noexcept { return Vector<T, 4>{z, x, x, y}; }
[[nodiscard]] constexpr auto zxxz_() const noexcept { return Vector<T, 4>{z, x, x, z}; }
[[nodiscard]] constexpr auto zxxw_() const noexcept { return Vector<T, 4>{z, x, x, w}; }
[[nodiscard]] constexpr auto zxyx_() const noexcept { return Vector<T, 4>{z, x, y, x}; }
[[nodiscard]] constexpr auto zxyy_() const noexcept { return Vector<T, 4>{z, x, y, y}; }
[[nodiscard]] constexpr auto zxyz_() const noexcept { return Vector<T, 4>{z, x, y, z}; }
[[nodiscard]] constexpr auto zxyw_() const noexcept { return Vector<T, 4>{z, x, y, w}; }
[[nodiscard]] constexpr auto zxzx_() const noexcept { return Vector<T, 4>{z, x, z, x}; }
[[nodiscard]] constexpr auto zxzy_() const noexcept { return Vector<T, 4>{z, x, z, y}; }
[[nodiscard]] constexpr auto zxzz_() const noexcept { return Vector<T, 4>{z, x, z, z}; }
[[nodiscard]] constexpr auto zxzw_() const noexcept { return Vector<T, 4>{z, x, z, w}; }
[[nodiscard]] constexpr auto zxwx_() const noexcept { return Vector<T, 4>{z, x, w, x}; }
[[nodiscard]] constexpr auto zxwy_() const noexcept { return Vector<T, 4>{z, x, w, y}; }
[[nodiscard]] constexpr auto zxwz_() const noexcept { return Vector<T, 4>{z, x, w, z}; }
[[nodiscard]] constexpr auto zxww_() const noexcept { return Vector<T, 4>{z, x, w, w}; }
[[nodiscard]] constexpr auto zyxx_() const noexcept { return Vector<T, 4>{z, y, x, x}; }
[[nodiscard]] constexpr auto zyxy_() const noexcept { return Vector<T, 4>{z, y, x, y}; }
[[nodiscard]] constexpr auto zyxz_() const noexcept { return Vector<T, 4>{z, y, x, z}; }
[[nodiscard]] constexpr auto zyxw_() const noexcept { return Vector<T, 4>{z, y, x, w}; }
[[nodiscard]] constexpr auto zyyx_() const noexcept { return Vector<T, 4>{z, y, y, x}; }
[[nodiscard]] constexpr auto zyyy_() const noexcept { return Vector<T, 4>{z, y, y, y}; }
[[nodiscard]] constexpr auto zyyz_() const noexcept { return Vector<T, 4>{z, y, y, z}; }
[[nodiscard]] constexpr auto zyyw_() const noexcept { return Vector<T, 4>{z, y, y, w}; }
[[nodiscard]] constexpr auto zyzx_() const noexcept { return Vector<T, 4>{z, y, z, x}; }
[[nodiscard]] constexpr auto zyzy_() const noexcept { return Vector<T, 4>{z, y, z, y}; }
[[nodiscard]] constexpr auto zyzz_() const noexcept { return Vector<T, 4>{z, y, z, z}; }
[[nodiscard]] constexpr auto zyzw_() const noexcept { return Vector<T, 4>{z, y, z, w}; }
[[nodiscard]] constexpr auto zywx_() const noexcept { return Vector<T, 4>{z, y, w, x}; }
[[nodiscard]] constexpr auto zywy_() const noexcept { return Vector<T, 4>{z, y, w, y}; }
[[nodiscard]] constexpr auto zywz_() const noexcept { return Vector<T, 4>{z, y, w, z}; }
[[nodiscard]] constexpr auto zyww_() const noexcept { return Vector<T, 4>{z, y, w, w}; }
[[nodiscard]] constexpr auto zzxx_() const noexcept { return Vector<T, 4>{z, z, x, x}; }
[[nodiscard]] constexpr auto zzxy_() const noexcept { return Vector<T, 4>{z, z, x, y}; }
[[nodiscard]] constexpr auto zzxz_() const noexcept { return Vector<T, 4>{z, z, x, z}; }
[[nodiscard]] constexpr auto zzxw_() const noexcept { return Vector<T, 4>{z, z, x, w}; }
[[nodiscard]] constexpr auto zzyx_() const noexcept { return Vector<T, 4>{z, z, y, x}; }
[[nodiscard]] constexpr auto zzyy_() const noexcept { return Vector<T, 4>{z, z, y, y}; }
[[nodiscard]] constexpr auto zzyz_() const noexcept { return Vector<T, 4>{z, z, y, z}; }
[[nodiscard]] constexpr auto zzyw_() const noexcept { return Vector<T, 4>{z, z, y, w}; }
[[nodiscard]] constexpr auto zzzx_() const noexcept { return Vector<T, 4>{z, z, z, x}; }
[[nodiscard]] constexpr auto zzzy_() const noexcept { return Vector<T, 4>{z, z, z, y}; }
[[nodiscard]] constexpr auto zzzz_() const noexcept { return Vector<T, 4>{z, z, z, z}; }
[[nodiscard]] constexpr auto zzzw_() const noexcept { return Vector<T, 4>{z, z, z, w}; }
[[nodiscard]] constexpr auto zzwx_() const noexcept { return Vector<T, 4>{z, z, w, x}; }
[[nodiscard]] constexpr auto zzwy_() const noexcept { return Vector<T, 4>{z, z, w, y}; }
[[nodiscard]] constexpr auto zzwz_() const noexcept { return Vector<T, 4>{z, z, w, z}; }
[[nodiscard]] constexpr auto zzww_() const noexcept { return Vector<T, 4>{z, z, w, w}; }
[[nodiscard]] constexpr auto zwxx_() const noexcept { return Vector<T, 4>{z, w, x, x}; }
[[nodiscard]] constexpr auto zwxy_() const noexcept { return Vector<T, 4>{z, w, x, y}; }
[[nodiscard]] constexpr auto zwxz_() const noexcept { return Vector<T, 4>{z, w, x, z}; }
[[nodiscard]] constexpr auto zwxw_() const noexcept { return Vector<T, 4>{z, w, x, w}; }
[[nodiscard]] constexpr auto zwyx_() const noexcept { return Vector<T, 4>{z, w, y, x}; }
[[nodiscard]] constexpr auto zwyy_() const noexcept { return Vector<T, 4>{z, w, y, y}; }
[[nodiscard]] constexpr auto zwyz_() const noexcept { return Vector<T, 4>{z, w, y, z}; }
[[nodiscard]] constexpr auto zwyw_() const noexcept { return Vector<T, 4>{z, w, y, w}; }
[[nodiscard]] constexpr auto zwzx_() const noexcept { return Vector<T, 4>{z, w, z, x}; }
[[nodiscard]] constexpr auto zwzy_() const noexcept { return Vector<T, 4>{z, w, z, y}; }
[[nodiscard]] constexpr auto zwzz_() const noexcept { return Vector<T, 4>{z, w, z, z}; }
[[nodiscard]] constexpr auto zwzw_() const noexcept { return Vector<T, 4>{z, w, z, w}; }
[[nodiscard]] constexpr auto zwwx_() const noexcept { return Vector<T, 4>{z, w, w, x}; }
[[nodiscard]] constexpr auto zwwy_() const noexcept { return Vector<T, 4>{z, w, w, y}; }
[[nodiscard]] constexpr auto zwwz_() const noexcept { return Vector<T, 4>{z, w, w, z}; }
[[nodiscard]] constexpr auto zwww_() const noexcept { return Vector<T, 4>{z, w, w, w}; }
[[nodiscard]] constexpr auto wxxx_() const noexcept { return Vector<T, 4>{w, x, x, x}; }
[[nodiscard]] constexpr auto wxxy_() const noexcept { return Vector<T, 4>{w, x, x, y}; }
[[nodiscard]] constexpr auto wxxz_() const noexcept { return Vector<T, 4>{w, x, x, z}; }
[[nodiscard]] constexpr auto wxxw_() const noexcept { return Vector<T, 4>{w, x, x, w}; }
[[nodiscard]] constexpr auto wxyx_() const noexcept { return Vector<T, 4>{w, x, y, x}; }
[[nodiscard]] constexpr auto wxyy_() const noexcept { return Vector<T, 4>{w, x, y, y}; }
[[nodiscard]] constexpr auto wxyz_() const noexcept { return Vector<T, 4>{w, x, y, z}; }
[[nodiscard]] constexpr auto wxyw_() const noexcept { return Vector<T, 4>{w, x, y, w}; }
[[nodiscard]] constexpr auto wxzx_() const noexcept { return Vector<T, 4>{w, x, z, x}; }
[[nodiscard]] constexpr auto wxzy_() const noexcept { return Vector<T, 4>{w, x, z, y}; }
[[nodiscard]] constexpr auto wxzz_() const noexcept { return Vector<T, 4>{w, x, z, z}; }
[[nodiscard]] constexpr auto wxzw_() const noexcept { return Vector<T, 4>{w, x, z, w}; }
[[nodiscard]] constexpr auto wxwx_() const noexcept { return Vector<T, 4>{w, x, w, x}; }
[[nodiscard]] constexpr auto wxwy_() const noexcept { return Vector<T, 4>{w, x, w, y}; }
[[nodiscard]] constexpr auto wxwz_() const noexcept { return Vector<T, 4>{w, x, w, z}; }
[[nodiscard]] constexpr auto wxww_() const noexcept { return Vector<T, 4>{w, x, w, w}; }
[[nodiscard]] constexpr auto wyxx_() const noexcept { return Vector<T, 4>{w, y, x, x}; }
[[nodiscard]] constexpr auto wyxy_() const noexcept { return Vector<T, 4>{w, y, x, y}; }
[[nodiscard]] constexpr auto wyxz_() const noexcept { return Vector<T, 4>{w, y, x, z}; }
[[nodiscard]] constexpr auto wyxw_() const noexcept { return Vector<T, 4>{w, y, x, w}; }
[[nodiscard]] constexpr auto wyyx_() const noexcept { return Vector<T, 4>{w, y, y, x}; }
[[nodiscard]] constexpr auto wyyy_() const noexcept { return Vector<T, 4>{w, y, y, y}; }
[[nodiscard]] constexpr auto wyyz_() const noexcept { return Vector<T, 4>{w, y, y, z}; }
[[nodiscard]] constexpr auto wyyw_() const noexcept { return Vector<T, 4>{w, y, y, w}; }
[[nodiscard]] constexpr auto wyzx_() const noexcept { return Vector<T, 4>{w, y, z, x}; }
[[nodiscard]] constexpr auto wyzy_() const noexcept { return Vector<T, 4>{w, y, z, y}; }
[[nodiscard]] constexpr auto wyzz_() const noexcept { return Vector<T, 4>{w, y, z, z}; }
[[nodiscard]] constexpr auto wyzw_() const noexcept { return Vector<T, 4>{w, y, z, w}; }
[[nodiscard]] constexpr auto wywx_() const noexcept { return Vector<T, 4>{w, y, w, x}; }
[[nodiscard]] constexpr auto wywy_() const noexcept { return Vector<T, 4>{w, y, w, y}; }
[[nodiscard]] constexpr auto wywz_() const noexcept { return Vector<T, 4>{w, y, w, z}; }
[[nodiscard]] constexpr auto wyww_() const noexcept { return Vector<T, 4>{w, y, w, w}; }
[[nodiscard]] constexpr auto wzxx_() const noexcept { return Vector<T, 4>{w, z, x, x}; }
[[nodiscard]] constexpr auto wzxy_() const noexcept { return Vector<T, 4>{w, z, x, y}; }
[[nodiscard]] constexpr auto wzxz_() const noexcept { return Vector<T, 4>{w, z, x, z}; }
[[nodiscard]] constexpr auto wzxw_() const noexcept { return Vector<T, 4>{w, z, x, w}; }
[[nodiscard]] constexpr auto wzyx_() const noexcept { return Vector<T, 4>{w, z, y, x}; }
[[nodiscard]] constexpr auto wzyy_() const noexcept { return Vector<T, 4>{w, z, y, y}; }
[[nodiscard]] constexpr auto wzyz_() const noexcept { return Vector<T, 4>{w, z, y, z}; }
[[nodiscard]] constexpr auto wzyw_() const noexcept { return Vector<T, 4>{w, z, y, w}; }
[[nodiscard]] constexpr auto wzzx_() const noexcept { return Vector<T, 4>{w, z, z, x}; }
[[nodiscard]] constexpr auto wzzy_() const noexcept { return Vector<T, 4>{w, z, z, y}; }
[[nodiscard]] constexpr auto wzzz_() const noexcept { return Vector<T, 4>{w, z, z, z}; }
[[nodiscard]] constexpr auto wzzw_() const noexcept { return Vector<T, 4>{w, z, z, w}; }
[[nodiscard]] constexpr auto wzwx_() const noexcept { return Vector<T, 4>{w, z, w, x}; }
[[nodiscard]] constexpr auto wzwy_() const noexcept { return Vector<T, 4>{w, z, w, y}; }
[[nodiscard]] constexpr auto wzwz_() const noexcept { return Vector<T, 4>{w, z, w, z}; }
[[nodiscard]] constexpr auto wzww_() const noexcept { return Vector<T, 4>{w, z, w, w}; }
[[nodiscard]] constexpr auto wwxx_() const noexcept { return Vector<T, 4>{w, w, x, x}; }
[[nodiscard]] constexpr auto wwxy_() const noexcept { return Vector<T, 4>{w, w, x, y}; }
[[nodiscard]] constexpr auto wwxz_() const noexcept { return Vector<T, 4>{w, w, x, z}; }
[[nodiscard]] constexpr auto wwxw_() const noexcept { return Vector<T, 4>{w, w, x, w}; }
[[nodiscard]] constexpr auto wwyx_() const noexcept { return Vector<T, 4>{w, w, y, x}; }
[[nodiscard]] constexpr auto wwyy_() const noexcept { return Vector<T, 4>{w, w, y, y}; }
[[nodiscard]] constexpr auto wwyz_() const noexcept { return Vector<T, 4>{w, w, y, z}; }
[[nodiscard]] constexpr auto wwyw_() const noexcept { return Vector<T, 4>{w, w, y, w}; }
[[nodiscard]] constexpr auto wwzx_() const noexcept { return Vector<T, 4>{w, w, z, x}; }
[[nodiscard]] constexpr auto wwzy_() const noexcept { return Vector<T, 4>{w, w, z, y}; }
[[nodiscard]] constexpr auto wwzz_() const noexcept { return Vector<T, 4>{w, w, z, z}; }
[[nodiscard]] constexpr auto wwzw_() const noexcept { return Vector<T, 4>{w, w, z, w}; }
[[nodiscard]] constexpr auto wwwx_() const noexcept { return Vector<T, 4>{w, w, w, x}; }
[[nodiscard]] constexpr auto wwwy_() const noexcept { return Vector<T, 4>{w, w, w, y}; }
[[nodiscard]] constexpr auto wwwz_() const noexcept { return Vector<T, 4>{w, w, w, z}; }
[[nodiscard]] constexpr auto wwww_() const noexcept { return Vector<T, 4>{w, w, w, w}; }
