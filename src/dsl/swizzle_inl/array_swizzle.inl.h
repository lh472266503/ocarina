
[[nodiscard]] auto xx() const { return Var<Vector<T, 2>>(at(0), at(0)) }
[[nodiscard]] auto xy() const { return Var<Vector<T, 2>>(at(0), at(1)) }
[[nodiscard]] auto xz() const { return Var<Vector<T, 2>>(at(0), at(2)) }
[[nodiscard]] auto xw() const { return Var<Vector<T, 2>>(at(0), at(3)) }
[[nodiscard]] auto yx() const { return Var<Vector<T, 2>>(at(1), at(0)) }
[[nodiscard]] auto yy() const { return Var<Vector<T, 2>>(at(1), at(1)) }
[[nodiscard]] auto yz() const { return Var<Vector<T, 2>>(at(1), at(2)) }
[[nodiscard]] auto yw() const { return Var<Vector<T, 2>>(at(1), at(3)) }
[[nodiscard]] auto zx() const { return Var<Vector<T, 2>>(at(2), at(0)) }
[[nodiscard]] auto zy() const { return Var<Vector<T, 2>>(at(2), at(1)) }
[[nodiscard]] auto zz() const { return Var<Vector<T, 2>>(at(2), at(2)) }
[[nodiscard]] auto zw() const { return Var<Vector<T, 2>>(at(2), at(3)) }
[[nodiscard]] auto wx() const { return Var<Vector<T, 2>>(at(3), at(0)) }
[[nodiscard]] auto wy() const { return Var<Vector<T, 2>>(at(3), at(1)) }
[[nodiscard]] auto wz() const { return Var<Vector<T, 2>>(at(3), at(2)) }
[[nodiscard]] auto ww() const { return Var<Vector<T, 2>>(at(3), at(3)) }

[[nodiscard]] auto xxx() const { return Var<Vector<T, 3>>(at(0), at(0), at(0)); }
[[nodiscard]] auto xxy() const { return Var<Vector<T, 3>>(at(0), at(0), at(1)); }
[[nodiscard]] auto xxz() const { return Var<Vector<T, 3>>(at(0), at(0), at(2)); }
[[nodiscard]] auto xxw() const { return Var<Vector<T, 3>>(at(0), at(0), at(3)); }
[[nodiscard]] auto xyx() const { return Var<Vector<T, 3>>(at(0), at(1), at(0)); }
[[nodiscard]] auto xyy() const { return Var<Vector<T, 3>>(at(0), at(1), at(1)); }
[[nodiscard]] auto xyz() const { return Var<Vector<T, 3>>(at(0), at(1), at(2)); }
[[nodiscard]] auto xyw() const { return Var<Vector<T, 3>>(at(0), at(1), at(3)); }
[[nodiscard]] auto xzx() const { return Var<Vector<T, 3>>(at(0), at(2), at(0)); }
[[nodiscard]] auto xzy() const { return Var<Vector<T, 3>>(at(0), at(2), at(1)); }
[[nodiscard]] auto xzz() const { return Var<Vector<T, 3>>(at(0), at(2), at(2)); }
[[nodiscard]] auto xzw() const { return Var<Vector<T, 3>>(at(0), at(2), at(3)); }
[[nodiscard]] auto xwx() const { return Var<Vector<T, 3>>(at(0), at(3), at(0)); }
[[nodiscard]] auto xwy() const { return Var<Vector<T, 3>>(at(0), at(3), at(1)); }
[[nodiscard]] auto xwz() const { return Var<Vector<T, 3>>(at(0), at(3), at(2)); }
[[nodiscard]] auto xww() const { return Var<Vector<T, 3>>(at(0), at(3), at(3)); }
[[nodiscard]] auto yxx() const { return Var<Vector<T, 3>>(at(1), at(0), at(0)); }
[[nodiscard]] auto yxy() const { return Var<Vector<T, 3>>(at(1), at(0), at(1)); }
[[nodiscard]] auto yxz() const { return Var<Vector<T, 3>>(at(1), at(0), at(2)); }
[[nodiscard]] auto yxw() const { return Var<Vector<T, 3>>(at(1), at(0), at(3)); }
[[nodiscard]] auto yyx() const { return Var<Vector<T, 3>>(at(1), at(1), at(0)); }
[[nodiscard]] auto yyy() const { return Var<Vector<T, 3>>(at(1), at(1), at(1)); }
[[nodiscard]] auto yyz() const { return Var<Vector<T, 3>>(at(1), at(1), at(2)); }
[[nodiscard]] auto yyw() const { return Var<Vector<T, 3>>(at(1), at(1), at(3)); }
[[nodiscard]] auto yzx() const { return Var<Vector<T, 3>>(at(1), at(2), at(0)); }
[[nodiscard]] auto yzy() const { return Var<Vector<T, 3>>(at(1), at(2), at(1)); }
[[nodiscard]] auto yzz() const { return Var<Vector<T, 3>>(at(1), at(2), at(2)); }
[[nodiscard]] auto yzw() const { return Var<Vector<T, 3>>(at(1), at(2), at(3)); }
[[nodiscard]] auto ywx() const { return Var<Vector<T, 3>>(at(1), at(3), at(0)); }
[[nodiscard]] auto ywy() const { return Var<Vector<T, 3>>(at(1), at(3), at(1)); }
[[nodiscard]] auto ywz() const { return Var<Vector<T, 3>>(at(1), at(3), at(2)); }
[[nodiscard]] auto yww() const { return Var<Vector<T, 3>>(at(1), at(3), at(3)); }
[[nodiscard]] auto zxx() const { return Var<Vector<T, 3>>(at(2), at(0), at(0)); }
[[nodiscard]] auto zxy() const { return Var<Vector<T, 3>>(at(2), at(0), at(1)); }
[[nodiscard]] auto zxz() const { return Var<Vector<T, 3>>(at(2), at(0), at(2)); }
[[nodiscard]] auto zxw() const { return Var<Vector<T, 3>>(at(2), at(0), at(3)); }
[[nodiscard]] auto zyx() const { return Var<Vector<T, 3>>(at(2), at(1), at(0)); }
[[nodiscard]] auto zyy() const { return Var<Vector<T, 3>>(at(2), at(1), at(1)); }
[[nodiscard]] auto zyz() const { return Var<Vector<T, 3>>(at(2), at(1), at(2)); }
[[nodiscard]] auto zyw() const { return Var<Vector<T, 3>>(at(2), at(1), at(3)); }
[[nodiscard]] auto zzx() const { return Var<Vector<T, 3>>(at(2), at(2), at(0)); }
[[nodiscard]] auto zzy() const { return Var<Vector<T, 3>>(at(2), at(2), at(1)); }
[[nodiscard]] auto zzz() const { return Var<Vector<T, 3>>(at(2), at(2), at(2)); }
[[nodiscard]] auto zzw() const { return Var<Vector<T, 3>>(at(2), at(2), at(3)); }
[[nodiscard]] auto zwx() const { return Var<Vector<T, 3>>(at(2), at(3), at(0)); }
[[nodiscard]] auto zwy() const { return Var<Vector<T, 3>>(at(2), at(3), at(1)); }
[[nodiscard]] auto zwz() const { return Var<Vector<T, 3>>(at(2), at(3), at(2)); }
[[nodiscard]] auto zww() const { return Var<Vector<T, 3>>(at(2), at(3), at(3)); }
[[nodiscard]] auto wxx() const { return Var<Vector<T, 3>>(at(3), at(0), at(0)); }
[[nodiscard]] auto wxy() const { return Var<Vector<T, 3>>(at(3), at(0), at(1)); }
[[nodiscard]] auto wxz() const { return Var<Vector<T, 3>>(at(3), at(0), at(2)); }
[[nodiscard]] auto wxw() const { return Var<Vector<T, 3>>(at(3), at(0), at(3)); }
[[nodiscard]] auto wyx() const { return Var<Vector<T, 3>>(at(3), at(1), at(0)); }
[[nodiscard]] auto wyy() const { return Var<Vector<T, 3>>(at(3), at(1), at(1)); }
[[nodiscard]] auto wyz() const { return Var<Vector<T, 3>>(at(3), at(1), at(2)); }
[[nodiscard]] auto wyw() const { return Var<Vector<T, 3>>(at(3), at(1), at(3)); }
[[nodiscard]] auto wzx() const { return Var<Vector<T, 3>>(at(3), at(2), at(0)); }
[[nodiscard]] auto wzy() const { return Var<Vector<T, 3>>(at(3), at(2), at(1)); }
[[nodiscard]] auto wzz() const { return Var<Vector<T, 3>>(at(3), at(2), at(2)); }
[[nodiscard]] auto wzw() const { return Var<Vector<T, 3>>(at(3), at(2), at(3)); }
[[nodiscard]] auto wwx() const { return Var<Vector<T, 3>>(at(3), at(3), at(0)); }
[[nodiscard]] auto wwy() const { return Var<Vector<T, 3>>(at(3), at(3), at(1)); }
[[nodiscard]] auto wwz() const { return Var<Vector<T, 3>>(at(3), at(3), at(2)); }
[[nodiscard]] auto www() const { return Var<Vector<T, 3>>(at(3), at(3), at(3)); }

[[nodiscard]] auto xxxx() const { return Var<Vector<T, 4>>(at(0), at(0), at(0), at(0)); }
[[nodiscard]] auto xxxy() const { return Var<Vector<T, 4>>(at(0), at(0), at(0), at(1)); }
[[nodiscard]] auto xxxz() const { return Var<Vector<T, 4>>(at(0), at(0), at(0), at(2)); }
[[nodiscard]] auto xxxw() const { return Var<Vector<T, 4>>(at(0), at(0), at(0), at(3)); }
[[nodiscard]] auto xxyx() const { return Var<Vector<T, 4>>(at(0), at(0), at(1), at(0)); }
[[nodiscard]] auto xxyy() const { return Var<Vector<T, 4>>(at(0), at(0), at(1), at(1)); }
[[nodiscard]] auto xxyz() const { return Var<Vector<T, 4>>(at(0), at(0), at(1), at(2)); }
[[nodiscard]] auto xxyw() const { return Var<Vector<T, 4>>(at(0), at(0), at(1), at(3)); }
[[nodiscard]] auto xxzx() const { return Var<Vector<T, 4>>(at(0), at(0), at(2), at(0)); }
[[nodiscard]] auto xxzy() const { return Var<Vector<T, 4>>(at(0), at(0), at(2), at(1)); }
[[nodiscard]] auto xxzz() const { return Var<Vector<T, 4>>(at(0), at(0), at(2), at(2)); }
[[nodiscard]] auto xxzw() const { return Var<Vector<T, 4>>(at(0), at(0), at(2), at(3)); }
[[nodiscard]] auto xxwx() const { return Var<Vector<T, 4>>(at(0), at(0), at(3), at(0)); }
[[nodiscard]] auto xxwy() const { return Var<Vector<T, 4>>(at(0), at(0), at(3), at(1)); }
[[nodiscard]] auto xxwz() const { return Var<Vector<T, 4>>(at(0), at(0), at(3), at(2)); }
[[nodiscard]] auto xxww() const { return Var<Vector<T, 4>>(at(0), at(0), at(3), at(3)); }
[[nodiscard]] auto xyxx() const { return Var<Vector<T, 4>>(at(0), at(1), at(0), at(0)); }
[[nodiscard]] auto xyxy() const { return Var<Vector<T, 4>>(at(0), at(1), at(0), at(1)); }
[[nodiscard]] auto xyxz() const { return Var<Vector<T, 4>>(at(0), at(1), at(0), at(2)); }
[[nodiscard]] auto xyxw() const { return Var<Vector<T, 4>>(at(0), at(1), at(0), at(3)); }
[[nodiscard]] auto xyyx() const { return Var<Vector<T, 4>>(at(0), at(1), at(1), at(0)); }
[[nodiscard]] auto xyyy() const { return Var<Vector<T, 4>>(at(0), at(1), at(1), at(1)); }
[[nodiscard]] auto xyyz() const { return Var<Vector<T, 4>>(at(0), at(1), at(1), at(2)); }
[[nodiscard]] auto xyyw() const { return Var<Vector<T, 4>>(at(0), at(1), at(1), at(3)); }
[[nodiscard]] auto xyzx() const { return Var<Vector<T, 4>>(at(0), at(1), at(2), at(0)); }
[[nodiscard]] auto xyzy() const { return Var<Vector<T, 4>>(at(0), at(1), at(2), at(1)); }
[[nodiscard]] auto xyzz() const { return Var<Vector<T, 4>>(at(0), at(1), at(2), at(2)); }
[[nodiscard]] auto xyzw() const { return Var<Vector<T, 4>>(at(0), at(1), at(2), at(3)); }
[[nodiscard]] auto xywx() const { return Var<Vector<T, 4>>(at(0), at(1), at(3), at(0)); }
[[nodiscard]] auto xywy() const { return Var<Vector<T, 4>>(at(0), at(1), at(3), at(1)); }
[[nodiscard]] auto xywz() const { return Var<Vector<T, 4>>(at(0), at(1), at(3), at(2)); }
[[nodiscard]] auto xyww() const { return Var<Vector<T, 4>>(at(0), at(1), at(3), at(3)); }
[[nodiscard]] auto xzxx() const { return Var<Vector<T, 4>>(at(0), at(2), at(0), at(0)); }
[[nodiscard]] auto xzxy() const { return Var<Vector<T, 4>>(at(0), at(2), at(0), at(1)); }
[[nodiscard]] auto xzxz() const { return Var<Vector<T, 4>>(at(0), at(2), at(0), at(2)); }
[[nodiscard]] auto xzxw() const { return Var<Vector<T, 4>>(at(0), at(2), at(0), at(3)); }
[[nodiscard]] auto xzyx() const { return Var<Vector<T, 4>>(at(0), at(2), at(1), at(0)); }
[[nodiscard]] auto xzyy() const { return Var<Vector<T, 4>>(at(0), at(2), at(1), at(1)); }
[[nodiscard]] auto xzyz() const { return Var<Vector<T, 4>>(at(0), at(2), at(1), at(2)); }
[[nodiscard]] auto xzyw() const { return Var<Vector<T, 4>>(at(0), at(2), at(1), at(3)); }
[[nodiscard]] auto xzzx() const { return Var<Vector<T, 4>>(at(0), at(2), at(2), at(0)); }
[[nodiscard]] auto xzzy() const { return Var<Vector<T, 4>>(at(0), at(2), at(2), at(1)); }
[[nodiscard]] auto xzzz() const { return Var<Vector<T, 4>>(at(0), at(2), at(2), at(2)); }
[[nodiscard]] auto xzzw() const { return Var<Vector<T, 4>>(at(0), at(2), at(2), at(3)); }
[[nodiscard]] auto xzwx() const { return Var<Vector<T, 4>>(at(0), at(2), at(3), at(0)); }
[[nodiscard]] auto xzwy() const { return Var<Vector<T, 4>>(at(0), at(2), at(3), at(1)); }
[[nodiscard]] auto xzwz() const { return Var<Vector<T, 4>>(at(0), at(2), at(3), at(2)); }
[[nodiscard]] auto xzww() const { return Var<Vector<T, 4>>(at(0), at(2), at(3), at(3)); }
[[nodiscard]] auto xwxx() const { return Var<Vector<T, 4>>(at(0), at(3), at(0), at(0)); }
[[nodiscard]] auto xwxy() const { return Var<Vector<T, 4>>(at(0), at(3), at(0), at(1)); }
[[nodiscard]] auto xwxz() const { return Var<Vector<T, 4>>(at(0), at(3), at(0), at(2)); }
[[nodiscard]] auto xwxw() const { return Var<Vector<T, 4>>(at(0), at(3), at(0), at(3)); }
[[nodiscard]] auto xwyx() const { return Var<Vector<T, 4>>(at(0), at(3), at(1), at(0)); }
[[nodiscard]] auto xwyy() const { return Var<Vector<T, 4>>(at(0), at(3), at(1), at(1)); }
[[nodiscard]] auto xwyz() const { return Var<Vector<T, 4>>(at(0), at(3), at(1), at(2)); }
[[nodiscard]] auto xwyw() const { return Var<Vector<T, 4>>(at(0), at(3), at(1), at(3)); }
[[nodiscard]] auto xwzx() const { return Var<Vector<T, 4>>(at(0), at(3), at(2), at(0)); }
[[nodiscard]] auto xwzy() const { return Var<Vector<T, 4>>(at(0), at(3), at(2), at(1)); }
[[nodiscard]] auto xwzz() const { return Var<Vector<T, 4>>(at(0), at(3), at(2), at(2)); }
[[nodiscard]] auto xwzw() const { return Var<Vector<T, 4>>(at(0), at(3), at(2), at(3)); }
[[nodiscard]] auto xwwx() const { return Var<Vector<T, 4>>(at(0), at(3), at(3), at(0)); }
[[nodiscard]] auto xwwy() const { return Var<Vector<T, 4>>(at(0), at(3), at(3), at(1)); }
[[nodiscard]] auto xwwz() const { return Var<Vector<T, 4>>(at(0), at(3), at(3), at(2)); }
[[nodiscard]] auto xwww() const { return Var<Vector<T, 4>>(at(0), at(3), at(3), at(3)); }
[[nodiscard]] auto yxxx() const { return Var<Vector<T, 4>>(at(1), at(0), at(0), at(0)); }
[[nodiscard]] auto yxxy() const { return Var<Vector<T, 4>>(at(1), at(0), at(0), at(1)); }
[[nodiscard]] auto yxxz() const { return Var<Vector<T, 4>>(at(1), at(0), at(0), at(2)); }
[[nodiscard]] auto yxxw() const { return Var<Vector<T, 4>>(at(1), at(0), at(0), at(3)); }
[[nodiscard]] auto yxyx() const { return Var<Vector<T, 4>>(at(1), at(0), at(1), at(0)); }
[[nodiscard]] auto yxyy() const { return Var<Vector<T, 4>>(at(1), at(0), at(1), at(1)); }
[[nodiscard]] auto yxyz() const { return Var<Vector<T, 4>>(at(1), at(0), at(1), at(2)); }
[[nodiscard]] auto yxyw() const { return Var<Vector<T, 4>>(at(1), at(0), at(1), at(3)); }
[[nodiscard]] auto yxzx() const { return Var<Vector<T, 4>>(at(1), at(0), at(2), at(0)); }
[[nodiscard]] auto yxzy() const { return Var<Vector<T, 4>>(at(1), at(0), at(2), at(1)); }
[[nodiscard]] auto yxzz() const { return Var<Vector<T, 4>>(at(1), at(0), at(2), at(2)); }
[[nodiscard]] auto yxzw() const { return Var<Vector<T, 4>>(at(1), at(0), at(2), at(3)); }
[[nodiscard]] auto yxwx() const { return Var<Vector<T, 4>>(at(1), at(0), at(3), at(0)); }
[[nodiscard]] auto yxwy() const { return Var<Vector<T, 4>>(at(1), at(0), at(3), at(1)); }
[[nodiscard]] auto yxwz() const { return Var<Vector<T, 4>>(at(1), at(0), at(3), at(2)); }
[[nodiscard]] auto yxww() const { return Var<Vector<T, 4>>(at(1), at(0), at(3), at(3)); }
[[nodiscard]] auto yyxx() const { return Var<Vector<T, 4>>(at(1), at(1), at(0), at(0)); }
[[nodiscard]] auto yyxy() const { return Var<Vector<T, 4>>(at(1), at(1), at(0), at(1)); }
[[nodiscard]] auto yyxz() const { return Var<Vector<T, 4>>(at(1), at(1), at(0), at(2)); }
[[nodiscard]] auto yyxw() const { return Var<Vector<T, 4>>(at(1), at(1), at(0), at(3)); }
[[nodiscard]] auto yyyx() const { return Var<Vector<T, 4>>(at(1), at(1), at(1), at(0)); }
[[nodiscard]] auto yyyy() const { return Var<Vector<T, 4>>(at(1), at(1), at(1), at(1)); }
[[nodiscard]] auto yyyz() const { return Var<Vector<T, 4>>(at(1), at(1), at(1), at(2)); }
[[nodiscard]] auto yyyw() const { return Var<Vector<T, 4>>(at(1), at(1), at(1), at(3)); }
[[nodiscard]] auto yyzx() const { return Var<Vector<T, 4>>(at(1), at(1), at(2), at(0)); }
[[nodiscard]] auto yyzy() const { return Var<Vector<T, 4>>(at(1), at(1), at(2), at(1)); }
[[nodiscard]] auto yyzz() const { return Var<Vector<T, 4>>(at(1), at(1), at(2), at(2)); }
[[nodiscard]] auto yyzw() const { return Var<Vector<T, 4>>(at(1), at(1), at(2), at(3)); }
[[nodiscard]] auto yywx() const { return Var<Vector<T, 4>>(at(1), at(1), at(3), at(0)); }
[[nodiscard]] auto yywy() const { return Var<Vector<T, 4>>(at(1), at(1), at(3), at(1)); }
[[nodiscard]] auto yywz() const { return Var<Vector<T, 4>>(at(1), at(1), at(3), at(2)); }
[[nodiscard]] auto yyww() const { return Var<Vector<T, 4>>(at(1), at(1), at(3), at(3)); }
[[nodiscard]] auto yzxx() const { return Var<Vector<T, 4>>(at(1), at(2), at(0), at(0)); }
[[nodiscard]] auto yzxy() const { return Var<Vector<T, 4>>(at(1), at(2), at(0), at(1)); }
[[nodiscard]] auto yzxz() const { return Var<Vector<T, 4>>(at(1), at(2), at(0), at(2)); }
[[nodiscard]] auto yzxw() const { return Var<Vector<T, 4>>(at(1), at(2), at(0), at(3)); }
[[nodiscard]] auto yzyx() const { return Var<Vector<T, 4>>(at(1), at(2), at(1), at(0)); }
[[nodiscard]] auto yzyy() const { return Var<Vector<T, 4>>(at(1), at(2), at(1), at(1)); }
[[nodiscard]] auto yzyz() const { return Var<Vector<T, 4>>(at(1), at(2), at(1), at(2)); }
[[nodiscard]] auto yzyw() const { return Var<Vector<T, 4>>(at(1), at(2), at(1), at(3)); }
[[nodiscard]] auto yzzx() const { return Var<Vector<T, 4>>(at(1), at(2), at(2), at(0)); }
[[nodiscard]] auto yzzy() const { return Var<Vector<T, 4>>(at(1), at(2), at(2), at(1)); }
[[nodiscard]] auto yzzz() const { return Var<Vector<T, 4>>(at(1), at(2), at(2), at(2)); }
[[nodiscard]] auto yzzw() const { return Var<Vector<T, 4>>(at(1), at(2), at(2), at(3)); }
[[nodiscard]] auto yzwx() const { return Var<Vector<T, 4>>(at(1), at(2), at(3), at(0)); }
[[nodiscard]] auto yzwy() const { return Var<Vector<T, 4>>(at(1), at(2), at(3), at(1)); }
[[nodiscard]] auto yzwz() const { return Var<Vector<T, 4>>(at(1), at(2), at(3), at(2)); }
[[nodiscard]] auto yzww() const { return Var<Vector<T, 4>>(at(1), at(2), at(3), at(3)); }
[[nodiscard]] auto ywxx() const { return Var<Vector<T, 4>>(at(1), at(3), at(0), at(0)); }
[[nodiscard]] auto ywxy() const { return Var<Vector<T, 4>>(at(1), at(3), at(0), at(1)); }
[[nodiscard]] auto ywxz() const { return Var<Vector<T, 4>>(at(1), at(3), at(0), at(2)); }
[[nodiscard]] auto ywxw() const { return Var<Vector<T, 4>>(at(1), at(3), at(0), at(3)); }
[[nodiscard]] auto ywyx() const { return Var<Vector<T, 4>>(at(1), at(3), at(1), at(0)); }
[[nodiscard]] auto ywyy() const { return Var<Vector<T, 4>>(at(1), at(3), at(1), at(1)); }
[[nodiscard]] auto ywyz() const { return Var<Vector<T, 4>>(at(1), at(3), at(1), at(2)); }
[[nodiscard]] auto ywyw() const { return Var<Vector<T, 4>>(at(1), at(3), at(1), at(3)); }
[[nodiscard]] auto ywzx() const { return Var<Vector<T, 4>>(at(1), at(3), at(2), at(0)); }
[[nodiscard]] auto ywzy() const { return Var<Vector<T, 4>>(at(1), at(3), at(2), at(1)); }
[[nodiscard]] auto ywzz() const { return Var<Vector<T, 4>>(at(1), at(3), at(2), at(2)); }
[[nodiscard]] auto ywzw() const { return Var<Vector<T, 4>>(at(1), at(3), at(2), at(3)); }
[[nodiscard]] auto ywwx() const { return Var<Vector<T, 4>>(at(1), at(3), at(3), at(0)); }
[[nodiscard]] auto ywwy() const { return Var<Vector<T, 4>>(at(1), at(3), at(3), at(1)); }
[[nodiscard]] auto ywwz() const { return Var<Vector<T, 4>>(at(1), at(3), at(3), at(2)); }
[[nodiscard]] auto ywww() const { return Var<Vector<T, 4>>(at(1), at(3), at(3), at(3)); }
[[nodiscard]] auto zxxx() const { return Var<Vector<T, 4>>(at(2), at(0), at(0), at(0)); }
[[nodiscard]] auto zxxy() const { return Var<Vector<T, 4>>(at(2), at(0), at(0), at(1)); }
[[nodiscard]] auto zxxz() const { return Var<Vector<T, 4>>(at(2), at(0), at(0), at(2)); }
[[nodiscard]] auto zxxw() const { return Var<Vector<T, 4>>(at(2), at(0), at(0), at(3)); }
[[nodiscard]] auto zxyx() const { return Var<Vector<T, 4>>(at(2), at(0), at(1), at(0)); }
[[nodiscard]] auto zxyy() const { return Var<Vector<T, 4>>(at(2), at(0), at(1), at(1)); }
[[nodiscard]] auto zxyz() const { return Var<Vector<T, 4>>(at(2), at(0), at(1), at(2)); }
[[nodiscard]] auto zxyw() const { return Var<Vector<T, 4>>(at(2), at(0), at(1), at(3)); }
[[nodiscard]] auto zxzx() const { return Var<Vector<T, 4>>(at(2), at(0), at(2), at(0)); }
[[nodiscard]] auto zxzy() const { return Var<Vector<T, 4>>(at(2), at(0), at(2), at(1)); }
[[nodiscard]] auto zxzz() const { return Var<Vector<T, 4>>(at(2), at(0), at(2), at(2)); }
[[nodiscard]] auto zxzw() const { return Var<Vector<T, 4>>(at(2), at(0), at(2), at(3)); }
[[nodiscard]] auto zxwx() const { return Var<Vector<T, 4>>(at(2), at(0), at(3), at(0)); }
[[nodiscard]] auto zxwy() const { return Var<Vector<T, 4>>(at(2), at(0), at(3), at(1)); }
[[nodiscard]] auto zxwz() const { return Var<Vector<T, 4>>(at(2), at(0), at(3), at(2)); }
[[nodiscard]] auto zxww() const { return Var<Vector<T, 4>>(at(2), at(0), at(3), at(3)); }
[[nodiscard]] auto zyxx() const { return Var<Vector<T, 4>>(at(2), at(1), at(0), at(0)); }
[[nodiscard]] auto zyxy() const { return Var<Vector<T, 4>>(at(2), at(1), at(0), at(1)); }
[[nodiscard]] auto zyxz() const { return Var<Vector<T, 4>>(at(2), at(1), at(0), at(2)); }
[[nodiscard]] auto zyxw() const { return Var<Vector<T, 4>>(at(2), at(1), at(0), at(3)); }
[[nodiscard]] auto zyyx() const { return Var<Vector<T, 4>>(at(2), at(1), at(1), at(0)); }
[[nodiscard]] auto zyyy() const { return Var<Vector<T, 4>>(at(2), at(1), at(1), at(1)); }
[[nodiscard]] auto zyyz() const { return Var<Vector<T, 4>>(at(2), at(1), at(1), at(2)); }
[[nodiscard]] auto zyyw() const { return Var<Vector<T, 4>>(at(2), at(1), at(1), at(3)); }
[[nodiscard]] auto zyzx() const { return Var<Vector<T, 4>>(at(2), at(1), at(2), at(0)); }
[[nodiscard]] auto zyzy() const { return Var<Vector<T, 4>>(at(2), at(1), at(2), at(1)); }
[[nodiscard]] auto zyzz() const { return Var<Vector<T, 4>>(at(2), at(1), at(2), at(2)); }
[[nodiscard]] auto zyzw() const { return Var<Vector<T, 4>>(at(2), at(1), at(2), at(3)); }
[[nodiscard]] auto zywx() const { return Var<Vector<T, 4>>(at(2), at(1), at(3), at(0)); }
[[nodiscard]] auto zywy() const { return Var<Vector<T, 4>>(at(2), at(1), at(3), at(1)); }
[[nodiscard]] auto zywz() const { return Var<Vector<T, 4>>(at(2), at(1), at(3), at(2)); }
[[nodiscard]] auto zyww() const { return Var<Vector<T, 4>>(at(2), at(1), at(3), at(3)); }
[[nodiscard]] auto zzxx() const { return Var<Vector<T, 4>>(at(2), at(2), at(0), at(0)); }
[[nodiscard]] auto zzxy() const { return Var<Vector<T, 4>>(at(2), at(2), at(0), at(1)); }
[[nodiscard]] auto zzxz() const { return Var<Vector<T, 4>>(at(2), at(2), at(0), at(2)); }
[[nodiscard]] auto zzxw() const { return Var<Vector<T, 4>>(at(2), at(2), at(0), at(3)); }
[[nodiscard]] auto zzyx() const { return Var<Vector<T, 4>>(at(2), at(2), at(1), at(0)); }
[[nodiscard]] auto zzyy() const { return Var<Vector<T, 4>>(at(2), at(2), at(1), at(1)); }
[[nodiscard]] auto zzyz() const { return Var<Vector<T, 4>>(at(2), at(2), at(1), at(2)); }
[[nodiscard]] auto zzyw() const { return Var<Vector<T, 4>>(at(2), at(2), at(1), at(3)); }
[[nodiscard]] auto zzzx() const { return Var<Vector<T, 4>>(at(2), at(2), at(2), at(0)); }
[[nodiscard]] auto zzzy() const { return Var<Vector<T, 4>>(at(2), at(2), at(2), at(1)); }
[[nodiscard]] auto zzzz() const { return Var<Vector<T, 4>>(at(2), at(2), at(2), at(2)); }
[[nodiscard]] auto zzzw() const { return Var<Vector<T, 4>>(at(2), at(2), at(2), at(3)); }
[[nodiscard]] auto zzwx() const { return Var<Vector<T, 4>>(at(2), at(2), at(3), at(0)); }
[[nodiscard]] auto zzwy() const { return Var<Vector<T, 4>>(at(2), at(2), at(3), at(1)); }
[[nodiscard]] auto zzwz() const { return Var<Vector<T, 4>>(at(2), at(2), at(3), at(2)); }
[[nodiscard]] auto zzww() const { return Var<Vector<T, 4>>(at(2), at(2), at(3), at(3)); }
[[nodiscard]] auto zwxx() const { return Var<Vector<T, 4>>(at(2), at(3), at(0), at(0)); }
[[nodiscard]] auto zwxy() const { return Var<Vector<T, 4>>(at(2), at(3), at(0), at(1)); }
[[nodiscard]] auto zwxz() const { return Var<Vector<T, 4>>(at(2), at(3), at(0), at(2)); }
[[nodiscard]] auto zwxw() const { return Var<Vector<T, 4>>(at(2), at(3), at(0), at(3)); }
[[nodiscard]] auto zwyx() const { return Var<Vector<T, 4>>(at(2), at(3), at(1), at(0)); }
[[nodiscard]] auto zwyy() const { return Var<Vector<T, 4>>(at(2), at(3), at(1), at(1)); }
[[nodiscard]] auto zwyz() const { return Var<Vector<T, 4>>(at(2), at(3), at(1), at(2)); }
[[nodiscard]] auto zwyw() const { return Var<Vector<T, 4>>(at(2), at(3), at(1), at(3)); }
[[nodiscard]] auto zwzx() const { return Var<Vector<T, 4>>(at(2), at(3), at(2), at(0)); }
[[nodiscard]] auto zwzy() const { return Var<Vector<T, 4>>(at(2), at(3), at(2), at(1)); }
[[nodiscard]] auto zwzz() const { return Var<Vector<T, 4>>(at(2), at(3), at(2), at(2)); }
[[nodiscard]] auto zwzw() const { return Var<Vector<T, 4>>(at(2), at(3), at(2), at(3)); }
[[nodiscard]] auto zwwx() const { return Var<Vector<T, 4>>(at(2), at(3), at(3), at(0)); }
[[nodiscard]] auto zwwy() const { return Var<Vector<T, 4>>(at(2), at(3), at(3), at(1)); }
[[nodiscard]] auto zwwz() const { return Var<Vector<T, 4>>(at(2), at(3), at(3), at(2)); }
[[nodiscard]] auto zwww() const { return Var<Vector<T, 4>>(at(2), at(3), at(3), at(3)); }
[[nodiscard]] auto wxxx() const { return Var<Vector<T, 4>>(at(3), at(0), at(0), at(0)); }
[[nodiscard]] auto wxxy() const { return Var<Vector<T, 4>>(at(3), at(0), at(0), at(1)); }
[[nodiscard]] auto wxxz() const { return Var<Vector<T, 4>>(at(3), at(0), at(0), at(2)); }
[[nodiscard]] auto wxxw() const { return Var<Vector<T, 4>>(at(3), at(0), at(0), at(3)); }
[[nodiscard]] auto wxyx() const { return Var<Vector<T, 4>>(at(3), at(0), at(1), at(0)); }
[[nodiscard]] auto wxyy() const { return Var<Vector<T, 4>>(at(3), at(0), at(1), at(1)); }
[[nodiscard]] auto wxyz() const { return Var<Vector<T, 4>>(at(3), at(0), at(1), at(2)); }
[[nodiscard]] auto wxyw() const { return Var<Vector<T, 4>>(at(3), at(0), at(1), at(3)); }
[[nodiscard]] auto wxzx() const { return Var<Vector<T, 4>>(at(3), at(0), at(2), at(0)); }
[[nodiscard]] auto wxzy() const { return Var<Vector<T, 4>>(at(3), at(0), at(2), at(1)); }
[[nodiscard]] auto wxzz() const { return Var<Vector<T, 4>>(at(3), at(0), at(2), at(2)); }
[[nodiscard]] auto wxzw() const { return Var<Vector<T, 4>>(at(3), at(0), at(2), at(3)); }
[[nodiscard]] auto wxwx() const { return Var<Vector<T, 4>>(at(3), at(0), at(3), at(0)); }
[[nodiscard]] auto wxwy() const { return Var<Vector<T, 4>>(at(3), at(0), at(3), at(1)); }
[[nodiscard]] auto wxwz() const { return Var<Vector<T, 4>>(at(3), at(0), at(3), at(2)); }
[[nodiscard]] auto wxww() const { return Var<Vector<T, 4>>(at(3), at(0), at(3), at(3)); }
[[nodiscard]] auto wyxx() const { return Var<Vector<T, 4>>(at(3), at(1), at(0), at(0)); }
[[nodiscard]] auto wyxy() const { return Var<Vector<T, 4>>(at(3), at(1), at(0), at(1)); }
[[nodiscard]] auto wyxz() const { return Var<Vector<T, 4>>(at(3), at(1), at(0), at(2)); }
[[nodiscard]] auto wyxw() const { return Var<Vector<T, 4>>(at(3), at(1), at(0), at(3)); }
[[nodiscard]] auto wyyx() const { return Var<Vector<T, 4>>(at(3), at(1), at(1), at(0)); }
[[nodiscard]] auto wyyy() const { return Var<Vector<T, 4>>(at(3), at(1), at(1), at(1)); }
[[nodiscard]] auto wyyz() const { return Var<Vector<T, 4>>(at(3), at(1), at(1), at(2)); }
[[nodiscard]] auto wyyw() const { return Var<Vector<T, 4>>(at(3), at(1), at(1), at(3)); }
[[nodiscard]] auto wyzx() const { return Var<Vector<T, 4>>(at(3), at(1), at(2), at(0)); }
[[nodiscard]] auto wyzy() const { return Var<Vector<T, 4>>(at(3), at(1), at(2), at(1)); }
[[nodiscard]] auto wyzz() const { return Var<Vector<T, 4>>(at(3), at(1), at(2), at(2)); }
[[nodiscard]] auto wyzw() const { return Var<Vector<T, 4>>(at(3), at(1), at(2), at(3)); }
[[nodiscard]] auto wywx() const { return Var<Vector<T, 4>>(at(3), at(1), at(3), at(0)); }
[[nodiscard]] auto wywy() const { return Var<Vector<T, 4>>(at(3), at(1), at(3), at(1)); }
[[nodiscard]] auto wywz() const { return Var<Vector<T, 4>>(at(3), at(1), at(3), at(2)); }
[[nodiscard]] auto wyww() const { return Var<Vector<T, 4>>(at(3), at(1), at(3), at(3)); }
[[nodiscard]] auto wzxx() const { return Var<Vector<T, 4>>(at(3), at(2), at(0), at(0)); }
[[nodiscard]] auto wzxy() const { return Var<Vector<T, 4>>(at(3), at(2), at(0), at(1)); }
[[nodiscard]] auto wzxz() const { return Var<Vector<T, 4>>(at(3), at(2), at(0), at(2)); }
[[nodiscard]] auto wzxw() const { return Var<Vector<T, 4>>(at(3), at(2), at(0), at(3)); }
[[nodiscard]] auto wzyx() const { return Var<Vector<T, 4>>(at(3), at(2), at(1), at(0)); }
[[nodiscard]] auto wzyy() const { return Var<Vector<T, 4>>(at(3), at(2), at(1), at(1)); }
[[nodiscard]] auto wzyz() const { return Var<Vector<T, 4>>(at(3), at(2), at(1), at(2)); }
[[nodiscard]] auto wzyw() const { return Var<Vector<T, 4>>(at(3), at(2), at(1), at(3)); }
[[nodiscard]] auto wzzx() const { return Var<Vector<T, 4>>(at(3), at(2), at(2), at(0)); }
[[nodiscard]] auto wzzy() const { return Var<Vector<T, 4>>(at(3), at(2), at(2), at(1)); }
[[nodiscard]] auto wzzz() const { return Var<Vector<T, 4>>(at(3), at(2), at(2), at(2)); }
[[nodiscard]] auto wzzw() const { return Var<Vector<T, 4>>(at(3), at(2), at(2), at(3)); }
[[nodiscard]] auto wzwx() const { return Var<Vector<T, 4>>(at(3), at(2), at(3), at(0)); }
[[nodiscard]] auto wzwy() const { return Var<Vector<T, 4>>(at(3), at(2), at(3), at(1)); }
[[nodiscard]] auto wzwz() const { return Var<Vector<T, 4>>(at(3), at(2), at(3), at(2)); }
[[nodiscard]] auto wzww() const { return Var<Vector<T, 4>>(at(3), at(2), at(3), at(3)); }
[[nodiscard]] auto wwxx() const { return Var<Vector<T, 4>>(at(3), at(3), at(0), at(0)); }
[[nodiscard]] auto wwxy() const { return Var<Vector<T, 4>>(at(3), at(3), at(0), at(1)); }
[[nodiscard]] auto wwxz() const { return Var<Vector<T, 4>>(at(3), at(3), at(0), at(2)); }
[[nodiscard]] auto wwxw() const { return Var<Vector<T, 4>>(at(3), at(3), at(0), at(3)); }
[[nodiscard]] auto wwyx() const { return Var<Vector<T, 4>>(at(3), at(3), at(1), at(0)); }
[[nodiscard]] auto wwyy() const { return Var<Vector<T, 4>>(at(3), at(3), at(1), at(1)); }
[[nodiscard]] auto wwyz() const { return Var<Vector<T, 4>>(at(3), at(3), at(1), at(2)); }
[[nodiscard]] auto wwyw() const { return Var<Vector<T, 4>>(at(3), at(3), at(1), at(3)); }
[[nodiscard]] auto wwzx() const { return Var<Vector<T, 4>>(at(3), at(3), at(2), at(0)); }
[[nodiscard]] auto wwzy() const { return Var<Vector<T, 4>>(at(3), at(3), at(2), at(1)); }
[[nodiscard]] auto wwzz() const { return Var<Vector<T, 4>>(at(3), at(3), at(2), at(2)); }
[[nodiscard]] auto wwzw() const { return Var<Vector<T, 4>>(at(3), at(3), at(2), at(3)); }
[[nodiscard]] auto wwwx() const { return Var<Vector<T, 4>>(at(3), at(3), at(3), at(0)); }
[[nodiscard]] auto wwwy() const { return Var<Vector<T, 4>>(at(3), at(3), at(3), at(1)); }
[[nodiscard]] auto wwwz() const { return Var<Vector<T, 4>>(at(3), at(3), at(3), at(2)); }
[[nodiscard]] auto wwww() const { return Var<Vector<T, 4>>(at(3), at(3), at(3), at(3)); }
