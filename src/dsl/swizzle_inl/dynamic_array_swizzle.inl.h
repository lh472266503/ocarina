[[nodiscard]] DynamicArray<T> x_() const { OC_ASSERT(size_ > 0); return DynamicArray<T>::create(at(0)); }
[[nodiscard]] DynamicArray<T> y_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1)); }
[[nodiscard]] DynamicArray<T> z_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2)); }
[[nodiscard]] DynamicArray<T> w_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3)); }

[[nodiscard]] DynamicArray<T> xx_() const { OC_ASSERT(size_ > 0); return DynamicArray<T>::create(at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3)); }
[[nodiscard]] DynamicArray<T> yx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0)); }
[[nodiscard]] DynamicArray<T> yy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1)); }
[[nodiscard]] DynamicArray<T> yz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2)); }
[[nodiscard]] DynamicArray<T> yw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2)); }
[[nodiscard]] DynamicArray<T> ww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3)); }

[[nodiscard]] DynamicArray<T> xxx_() const { OC_ASSERT(size_ > 0); return DynamicArray<T>::create(at(0), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> xyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> xyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> xyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> xyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> xzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> xzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> xzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> xzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> xwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> xwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> xwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> xww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> yxx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> yxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> yxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> yxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> yyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> yyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> yyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> yyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> yzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> yzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> yzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> yzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> ywx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> ywy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> ywz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> yww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> zxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> zxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> zxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> zxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> zyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> zyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> zyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> zyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> zwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> zwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> zwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> zww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> wxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> wxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> wxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> wxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> wyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> wyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> wyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> wyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> wzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> wzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> wzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> wzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> www_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(3)); }

[[nodiscard]] DynamicArray<T> xxxx_() const { OC_ASSERT(size_ > 0); return DynamicArray<T>::create(at(0), at(0), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xxxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(0), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xxxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xxxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> xxyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(0), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> xxyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(0), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> xxyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> xxyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> xxzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> xxzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> xxzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(0), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> xxzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> xxwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> xxwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> xxwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> xxww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(0), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> xyxx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xyxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xyxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xyxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> xyyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> xyyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(0), at(1), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> xyyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> xyyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> xyzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> xyzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> xyzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(1), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> xyzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> xywx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> xywy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> xywz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> xyww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(1), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> xzxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xzxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xzxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xzxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> xzyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> xzyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> xzyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> xzyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> xzzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> xzzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> xzzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(0), at(2), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> xzzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> xzwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> xzwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> xzwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> xzww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(2), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> xwxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> xwxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> xwxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> xwxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> xwyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> xwyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> xwyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> xwyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> xwzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> xwzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> xwzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> xwzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> xwwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> xwwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> xwwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> xwww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(0), at(3), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> yxxx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> yxxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> yxxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> yxxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> yxyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> yxyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(0), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> yxyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> yxyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> yxzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> yxzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> yxzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(0), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> yxzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> yxwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> yxwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> yxwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> yxww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(0), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> yyxx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> yyxy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> yyxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> yyxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> yyyx_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> yyyy_() const { OC_ASSERT(size_ > 1); return DynamicArray<T>::create(at(1), at(1), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> yyyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> yyyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> yyzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> yyzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> yyzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(1), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> yyzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> yywx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> yywy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> yywz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> yyww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(1), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> yzxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> yzxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> yzxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> yzxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> yzyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> yzyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> yzyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> yzyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> yzzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> yzzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> yzzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(1), at(2), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> yzzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> yzwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> yzwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> yzwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> yzww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(2), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> ywxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> ywxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> ywxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> ywxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> ywyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> ywyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> ywyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> ywyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> ywzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> ywzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> ywzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> ywzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> ywwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> ywwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> ywwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> ywww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(1), at(3), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> zxxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> zxxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> zxxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> zxxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> zxyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> zxyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> zxyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> zxyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zxzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zxzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zxzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(0), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zxzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> zxwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> zxwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> zxwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> zxww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(0), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> zyxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> zyxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> zyxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> zyxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> zyyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> zyyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> zyyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> zyyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zyzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zyzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zyzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(1), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zyzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> zywx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> zywy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> zywz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> zyww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(1), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> zzxx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> zzxy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> zzxz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> zzxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> zzyx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> zzyy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> zzyz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> zzyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zzzx_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zzzy_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zzzz_() const { OC_ASSERT(size_ > 2); return DynamicArray<T>::create(at(2), at(2), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zzzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> zzwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> zzwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> zzwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> zzww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(2), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> zwxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> zwxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> zwxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> zwxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> zwyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> zwyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> zwyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> zwyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> zwzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> zwzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> zwzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> zwzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> zwwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> zwwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> zwwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> zwww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(2), at(3), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> wxxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> wxxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> wxxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> wxxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> wxyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> wxyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> wxyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> wxyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> wxzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> wxzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> wxzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> wxzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wxwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wxwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wxwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> wxww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(0), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> wyxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> wyxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> wyxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> wyxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> wyyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> wyyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> wyyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> wyyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> wyzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> wyzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> wyzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> wyzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wywx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wywy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wywz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> wyww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(1), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> wzxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> wzxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> wzxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> wzxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> wzyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> wzyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> wzyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> wzyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> wzzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> wzzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> wzzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> wzzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wzwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wzwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wzwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> wzww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(2), at(3), at(3)); }
[[nodiscard]] DynamicArray<T> wwxx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(0), at(0)); }
[[nodiscard]] DynamicArray<T> wwxy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(0), at(1)); }
[[nodiscard]] DynamicArray<T> wwxz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(0), at(2)); }
[[nodiscard]] DynamicArray<T> wwxw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(0), at(3)); }
[[nodiscard]] DynamicArray<T> wwyx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(1), at(0)); }
[[nodiscard]] DynamicArray<T> wwyy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(1), at(1)); }
[[nodiscard]] DynamicArray<T> wwyz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(1), at(2)); }
[[nodiscard]] DynamicArray<T> wwyw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(1), at(3)); }
[[nodiscard]] DynamicArray<T> wwzx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(2), at(0)); }
[[nodiscard]] DynamicArray<T> wwzy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(2), at(1)); }
[[nodiscard]] DynamicArray<T> wwzz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(2), at(2)); }
[[nodiscard]] DynamicArray<T> wwzw_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(2), at(3)); }
[[nodiscard]] DynamicArray<T> wwwx_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(3), at(0)); }
[[nodiscard]] DynamicArray<T> wwwy_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(3), at(1)); }
[[nodiscard]] DynamicArray<T> wwwz_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(3), at(2)); }
[[nodiscard]] DynamicArray<T> wwww_() const { OC_ASSERT(size_ > 3); return DynamicArray<T>::create(at(3), at(3), at(3), at(3)); }
