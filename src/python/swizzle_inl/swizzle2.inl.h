template<typename T, typename M>
void export_swizzle2(M &m) {
    m.def_property("xx", [](Vector<T, 2> &self) { return self.xx().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) {self.xx() = v;});
    m.def_property("xy", [](Vector<T, 2> &self) { return self.xy().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) {self.xy() = v;});
    m.def_property("yx", [](Vector<T, 2> &self) { return self.yx().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) {self.yx() = v;});
    m.def_property("yy", [](Vector<T, 2> &self) { return self.yy().decay(); }, [](Vector<T, 2> &self, Vector<T, 2> v) {self.yy() = v;});

    m.def_property("xxx", [](Vector<T, 2> &self) { return self.xxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.xxx() = v;});
    m.def_property("xxy", [](Vector<T, 2> &self) { return self.xxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.xxy() = v;});
    m.def_property("xyx", [](Vector<T, 2> &self) { return self.xyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.xyx() = v;});
    m.def_property("xyy", [](Vector<T, 2> &self) { return self.xyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.xyy() = v;});
    m.def_property("yxx", [](Vector<T, 2> &self) { return self.yxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.yxx() = v;});
    m.def_property("yxy", [](Vector<T, 2> &self) { return self.yxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.yxy() = v;});
    m.def_property("yyx", [](Vector<T, 2> &self) { return self.yyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.yyx() = v;});
    m.def_property("yyy", [](Vector<T, 2> &self) { return self.yyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 3> v) {self.yyy() = v;});

    m.def_property("xxxx", [](Vector<T, 2> &self) { return self.xxxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xxxx() = v;});
    m.def_property("xxxy", [](Vector<T, 2> &self) { return self.xxxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xxxy() = v;});
    m.def_property("xxyx", [](Vector<T, 2> &self) { return self.xxyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xxyx() = v;});
    m.def_property("xxyy", [](Vector<T, 2> &self) { return self.xxyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xxyy() = v;});
    m.def_property("xyxx", [](Vector<T, 2> &self) { return self.xyxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xyxx() = v;});
    m.def_property("xyxy", [](Vector<T, 2> &self) { return self.xyxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xyxy() = v;});
    m.def_property("xyyx", [](Vector<T, 2> &self) { return self.xyyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xyyx() = v;});
    m.def_property("xyyy", [](Vector<T, 2> &self) { return self.xyyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.xyyy() = v;});
    m.def_property("yxxx", [](Vector<T, 2> &self) { return self.yxxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yxxx() = v;});
    m.def_property("yxxy", [](Vector<T, 2> &self) { return self.yxxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yxxy() = v;});
    m.def_property("yxyx", [](Vector<T, 2> &self) { return self.yxyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yxyx() = v;});
    m.def_property("yxyy", [](Vector<T, 2> &self) { return self.yxyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yxyy() = v;});
    m.def_property("yyxx", [](Vector<T, 2> &self) { return self.yyxx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yyxx() = v;});
    m.def_property("yyxy", [](Vector<T, 2> &self) { return self.yyxy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yyxy() = v;});
    m.def_property("yyyx", [](Vector<T, 2> &self) { return self.yyyx().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yyyx() = v;});
    m.def_property("yyyy", [](Vector<T, 2> &self) { return self.yyyy().decay(); }, [](Vector<T, 2> &self, Vector<T, 4> v) {self.yyyy() = v;});
}
