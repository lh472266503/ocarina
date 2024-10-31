from os.path import realpath, dirname


def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    color_entries = ["r", "g", "b", "a"][:dim]

    print("template<typename T, typename M>", file=file)
    print(f"void export_swizzle{dim}(M &m) {{", file=file)

    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            print(f"    m.def_property(\"{x}{y}\", [](Vector<T, {dim}> &self) {{ return self.{x}{y}().decay(); }}, [](Vector<T, {dim}> &self, Vector<T, 2> v) {{self.{x}{y}() = v;}});", file=file)

    print(f"", file=file)
    for i, x in enumerate(entries):
        for j, y in enumerate(entries):
            for k, z in enumerate(entries):
                print(f"    m.def_property(\"{x}{y}{z}\", [](Vector<T, {dim}> &self) {{ return self.{x}{y}{z}().decay(); }}, [](Vector<T, {dim}> &self, Vector<T, 3> v) {{self.{x}{y}{z}() = v;}});", file=file)

    print(f"", file=file)
    for i, x in enumerate(entries):
        for j, y in enumerate(entries):
            for k, z in enumerate(entries):
                for n, w in enumerate(entries):
                    print(f"    m.def_property(\"{x}{y}{z}{w}\", [](Vector<T, {dim}> &self) {{ return self.{x}{y}{z}{w}().decay(); }}, [](Vector<T, {dim}> &self, Vector<T, 4> v) {{self.{x}{y}{z}{w}() = v;}});", file=file)

    print("}", file=file)


if __name__ == "__main__":
    base = dirname(realpath(__file__))
    for dim in range(2, 5):
        with open(f"{base}/swizzle{dim}.inl.h", "w") as file:
            generate(file, dim)
