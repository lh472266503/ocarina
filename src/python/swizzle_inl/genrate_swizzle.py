from os.path import realpath, dirname


def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    color_entries = ["r", "g", "b", "a"][:dim]
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            print(f".def_property(\"{x}{y}\", [](Vector<T, 2> &self) {{ return self.{x}{y}().decay(); }}, [](Vector<T, 2> &self, Vector<T, 2> v) {{self.{x}{y}() = v;}}) \\", file=file)


    for i, x in enumerate(entries):
        for j, y in enumerate(entries):
            for k, z in enumerate(entries):
                print(f".def_property(\"{x}{y}{z}\", [](Vector<T, 3> &self) {{ return self.{x}{y}{z}().decay(); }}, [](Vector<T, 3> &self, Vector<T, 3> v) {{self.{x}{y}{z}() = v;}}) \\", file=file)


    for i, x in enumerate(entries):
        for j, y in enumerate(entries):
            for k, z in enumerate(entries):
                for n, w in enumerate(entries):
                    print(f".def_property(\"{x}{y}{z}{w}\", [](Vector<T, 4> &self) {{ return self.{x}{y}{z}{w}().decay(); }}, [](Vector<T, 4> &self, Vector<T, 4> v) {{self.{x}{y}{z}{w}() = v;}}) \\", file=file)

    


if __name__ == "__main__":
    base = dirname(realpath(__file__))
    for dim in range(2, 5):
        with open(f"{base}/swizzle{dim}.inl.h", "w") as file:
            generate(file, dim)
