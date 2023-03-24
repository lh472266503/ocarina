from os.path import realpath, dirname


def generate(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            str = f"[[nodiscard]] auto {x}{y}() const {{ return eval<Vector<T, 2>>(Function::current()->swizzle(Type::of<T>(), expression(), 0x{i}{j}, 2)); }}"
            print(str, file=file)
    print("", file=file)
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            for k,z in enumerate(entries):
                str = f"[[nodiscard]] auto {x}{y}{z}() const {{ return eval<Vector<T, 3>>(Function::current()->swizzle(Type::of<T>(), expression(), 0x{i}{j}{k}, 3)); }}"
                print(str, file=file)
    print("", file=file)
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            for k,z in enumerate(entries):
                for l,w in enumerate(entries):
                    str = f"[[nodiscard]] auto {x}{y}{z}{w}() const {{ return eval<Vector<T, 4>>(Function::current()->swizzle(Type::of<T>(), expression(), 0x{i}{j}{k}{l}, 4)); }}"
                    print(str, file=file)

def array_swizzle(file, dim):
    entries = ["x", "y", "z", "w"][:dim]
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            str = f"[[nodiscard]] Array<T> {x}{y}() const {{ OC_ASSERT(_size >= 2); return Array<T>::create(at({i}), at({j})); }}"
            print(str, file=file)
    print("", file=file)
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            for k,z in enumerate(entries):
                str = f"[[nodiscard]] Array<T> {x}{y}{z}() const {{ OC_ASSERT(_size >= 3); return Array<T>::create(at({i}), at({j}), at({k})); }}"
                print(str, file=file)
    print("", file=file)
    for i,x in enumerate(entries):
        for j,y in enumerate(entries):
            for k,z in enumerate(entries):
                for l,w in enumerate(entries):
                    str = f"[[nodiscard]] Array<T> {x}{y}{z}{w}() const {{  OC_ASSERT(_size >= 4); return Array<T>::create(at({i}), at({j}), at({k}), at({l})); }}"
                    print(str, file=file)

if __name__ == "__main__":
    base = dirname(realpath(__file__))
    # for i,k in enumerate(["a", "b"]):
    #     print(i,k)
    for dim in range(2, 5):
        with open(f"{base}/swizzle_{dim}.inl.h", "w") as file:
            generate(file, dim)

    with open(f"{base}/array_swizzle.inl.h", "w") as file:
        array_swizzle(file, 4)