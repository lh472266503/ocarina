from ctypes import alignment
from os.path import realpath, dirname
import os
from posixpath import split
import struct

scalar_types = ["int", "uint", "float", "bool"]
native_types = ["int", "unsigned int", "float", "bool"]
vector_alignments = {2: 8, 3: 16, 4: 16}
indent = "\t"
name_lst = ["x", "y", "z", "w"]

prefix = "oc"

content = "#pragma once\n\n"

def get_indent(num = 1):
    return indent * num

def using_scalar():
    global content
    string = ""
    for i,scalar in enumerate(scalar_types):
        string += f"using {prefix}_{scalar} = {native_types[i]};\n"
    
    content += string

def emit_member(scalar_name, dim):
    ret = "\n"
    for d in range(0, dim):
        ret += indent + scalar_name + " " + name_lst[d] + ";\n"
    return ret

def emit_functions(scalar_name, dim):
    ret = "\n"
    struct_name = f"{scalar_name}{dim}"

    ret += indent + f"__device__ {struct_name}() noexcept \n" + get_indent(2) + ":"
    for d in range(0, dim):
        split = "," if d != dim - 1 else ""
        ret += f"{name_lst[d]}{{}}" + split 
    ret += " {}\n"

    ret += indent + f"__device__ {struct_name}({scalar_name} s) noexcept \n" + get_indent(2) + ":"
    for d in range(0, dim):
        split = "," if d != dim - 1 else ""
        ret += f"{name_lst[d]}(s)" + split 
    ret += " {}\n"

    return ret

def define_vector():
    global content
    content += "\n"
    for dim in range(2, 5):
        alignment = vector_alignments[dim];
        for j, scalar in enumerate(scalar_types):
            scalar_name = f"{prefix}_{scalar}"
            struct_name = f"struct alignas({alignment}) {scalar_name}{dim}" + "{"
            content += struct_name
            body = emit_member(scalar_name, dim)
            body += emit_functions(scalar_name, dim)
            content += body
            content += "};\n\n"

def save_to_inl(var_name, content, fn):
    string = f"static const char {var_name}[{len(content)}] = " + "{\n    "
    line_len = 39
    for i,s in enumerate(content):
        split = "," if i != len(content) - 1 else ""
        string += f"{ord(s)}" + split

        if i % line_len == line_len - 1:
            string += "\n    ";
        print(ord(s))
    string += "};"
    with open(fn, "w") as file:
        file.write(string)
        file.close()

def main():
    curr_dir = dirname(realpath(__file__))
    using_scalar()
    define_vector()
    print(content)

    math_lib = "cuda_math_lib"
    with open(os.path.join(curr_dir, math_lib + ".h"), "w") as file:
        file.write(content)
        file.close()
    
    save_to_inl(math_lib, content, os.path.join(curr_dir, math_lib + "_embed.h"))

if __name__ == "__main__":
    main()