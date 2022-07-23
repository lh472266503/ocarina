from os.path import realpath, dirname
import os

scalar_types = ["int", "uint", "float", "bool"]
native_types = ["int", "unsigned int", "float", "bool"]
vector_alignments = {2: 8, 3: 16, 4: 16}

prefix = "oc"

content = "#pragma once\n\n"

def using_scalar():
    global content
    string = ""
    for i,scalar in enumerate(scalar_types):
        string += f"using {prefix}_{scalar} = {native_types[i]};\n"
    
    content += string

def define_vector():
    global content

def save_to_inl(var_name, content, fn):
    string = f"const char {var_name}[] = " + "{"
    for s in content:
        string += s + f"'{s}'" + ","
    string += "}"
    with open(fn, "w") as file:
        file.write(string)
        file.close()

def main():
    curr_dir = dirname(realpath(__file__))
    using_scalar()
    print(content)





    
    math_lib = "cuda_math_lib"
    # with open(os.path.join(curr_dir, math_lib + ".h"), "w") as file:
    #     file.write(content)
    #     file.close()
    
    save_to_inl(math_lib, content, os.path.join(curr_dir, math_lib + "_embed.h"))

if __name__ == "__main__":
    main()