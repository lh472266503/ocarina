from os.path import realpath, dirname

def main():
    content = "#pragma once\n"
    scalar_types = ["int", "uint", "float", "bool"]
    native_types = ["int", "unsigned int", "float", "bool"]
    vector_alignments = {2: 8, 3: 16, 4: 16}
    
    print(content)

if __name__ == "__main__":
    main()