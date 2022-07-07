//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "runtime/context.h"
#include "generator/cpp_codegen.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());

    return 0;
}
