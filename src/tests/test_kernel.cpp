//
// Created by Zero on 26/04/2022.
//

#include "core/stl.h"
#include "dsl/common.h"
#include "runtime/context.h"
#include "generator/cpp_codegen.h"
#include "runtime/device.h"
#include "runtime/stream.h"
#include "runtime/command.h"

using namespace ocarina;

int main(int argc, char *argv[]) {
    Callable add = [&](Var<float> a, Var<float> b) {
        return a + b;
    };

    fs::path path(argv[0]);
    Context context(path.parent_path());
    Device device = context.create_device("cuda");
    Stream stream = device.create_stream();

    auto cmd = BufferUploadCommand::create(nullptr, 0, 0, 0);


    return 0;
}
