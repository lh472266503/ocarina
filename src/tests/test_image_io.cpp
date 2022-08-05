//
// Created by Zero on 05/08/2022.
//

#include "util/image.h"

using namespace ocarina;

int main() {
    auto path = R"(E:/work/compile/ocarina/res/test.png)";
    auto path2 = R"(E:/work/compile/ocarina/res/test.tga)";
    auto image = Image::load(path, LINEAR);
    image.save(path2);
    return 0;
}