//
// Created by Zero on 2024/9/26.
//

#include "util/image.h"
#include "core/stl.h"
#include "dsl/dsl.h"
#include "rhi/common.h"
#include "math/base.h"
#include "base/scattering/interaction.h"
#include "core/platform.h"
#include "util/file_manager.h"

using namespace ocarina;

void test_list(Device &device, Stream& stream) {
    using Elm = float4x4;
    size_t size = 10;
    auto list = device.create_list<Elm, SOA>(size);

    RegistrableList<Elm, SOA> rl{};
    vector<Elm> host;
    rl.set_list(std::move(list));

    host.resize(size , make_float4x4(2));

    BindlessArray bindless_array = device.create_bindless_array();
    rl.set_bindless_array(bindless_array);
    rl.register_self();
    stream << bindless_array->upload_buffer_handles(true) ;
    stream << bindless_array->upload_texture_handles(true) ;

    rl.super().clear_immediately();

    Kernel kernel = [&](Uint i) {
      Var mat = make_float4x4(dispatch_id() + 1);

      Var a = rl.read(dispatch_id());

        $info("\n {} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  {}\n",
              a[0], a[1], a[2], a[3], rl.advance_index());

        rl.write(dispatch_id(), mat);
//        rl.at(dispatch_id()) = mat;

        a = rl.read(dispatch_id());

        $info("\nnew  {} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  \n"
              "{} {} {} {}  {}\n",
              a[0], a[1], a[2], a[3], rl.advance_index());

    };
    auto shader = device.compile(kernel, "test_list");
    stream << rl->storage_segment().upload(host.data());
    stream << shader(6).dispatch(3);
    stream << rl->storage_segment().download(host.data());
    stream << Env::printer().retrieve();
    stream << synchronize() << commit();

    auto hs = rl.super().host_count();

    int i = 0;
}

int main(int argc, char *argv[]) {
    fs::path path(argv[0]);
    FileManager &file_manager = FileManager::instance();

    /**
     * Conventional scheme
     * create device and stream
     * stream used for process some command,e.g buffer upload and download, dispatch shader
     * default is asynchronous operation
     */
    Device device = file_manager.create_device("cuda");
    Stream stream = device.create_stream();
    Env::printer().init(device);
    Env::debugger().init(device);

    //    Env::set_code_obfuscation(true);
    Env::set_valid_check(false);

    test_list(device, stream);

    return 0;
}