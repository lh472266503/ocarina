//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "core/concepts.h"
#include "GUI/decl.h"

namespace ocarina {
class Device;
class DynamicModule;
class FileManager : public concepts::Noncopyable {

    OC_MAKE_INSTANCE_CONSTRUCTOR(FileManager, s_file_manager)
    OC_MAKE_INSTANCE_FUNC_DECL(FileManager)

private:
    struct Impl {
        fs::path runtime_directory;
        fs::path cache_directory;
        bool use_cache{true};
        ocarina::map<string, DynamicModule> modules;
        Impl() = default;
    };
    ocarina::unique_ptr<Impl> impl_;

public:
    FileManager &init(const fs::path &path, string_view cache_dir = ".cache");
    virtual ~FileManager() noexcept;
    [[nodiscard]] const fs::path &runtime_directory() const noexcept;
    [[nodiscard]] const fs::path &cache_directory() const noexcept;
    static bool create_directory_if_necessary(const fs::path &path);
    void write_global_cache(const string &fn, const string &text) const noexcept;
    [[nodiscard]] string read_global_cache(const string &fn) const noexcept;
    [[nodiscard]] static string read_file(const fs::path &fn);
    static void write_file(const fs::path &fn, const string &text);
    void clear_cache() const noexcept;
    [[nodiscard]] bool is_exist_cache(const string &fn) const noexcept;
    const DynamicModule *obtain_module(const string &module_name) noexcept;
    bool unload_module(const string &module_name) noexcept;
    void unload_module(void *handle) noexcept;
    [[nodiscard]] Device create_device(const string &backend_name) noexcept;
    [[nodiscard]] WindowWrapper create_window(const char *name, uint2 initial_size, const char *type = "gl", bool resizable = false);
};

}// namespace ocarina