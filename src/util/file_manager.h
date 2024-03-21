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
private:
    FileManager() = default;
    FileManager(const FileManager &) = delete;
    FileManager(FileManager &&) = delete;
    FileManager operator=(const FileManager &) = delete;
    FileManager operator=(FileManager &&) = delete;
    static FileManager *s_file_manager;

public:
    [[nodiscard]] static FileManager &instance() noexcept;
    static void destroy_instance();

private:
    struct Impl;
    ocarina::unique_ptr<Impl> _impl;

public:
    explicit FileManager(const fs::path &path, string_view cache_dir = ".cache");
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

#ifndef NDEBUG

#include "file_manager_impl.h"

#endif