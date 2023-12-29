//
// Created by Zero on 06/06/2022.
//

#include "dynamic_module.h"
#include "core/platform.h"
#include "core/logging.h"

namespace ocarina {

ocarina::vector<fs::path> &DynamicModule::_search_path() {
    static ocarina::vector<fs::path> ret;
    return ret;
}

void DynamicModule::add_search_path(fs::path path) noexcept{
    path = fs::canonical(path);
    if (std::find(_search_path().begin(), _search_path().end(), path) == _search_path().end()) {
        _search_path().push_back(path);
    }
}

void DynamicModule::remove_search_path(fs::path path) noexcept{
    path = fs::canonical(path);
    auto iter = std::find(_search_path().begin(), _search_path().end(), path);
    if (iter != _search_path().end()) {
        _search_path().erase(iter);
    }
}

void DynamicModule::clear_search_path() noexcept {
    _search_path().clear();
}

DynamicModule::DynamicModule(const string &name) noexcept {
    for (const auto &path : _search_path()) {
        _handle = dynamic_module_load(path / name);
        if (_handle) {
            OC_DEBUG_FORMAT_WITH_LOCATION("load {} success", (path/ name).string());
            return;
        }
    }
    OC_ERROR_FORMAT("load {} fail!", name);
}

DynamicModule::DynamicModule(fs::path path, const string &name) noexcept {
    path = fs::canonical(path);
    _handle = dynamic_module_load(path / name);
    if (_handle) {
        OC_INFO_FORMAT_WITH_LOCATION("load {} in {}", name, path.string());
    } else {
        OC_ERROR_FORMAT("load {} fail in", (path / name).string());
    }
}
void *DynamicModule::function_ptr(const string &func_name) const noexcept {
    return dynamic_module_find_symbol(const_cast<void*>(_handle), func_name);
}

}// namespace ocarina