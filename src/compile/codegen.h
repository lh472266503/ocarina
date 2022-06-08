//
// Created by Zero on 06/06/2022.
//

#pragma once

#include "core/header.h"
#include "core/stl.h"
#include "dsl/func.h"

namespace ocarina {

class Codegen {
private:
    class Scratch {
    private:
        ocarina::string _buffer;

    public:
        Scratch() = default;
        Scratch &operator<<(ocarina::string_view v) noexcept;
        Scratch &operator<<(ocarina::string v) noexcept;
        Scratch &operator<<(const char *v) noexcept;
        Scratch &operator<<(int v) noexcept;
        Scratch &operator<<(float v) noexcept;
        Scratch &operator<<(bool v) noexcept;
        void clear() noexcept;
        [[nodiscard]] const char *c_str() const noexcept;
        [[nodiscard]] ocarina::string_view view() const noexcept;
        [[nodiscard]] bool empty() const noexcept;
        [[nodiscard]] size_t size() const noexcept;
    };

protected:
    Scratch _scratch;

public:
    Codegen() = default;
    explicit Codegen(Scratch &scratch)
        : _scratch(scratch) {}
    virtual void emit(Function func) = 0;
};

}