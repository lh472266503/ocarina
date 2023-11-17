//
// Created by Zero 2023/11/17.
//

#include "debugger.h"
#include "syntax.h"

namespace ocarina {

Debugger *Debugger::s_debugger = nullptr;

Debugger &Debugger::instance() noexcept {
    if (s_debugger == nullptr) {
        s_debugger = new Debugger();
    }
    return *s_debugger;
}

void Debugger::destroy_instance() noexcept {
    if (s_debugger) {
        delete s_debugger;
        s_debugger = nullptr;
    }
}

}// namespace ocarina