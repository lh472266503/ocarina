//
// Created by Zero on 16/05/2022.
//

#include "printer.h"

namespace ocarina {

Printer *Printer::s_printer = nullptr;

Printer &Printer::instance() noexcept {
    if (s_printer == nullptr) {
        s_printer = new Printer();
    }
    return *s_printer;
}

void Printer::destroy_instance() noexcept {
    if (s_printer) {
        delete s_printer;
        s_printer = nullptr;
    }
}

void Printer::retrieve_immediately() noexcept {
    _buffer.download_immediately();
    uint length = std::min(
        static_cast<uint>(_buffer.host().size() - 1u),
        _buffer.back());
    bool truncated = _buffer.host().back() > length;
    uint offset = 0u;
    while (offset < length) {
        const uint *data = _buffer.host().data() + offset;
        Item item = _items[data[0]];
        offset += item.size + 1;
        if (offset > length) {
            truncated = true;
        } else {
            item.func(data + 1);
        }
    }
    if (truncated) [[unlikely]] {
        OC_WARNING("Kernel log truncated.");
    }
    reset();
}
}// namespace ocarina