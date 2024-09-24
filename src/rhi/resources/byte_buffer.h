//
// Created by Zero on 2024/2/22.
//

#pragma once

#include "resource.h"
#include "rhi/command.h"
#include "rhi/stats.h"
#include "bindless_array.h"
#include "buffer.h"

namespace ocarina {

class ByteBuffer;

class ByteBufferView {
private:
    handle_ty handle_{};
    size_t offset_{};
    size_t size_{};
    size_t total_size_{};
    mutable BufferProxy<> proxy_{};

public:
    ByteBufferView() = default;

    ByteBufferView(handle_ty handle, size_t offset, size_t size, size_t total_size)
        : handle_(handle), offset_(offset), size_(size), total_size_(total_size) {}

    ByteBufferView(handle_ty handle, size_t total_size)
        : handle_(handle), offset_(0), size_(total_size), total_size_(total_size) {}

    inline ByteBufferView(const ByteBuffer &buffer);

    const BufferProxy<> &proxy() const noexcept {
        proxy_.handle = reinterpret_cast<std::byte *>(head());
        proxy_.size = size_;
        return proxy_;
    }

    const BufferProxy<> *proxy_ptr() const noexcept {
        return &proxy();
    }

    [[nodiscard]] ByteBufferView subview(size_t offset, size_t size) const noexcept {
        return {handle_, offset_ + offset, size, total_size_};
    }

    template<typename T>
    [[nodiscard]] BufferView<T> view_as(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? size_ - offset : size;
        return BufferView<T>(handle_, (offset_ + offset) / sizeof(T),
                             size / sizeof(T), total_size_ / sizeof(T));
    }

    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t element_size() const noexcept { return 1; }
    [[nodiscard]] size_t size_in_byte() const noexcept { return size_ * element_size(); }
    [[nodiscard]] handle_ty head() const { return handle_ + offset_ * element_size(); }

    [[nodiscard]] BufferCopyCommand *copy_from(const ByteBufferView &src, bool async = true,
                                               uint dst_offset = 0) noexcept {
        return BufferCopyCommand::create(src.head(), head(), 0, dst_offset * element_size(),
                                         src.size_in_byte(), true);
    }

    template<typename Arg>
    requires is_buffer_or_view_v<Arg>
    [[nodiscard]] BufferCopyCommand *copy_to(Arg &dst, uint src_offset = 0,
                                             bool async = true) noexcept {
        return BufferCopyCommand::create(head(), dst.head(), src_offset * element_size(),
                                         0, dst.size_in_byte(), true);
    }

    [[nodiscard]] BufferDownloadCommand *download(void *data, uint src_offset = 0,
                                                  bool async = true) const noexcept {
        return BufferDownloadCommand::create(data, head() + src_offset * element_size(),
                                             size_in_byte(), async);
    }

    [[nodiscard]] BufferUploadCommand *upload(const void *data, bool async = true) const noexcept {
        return BufferUploadCommand::create(data, head(), size_in_byte(), async);
    }

    [[nodiscard]] BufferByteSetCommand *byte_set(uchar value, bool async = true) const noexcept {
        return BufferByteSetCommand::create(head(), size_in_byte(), value, async);
    }

    [[nodiscard]] BufferUploadCommand *upload_sync(const void *data) const noexcept {
        return BufferUploadCommand::create(data, head(), size_in_byte(), false);
    }

    [[nodiscard]] BufferByteSetCommand *reset(bool async = true) const noexcept {
        return byte_set(0, async);
    }
};

class ByteBuffer : public RHIResource {
private:
    /// just for construct memory block
    mutable BufferProxy<> proxy_{};
    size_t size_{};

public:
    ByteBuffer() = default;
    ByteBuffer(Device::Impl *device, size_t size, const string &desc = "")
        : RHIResource(device, Tag::BUFFER, device->create_buffer(size, desc)),
          size_(size) {}

    ByteBuffer(ByteBufferView buffer_view)
        : RHIResource(nullptr, Tag::BUFFER, buffer_view.head()),
          size_(buffer_view.size()) {}

    /// head of the buffer
    [[nodiscard]] handle_ty head() const noexcept { return handle(); }
    [[nodiscard]] size_t size() const noexcept { return size_; }
    template<typename Size = size_t>
    [[nodiscard]] Size size_in_byte() const noexcept { return size(); }

    void destroy() override {
        _destroy();
        size_ = 0;
    }

    const BufferProxy<std::byte> &proxy() const noexcept {
        proxy_.handle = reinterpret_cast<std::byte *>(handle_);
        proxy_.size = size_;
        return proxy_;
    }

    const BufferProxy<std::byte> *proxy_ptr() const noexcept {
        return &proxy();
    }

    [[nodiscard]] size_t data_alignment() const noexcept override {
        return alignof(decltype(proxy_));
    }

    [[nodiscard]] size_t data_size() const noexcept override {
        return sizeof(proxy_);
    }

    [[nodiscard]] MemoryBlock memory_block() const noexcept override {
        return {proxy_ptr(), data_size(), data_alignment(), max_member_size()};
    }

    template<typename U = uint>
    [[nodiscard]] const Expression *expression() const noexcept {
        const CapturedResource &captured_resource = Function::current()->get_captured_resource(Type::of<decltype(*this)>(),
                                                                                               Variable::Tag::BYTE_BUFFER,
                                                                                               memory_block());
        return captured_resource.expression();
    }

    [[nodiscard]] ByteBufferView view(size_t offset = 0, size_t size = 0) const noexcept {
        size = size == 0 ? size_ - offset : size;
        return ByteBufferView(handle_, offset, size, size_);
    }

    template<typename T>
    [[nodiscard]] BufferView<T> view_as(size_t offset = 0, size_t size = 0) const noexcept {
        return view().view_as<T>(offset, size);
    }

    template<typename... Args>
    [[nodiscard]] BufferCopyCommand *copy_from(Args &&...args) const noexcept {
        return view(0, size_).copy_from(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferDownloadCommand *download(Args &&...args) const noexcept {
        return view(0, size_).download(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferUploadCommand *upload(Args &&...args) const noexcept {
        return view(0, size_).upload(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferByteSetCommand *byte_set(Args &&...args) const noexcept {
        return view(0, size_).byte_set(OC_FORWARD(args)...);
    }
    template<typename... Args>
    [[nodiscard]] BufferByteSetCommand *reset(Args &&...args) const noexcept {
        return view(0, size_).reset(OC_FORWARD(args)...);
    }

    /// for dsl start
    template<uint N = 1, typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load(Offset &&offset) const noexcept {
        auto expr = make_expr<ByteBuffer>(expression());
        return expr.template load<N, Elm>(OC_FORWARD(offset));
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load2(Offset &&offset) const noexcept {
        return load<2, Elm>(OC_FORWARD(offset));
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load3(Offset &&offset) const noexcept {
        return load<3, Elm>(OC_FORWARD(offset));
    }

    template<typename Elm = uint, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] auto load4(Offset &&offset) const noexcept {
        return load<4, Elm>(OC_FORWARD(offset));
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Var<Target> load_as(Offset &&offset) const noexcept {
        const auto expr = make_expr<ByteBuffer>(expression());
        return expr.template load_as<Target>(OC_FORWARD(offset));
    }

    template<typename Target, typename Offset>
    requires is_integral_expr_v<Offset>
    [[nodiscard]] Var<Target> &load_as(Offset &&offset) noexcept {
        auto expr = make_expr<ByteBuffer>(expression());
        return expr.template load_as<Target>(OC_FORWARD(offset));
    }

    template<typename Elm, typename Offset>
    requires is_integral_expr_v<Offset>
    void store(Offset &&offset, const Elm &val) noexcept {
        auto expr = make_expr<ByteBuffer>(expression());
        expr.store(OC_FORWARD(offset), val);
    }

    template<typename Elm>
    [[nodiscard]] SOAView<Elm, Expr<ByteBuffer>> soa_view() noexcept {
        auto e = make_expr<ByteBuffer>(expression());
        return SOAView<Elm, Expr<ByteBuffer>>(e);
    }

    template<typename Target = uint, typename Index>
    requires concepts::integral<expr_value_t<Index>>
    [[nodiscard]] detail::AtomicRef<Target> atomic(Index &&index) const noexcept {
        return make_expr<ByteBuffer>(expression()).atomic<Target>(OC_FORWARD(index));
    }
    /// for dsl end
};

ByteBufferView::ByteBufferView(const ocarina::ByteBuffer &buffer)
    : ByteBufferView(buffer.handle(), buffer.size()) {}

}// namespace ocarina