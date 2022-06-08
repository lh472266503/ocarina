//
// Created by Zero on 08/06/2022.
//

#include "stl.h"
#include "EASTL/allocator.h"

namespace ocarina::detail {
void *allocator_allocate(size_t size, size_t alignment) noexcept {
    return eastl::GetDefaultAllocator()->allocate(size, alignment, 0u);
}

void allocator_deallocate(void *p, size_t) noexcept {
    eastl::GetDefaultAllocator()->deallocate(p, 0u);
}

void *allocator_reallocate(void *p, size_t size, size_t alignment) noexcept {
    auto &&allocator = eastl::GetDefaultAllocator();
    allocator->deallocate(p, 0u);
    return allocator->allocate(size, alignment, 0u);
}
}// namespace ocarina