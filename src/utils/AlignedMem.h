#pragma once
#include <cstdlib>
#include <new>
#include <cstdint>

constexpr size_t ALIGNMENT = 32;

template <typename T>
struct AlignedAllocator
{
    static T *alloc(size_t count)
    {
        size_t size = count * sizeof(T);
        void *ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(size, ALIGNMENT);
#else
        if (posix_memalign(&ptr, ALIGNMENT, size) != 0)
            ptr = nullptr;
#endif
        if (!ptr)
            throw std::bad_alloc();
        return static_cast<T *>(ptr);
    }

    static void free(T *ptr)
    {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
};