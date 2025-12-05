#pragma once
#include <vector>
#include <cstdint>
#include "../utils/AlignedMem.h"

enum class RGBFormat
{
    RGB888 = 3,  // 3 channels
    ARGB8888 = 4  // 4 channels
};

struct YUVFrame
{
    int width;
    int height;
    uint8_t *Y;
    uint8_t *U;
    uint8_t *V;

    YUVFrame(int w, int h) : width(w), height(h)
    {
        int size = w * h;
        Y = AlignedAllocator<uint8_t>::alloc(size);
        U = AlignedAllocator<uint8_t>::alloc(size / 4);
        V = AlignedAllocator<uint8_t>::alloc(size / 4);
    }

    ~YUVFrame()
    {
        AlignedAllocator<uint8_t>::free(Y);
        AlignedAllocator<uint8_t>::free(U);
        AlignedAllocator<uint8_t>::free(V);
    }
};

struct RGBFrame
{
    int width;
    int height;
    int channels; // 3 or 4, meaning RGB or ARGB
    uint8_t *data;

    RGBFrame(int w, int h, RGBFormat format) : width(w), height(h)
    {
        channels = static_cast<int>(format);
        data = AlignedAllocator<uint8_t>::alloc(w * h * channels + 16);
    }

    ~RGBFrame()
    {
        AlignedAllocator<uint8_t>::free(data);
    }
};