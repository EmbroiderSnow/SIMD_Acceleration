#pragma once
#include <cstdint>
#include "ImageTypes.h"

namespace Scalar {
    void YUV2RGB_ARGB8888(const YUVFrame& src, RGBFrame& dst, uint8_t alpha);
    void YUV2RGB_RGB888(const YUVFrame& src, RGBFrame& dst);
    void AlphaBlend(RGBFrame& img, uint8_t alpha);
    void ImageOverlay(const RGBFrame& src1, const RGBFrame& src2, RGBFrame& dst, uint8_t alpha);
    void RGB2YUV_ARGB8888(const RGBFrame& src, YUVFrame& dst);
    void RGB2YUV_RGB888(const RGBFrame& src, YUVFrame& dst); 
}

namespace MMX {
    void YUV2RGB_ARGB8888(const YUVFrame& src, RGBFrame& dst, uint8_t alpha);
    void YUV2RGB_RGB888(const YUVFrame& src, RGBFrame& dst);
    void AlphaBlend(RGBFrame& img, uint8_t alpha);
    void ImageOverlay(const RGBFrame& src1, const RGBFrame& src2, RGBFrame& dst, uint8_t alpha);
    void RGB2YUV_ARGB8888(const RGBFrame& src, YUVFrame& dst);
    void RGB2YUV_RGB888(const RGBFrame& src, YUVFrame& dst);
}

namespace SSE {
    void YUV2RGB_ARGB8888(const YUVFrame& src, RGBFrame& dst, uint8_t alpha);
    void YUV2RGB_RGB888(const YUVFrame& src, RGBFrame& dst);
    void AlphaBlend(RGBFrame& img, uint8_t alpha);
    void ImageOverlay(const RGBFrame& src1, const RGBFrame& src2, RGBFrame& dst, uint8_t alpha);
    void RGB2YUV_ARGB8888(const RGBFrame& src, YUVFrame& dst);
    void RGB2YUV_RGB888(const RGBFrame& src, YUVFrame& dst);
}

namespace AVX {
    void YUV2RGB_ARGB8888(const YUVFrame& src, RGBFrame& dst, uint8_t alpha);
    void YUV2RGB_RGB888(const YUVFrame& src, RGBFrame& dst);
    void AlphaBlend(RGBFrame& img, uint8_t alpha);
    void ImageOverlay(const RGBFrame& src1, const RGBFrame& src2, RGBFrame& dst, uint8_t alpha);
    void RGB2YUV_ARGB8888(const RGBFrame& src, YUVFrame& dst);
    void RGB2YUV_RGB888(const RGBFrame& src, YUVFrame& dst);
}