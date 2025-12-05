#include "Converter.h"
#include <algorithm>
#include <cstdint>

inline uint8_t clamp(int v)
{
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

namespace Scalar
{
    // YUV420 -> ARGB8888 (Scalar)
    // Use BT.601 standard conversion with 8 bit shift optimizations
    void YUV2RGB_ARGB8888(const YUVFrame &src, RGBFrame &dst, uint8_t alpha)
    {
        int width = src.width;
        int height = src.height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int yIndex = y * width + x;
                int uvIndex = (y / 2) * (width / 2) + (x / 2);

                int Y = src.Y[yIndex];
                int U = src.U[uvIndex] - 128;
                int V = src.V[uvIndex] - 128;

                int R = Y + ((359 * V) >> 8);
                int G = Y - ((88 * U + 183 * V) >> 8);
                int B = Y + ((454 * U) >> 8);

                int pixelIndex = (y * width + x) * 4;

                dst.data[pixelIndex + 0] = clamp(B);
                dst.data[pixelIndex + 1] = clamp(G);
                dst.data[pixelIndex + 2] = clamp(R);
                dst.data[pixelIndex + 3] = alpha;
            }
        }
    }

    // YUV420 -> RGB888 (Scalar)
    // Use BT.601 standard conversion with 8 bit shift optimizations
    void YUV2RGB_RGB888(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int yIndex = y * width + x;
                int uvIndex = (y / 2) * (width / 2) + (x / 2);

                int Y = src.Y[yIndex];
                int U = src.U[uvIndex] - 128;
                int V = src.V[uvIndex] - 128;

                int R = Y + ((359 * V) >> 8);
                int G = Y - ((88 * U + 183 * V) >> 8);
                int B = Y + ((454 * U) >> 8);

                int pixelIndex = (y * width + x) * 3;

                dst.data[pixelIndex + 0] = clamp(B);
                dst.data[pixelIndex + 1] = clamp(G);
                dst.data[pixelIndex + 2] = clamp(R);
            }
        }
    }

    // Part 2: Alpha Blending (Scalar)
    // Format: New = (Alpha * RGB) / 256
    // Maybe we don't care about 256 or 255 difference for simplicity
    void AlphaBlend(RGBFrame &img, uint8_t alpha)
    {
        int numPixels = img.width * img.height;

        for (int i = 0; i < numPixels; ++i)
        {
            int baseIdx = i * 4;

            img.data[baseIdx + 0] = (img.data[baseIdx + 0] * alpha) >> 8; // B
            img.data[baseIdx + 1] = (img.data[baseIdx + 1] * alpha) >> 8; // G
            img.data[baseIdx + 2] = (img.data[baseIdx + 2] * alpha) >> 8; // R
        }
    }

    // Part 3: RGB888 -> YUV420 (Scalar)
    void RGB2YUV_RGB888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIdx = (y * width + x) * 3; 

                int B00 = src.data[pixelIdx + 0];
                int G00 = src.data[pixelIdx + 1];
                int R00 = src.data[pixelIdx + 2];
                
                int Y = ((77 * R00 + 150 * G00 + 29 * B00) >> 8);
                dst.Y[y * width + x] = clamp(Y);

                if (y % 2 == 0 && x % 2 == 0)
                {   
                    int B01 = src.data[pixelIdx + 3 + 0]; 
                    int G01 = src.data[pixelIdx + 3 + 1];
                    int R01 = src.data[pixelIdx + 3 + 2];

                    int baseRow2 = pixelIdx + width * 3;
                    int B10 = src.data[baseRow2 + 0];
                    int G10 = src.data[baseRow2 + 1];
                    int R10 = src.data[baseRow2 + 2];

                    int B11 = src.data[baseRow2 + 3 + 0];
                    int G11 = src.data[baseRow2 + 3 + 1];
                    int R11 = src.data[baseRow2 + 3 + 2];

                    int R = (R00 + R01 + R10 + R11) >> 2;
                    int G = (G00 + G01 + G10 + G11) >> 2;
                    int B = (B00 + B01 + B10 + B11) >> 2;

                    int U = ((-43 * R - 84 * G + 127 * B) >> 8) + 128;
                    int V = ((127 * R - 107 * G - 20 * B) >> 8) + 128;

                    int uvIndex = (y / 2) * (width / 2) + (x / 2);
                    dst.U[uvIndex] = clamp(U);
                    dst.V[uvIndex] = clamp(V);
                }
            }
        }
    }

    // Part 3: ARGB8888 -> YUV420 (Scalar)
    void RGB2YUV_ARGB8888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int pixelIdx = (y * width + x) * 4;

                int B00 = src.data[pixelIdx + 0];
                int G00 = src.data[pixelIdx + 1];
                int R00 = src.data[pixelIdx + 2];

                int Y = ((77 * R00 + 150 * G00 + 29 * B00) >> 8);
                dst.Y[y * width + x] = clamp(Y);

                if (y % 2 == 0 && x % 2 == 0)
                {
                    int B01 = src.data[pixelIdx + 4 + 0];
                    int G01 = src.data[pixelIdx + 4 + 1];
                    int R01 = src.data[pixelIdx + 4 + 2];

                    int B10 = src.data[pixelIdx + width * 4 + 0];
                    int G10 = src.data[pixelIdx + width * 4 + 1];
                    int R10 = src.data[pixelIdx + width * 4 + 2];

                    int B11 = src.data[pixelIdx + width * 4 + 4 + 0];
                    int G11 = src.data[pixelIdx + width * 4 + 4 + 1];
                    int R11 = src.data[pixelIdx + width * 4 + 4 + 2];

                    int R = (R00 + R01 + R10 + R11) >> 2;
                    int G = (G00 + G01 + G10 + G11) >> 2;
                    int B = (B00 + B01 + B10 + B11) >> 2;

                    // U = -0.1687R - 0.3313G + 0.5B + 128
                    // V = 0.5R - 0.4187G - 0.0813B + 128
                    int U = ((-43 * R - 84 * G + 127 * B) >> 8) + 128;
                    int V = ((127 * R - 107 * G - 20 * B) >> 8) + 128;

                    int uvIndex = (y / 2) * (width / 2) + (x / 2);
                    dst.U[uvIndex] = clamp(U);
                    dst.V[uvIndex] = clamp(V);
                }
            }
        }
    }

    void ImageOverlay(const RGBFrame& src1, const RGBFrame& src2, RGBFrame& dst, uint8_t alpha)
    {
        int width = src1.width;
        int height = src1.height;
        
        int channels = src1.channels; 
        int totalBytes = width * height * channels;

        int w1 = 256 - alpha;
        int w2 = alpha;

        for (int i = 0; i < totalBytes; ++i)
        {
            int val = (src1.data[i] * w1 + src2.data[i] * w2) >> 8;
            
            dst.data[i] = static_cast<uint8_t>(val);
        }
    }

    // ---------------------------------------------------------
    // MemOnly: 模拟逐字节写入 RGB888 的内存压力
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_MemOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        uint8_t *dstPtr = dst.data;
        const uint8_t *YPtr = src.Y;

        for (int y = 0; y < height; y++)
        {
            // 强制读取 UV (防止由于只读 Y 导致的 Cache 差异)
            volatile uint8_t u_dummy = src.U[(y / 2) * (width / 2)];
            volatile uint8_t v_dummy = src.V[(y / 2) * (width / 2)];
            (void)u_dummy; (void)v_dummy;

            for (int x = 0; x < width; x++)
            {
                // 读取 Y
                uint8_t Y = YPtr[y * width + x];
                
                // 模拟写入: 直接写 3 个字节
                // 我们直接复制 Y 值来模拟数据流
                int pixelIndex = (y * width + x) * 3;
                dstPtr[pixelIndex + 0] = Y;
                dstPtr[pixelIndex + 1] = Y;
                dstPtr[pixelIndex + 2] = Y;
            }
        }
    }

    // ---------------------------------------------------------
    // ComputeOnly: 执行所有数学运算，但防止写入内存
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_ComputeOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        // 累加器，防止被优化
        volatile int accum = 0;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int yIndex = y * width + x;
                int uvIndex = (y / 2) * (width / 2) + (x / 2);

                int Y = src.Y[yIndex];
                int U = src.U[uvIndex] - 128;
                int V = src.V[uvIndex] - 128;

                // 完整的数学运算
                int R = Y + ((359 * V) >> 8);
                int G = Y - ((88 * U + 183 * V) >> 8);
                int B = Y + ((454 * U) >> 8);

                // 虚假使用结果
                accum += (R ^ G ^ B);
            }
        }
    }

    // =========================================================
    // ShuffleOnly: 仅保留数据读取、依赖传递和写入
    // =========================================================
    void YUV2RGB_RGB888_ShuffleOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int yIndex = y * width + x;
                int uvIndex = (y / 2) * (width / 2) + (x / 2);

                int Y = src.Y[yIndex];
                int U = src.U[uvIndex]; // 不减 128，纯搬运
                int V = src.V[uvIndex];

                // 模拟数据依赖，但只做极低开销的位运算
                // 目的是防止编译器发现变量未被修改而优化掉读取
                int R = Y ^ V;
                int G = Y ^ U ^ V;
                int B = Y ^ U;

                int pixelIndex = (y * width + x) * 3;

                // 核心 Layout 开销：逐字节非对齐写入
                dst.data[pixelIndex + 0] = static_cast<uint8_t>(B);
                dst.data[pixelIndex + 1] = static_cast<uint8_t>(G);
                dst.data[pixelIndex + 2] = static_cast<uint8_t>(R);
            }
        }
    }
}