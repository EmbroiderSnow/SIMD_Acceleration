#include "Converter.h"
#include <mmintrin.h>
#include <algorithm>
#include <cstdint>

inline uint8_t clamp(int v)
{
    return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

namespace MMX
{
    // =========================================================
    // Part 2: Alpha Blending (MMX)
    // Formula: New = (Alpha * RGB) / 256
    // =========================================================
    void AlphaBlend(RGBFrame &img, uint8_t alpha)
    {
        int totalBytes = img.width * img.height * 4;
        uint8_t *data = img.data;

        // Replicate alpha into 4x16-bit words: [0, A, 0, A, 0, A, 0, A] (conceptually)
        __m64 alpha_vec = _mm_set1_pi16((short)alpha);
        __m64 zero = _mm_setzero_si64();

        int i = 0;
        // Process 8 bytes (2 ARGB pixels) per iteration
        for (; i <= totalBytes - 8; i += 8)
        {
            __m64 src = *(__m64 *)(data + i);

            // Unpack 8-bit to 16-bit to allow multiplication
            __m64 src_lo = _mm_unpacklo_pi8(src, zero);
            __m64 src_hi = _mm_unpackhi_pi8(src, zero);

            // Multiply: pixel * alpha
            src_lo = _mm_mullo_pi16(src_lo, alpha_vec);
            src_hi = _mm_mullo_pi16(src_hi, alpha_vec);

            // Divide by 256 (Bit shift right 8)
            src_lo = _mm_srli_pi16(src_lo, 8);
            src_hi = _mm_srli_pi16(src_hi, 8);

            // Pack 16-bit back to 8-bit with saturation
            __m64 result = _mm_packs_pu16(src_lo, src_hi);

            // Store result
            *(__m64 *)(data + i) = result;
        }

        // Handle remaining bytes (Scalar fallback)
        for (; i < totalBytes; ++i)
        {
            data[i] = (data[i] * alpha) >> 8;
        }

        // Clear MMX state
        _mm_empty();
    }

    // =========================================================
    // Part 3: Image Overlay (MMX)
    // Formula: Dst = (Src1 * (256-a) + Src2 * a) >> 8
    // =========================================================
    void ImageOverlay(const RGBFrame &src1, const RGBFrame &src2, RGBFrame &dst, uint8_t alpha)
    {
        int totalBytes = src1.width * src1.height * 3; // RGB888

        const uint8_t *pSrc1 = src1.data;
        const uint8_t *pSrc2 = src2.data;
        uint8_t *pDst = dst.data;

        short w1_val = 256 - alpha;
        short w2_val = alpha;

        __m64 w1_vec = _mm_set1_pi16(w1_val);
        __m64 w2_vec = _mm_set1_pi16(w2_val);
        __m64 zero = _mm_setzero_si64();

        int i = 0;
        // Process 8 bytes per iteration
        for (; i <= totalBytes - 8; i += 8)
        {
            // 1. Load data
            __m64 s1 = *(__m64 *)(pSrc1 + i);
            __m64 s2 = *(__m64 *)(pSrc2 + i);

            // 2. Unpack 8-bit to 16-bit
            __m64 s1_lo = _mm_unpacklo_pi8(s1, zero);
            __m64 s1_hi = _mm_unpackhi_pi8(s1, zero);

            __m64 s2_lo = _mm_unpacklo_pi8(s2, zero);
            __m64 s2_hi = _mm_unpackhi_pi8(s2, zero);

            // 3. Weighted Sum
            // Part 1: Src1 * (256 - Alpha)
            s1_lo = _mm_mullo_pi16(s1_lo, w1_vec);
            s1_hi = _mm_mullo_pi16(s1_hi, w1_vec);

            // Part 2: Src2 * Alpha
            s2_lo = _mm_mullo_pi16(s2_lo, w2_vec);
            s2_hi = _mm_mullo_pi16(s2_hi, w2_vec);

            // Sum parts
            __m64 res_lo = _mm_add_pi16(s1_lo, s2_lo);
            __m64 res_hi = _mm_add_pi16(s1_hi, s2_hi);

            // 4. Normalize (Shift right 8)
            res_lo = _mm_srli_pi16(res_lo, 8);
            res_hi = _mm_srli_pi16(res_hi, 8);

            // 5. Pack and Store
            __m64 result = _mm_packs_pu16(res_lo, res_hi);
            *(__m64 *)(pDst + i) = result;
        }

        // Handle remaining bytes (Scalar fallback)
        for (; i < totalBytes; ++i)
        {
            int val = (pSrc1[i] * w1_val + pSrc2[i] * w2_val) >> 8;
            pDst[i] = (uint8_t)val;
        }

        _mm_empty();
    }

    // Helper: Expand 8-bit to 16-bit and subtract 128 (for UV channels)
    static inline __m64 expand_and_sub128(__m64 data_u8, __m64 zero, bool high)
    {
        __m64 data_u16;
        if (high)
        {
            data_u16 = _mm_unpackhi_pi8(data_u8, zero);
        }
        else
        {
            data_u16 = _mm_unpacklo_pi8(data_u8, zero);
        }
        return _mm_sub_pi16(data_u16, _mm_set1_pi16(128));
    }

    // =========================================================
    // YUV420 -> ARGB8888 (MMX)
    // =========================================================
    void YUV2RGB_ARGB8888(const YUVFrame &src, RGBFrame &dst, uint8_t alpha)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // Coefficients scaled by 64 (bit shift 6) to prevent 16-bit multiplication overflow
        // R: 1.402 * 64 ~ 90
        // G: 0.344 * 64 ~ 22, 0.714 * 64 ~ 46
        // B: 1.772 * 64 ~ 113
        __m64 c90 = _mm_set1_pi16(90);
        __m64 c22 = _mm_set1_pi16(22);
        __m64 c46 = _mm_set1_pi16(46);
        __m64 c113 = _mm_set1_pi16(113);

        __m64 zero = _mm_setzero_si64();
        __m64 alphaVec = _mm_set1_pi8(alpha);

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 8)
            {
                // Load Y (8 pixels)
                __m64 y_raw = *(__m64 *)(YPtr + y * width + x);

                // Load U, V (subsampled, need upsampling)
                int u_val = *(int *)(UPtr + uvOffset + (x / 2));
                int v_val = *(int *)(VPtr + uvOffset + (x / 2));
                __m64 u_raw_4 = _mm_cvtsi32_si64(u_val);
                __m64 v_raw_4 = _mm_cvtsi32_si64(v_val);

                // Duplicate U/V to match Y resolution
                __m64 u_raw = _mm_unpacklo_pi8(u_raw_4, u_raw_4);
                __m64 v_raw = _mm_unpacklo_pi8(v_raw_4, v_raw_4);

                // --- Process Low 4 pixels ---
                __m64 y_lo = _mm_unpacklo_pi8(y_raw, zero);
                __m64 u_lo = expand_and_sub128(u_raw, zero, false);
                __m64 v_lo = expand_and_sub128(v_raw, zero, false);

                // Convert YUV to RGB (using Arithmetic Shift Right to preserve sign)
                // R = Y + 1.402 * V
                __m64 r_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c90, v_lo), 6));

                // G = Y - 0.344 * U - 0.714 * V
                __m64 g_part_lo = _mm_add_pi16(_mm_mullo_pi16(c22, u_lo), _mm_mullo_pi16(c46, v_lo));
                __m64 g_lo = _mm_sub_pi16(y_lo, _mm_srai_pi16(g_part_lo, 6));

                // B = Y + 1.772 * U
                __m64 b_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c113, u_lo), 6));

                // --- Process High 4 pixels ---
                __m64 y_hi = _mm_unpackhi_pi8(y_raw, zero);
                __m64 u_hi = expand_and_sub128(u_raw, zero, true);
                __m64 v_hi = expand_and_sub128(v_raw, zero, true);

                __m64 r_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c90, v_hi), 6));

                __m64 g_part_hi = _mm_add_pi16(_mm_mullo_pi16(c22, u_hi), _mm_mullo_pi16(c46, v_hi));
                __m64 g_hi = _mm_sub_pi16(y_hi, _mm_srai_pi16(g_part_hi, 6));

                __m64 b_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c113, u_hi), 6));

                // Pack 16-bit results back to 8-bit
                __m64 R = _mm_packs_pu16(r_lo, r_hi);
                __m64 G = _mm_packs_pu16(g_lo, g_hi);
                __m64 B = _mm_packs_pu16(b_lo, b_hi);
                __m64 A = alphaVec;

                // Interleave channels to form BGRA format
                __m64 BG_lo = _mm_unpacklo_pi8(B, G);
                __m64 BG_hi = _mm_unpackhi_pi8(B, G);
                __m64 RA_lo = _mm_unpacklo_pi8(R, A);
                __m64 RA_hi = _mm_unpackhi_pi8(R, A);

                __m64 BGRA_0 = _mm_unpacklo_pi16(BG_lo, RA_lo);
                __m64 BGRA_1 = _mm_unpackhi_pi16(BG_lo, RA_lo);
                __m64 BGRA_2 = _mm_unpacklo_pi16(BG_hi, RA_hi);
                __m64 BGRA_3 = _mm_unpackhi_pi16(BG_hi, RA_hi);

                // Store to destination
                __m64 *dstPtr64 = (__m64 *)(dstPtr + (y * width + x) * 4);
                dstPtr64[0] = BGRA_0;
                dstPtr64[1] = BGRA_1;
                dstPtr64[2] = BGRA_2;
                dstPtr64[3] = BGRA_3;
            }
        }
        _mm_empty();
    }

    // =========================================================
    // YUV420 -> RGB888 (MMX)
    // =========================================================
    void YUV2RGB_RGB888(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // Coefficients scaled by 64 (bit shift 6)
        __m64 c90 = _mm_set1_pi16(90);
        __m64 c22 = _mm_set1_pi16(22);
        __m64 c46 = _mm_set1_pi16(46);
        __m64 c113 = _mm_set1_pi16(113);

        __m64 zero = _mm_setzero_si64();
        // Alpha set to 255 for temporary BGRA structure
        __m64 alphaVec = _mm_set1_pi8(255);

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 8)
            {
                // Load Y
                __m64 y_raw = *(__m64 *)(YPtr + y * width + x);

                // Load U, V
                int u_val = *(int *)(UPtr + uvOffset + (x / 2));
                int v_val = *(int *)(VPtr + uvOffset + (x / 2));
                __m64 u_raw_4 = _mm_cvtsi32_si64(u_val);
                __m64 v_raw_4 = _mm_cvtsi32_si64(v_val);

                // Upsample
                __m64 u_raw = _mm_unpacklo_pi8(u_raw_4, u_raw_4);
                __m64 v_raw = _mm_unpacklo_pi8(v_raw_4, v_raw_4);

                // --- Process Low 4 pixels ---
                __m64 y_lo = _mm_unpacklo_pi8(y_raw, zero);
                __m64 u_lo = expand_and_sub128(u_raw, zero, false);
                __m64 v_lo = expand_and_sub128(v_raw, zero, false);

                // YUV to RGB Conversion (Shift 6)
                __m64 r_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c90, v_lo), 6));

                __m64 g_part_lo = _mm_add_pi16(_mm_mullo_pi16(c22, u_lo), _mm_mullo_pi16(c46, v_lo));
                __m64 g_lo = _mm_sub_pi16(y_lo, _mm_srai_pi16(g_part_lo, 6));

                __m64 b_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c113, u_lo), 6));

                // --- Process High 4 pixels ---
                __m64 y_hi = _mm_unpackhi_pi8(y_raw, zero);
                __m64 u_hi = expand_and_sub128(u_raw, zero, true);
                __m64 v_hi = expand_and_sub128(v_raw, zero, true);

                __m64 r_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c90, v_hi), 6));

                __m64 g_part_hi = _mm_add_pi16(_mm_mullo_pi16(c22, u_hi), _mm_mullo_pi16(c46, v_hi));
                __m64 g_hi = _mm_sub_pi16(y_hi, _mm_srai_pi16(g_part_hi, 6));

                __m64 b_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c113, u_hi), 6));

                // Pack
                __m64 R = _mm_packs_pu16(r_lo, r_hi);
                __m64 G = _mm_packs_pu16(g_lo, g_hi);
                __m64 B = _mm_packs_pu16(b_lo, b_hi);
                __m64 A = alphaVec;

                // Interleave to BGRA (temporary)
                __m64 BG_lo = _mm_unpacklo_pi8(B, G);
                __m64 BG_hi = _mm_unpackhi_pi8(B, G);
                __m64 RA_lo = _mm_unpacklo_pi8(R, A);
                __m64 RA_hi = _mm_unpackhi_pi8(R, A);

                __m64 BGRA_0 = _mm_unpacklo_pi16(BG_lo, RA_lo);
                __m64 BGRA_1 = _mm_unpackhi_pi16(BG_lo, RA_lo);
                __m64 BGRA_2 = _mm_unpacklo_pi16(BG_hi, RA_hi);
                __m64 BGRA_3 = _mm_unpackhi_pi16(BG_hi, RA_hi);

                // Store 24-bit RGB data using overlapping 32-bit writes
                // Effectively drops the Alpha byte
                uint8_t *ptr = dstPtr + (y * width + x) * 3;

                int pixels[8];
                pixels[0] = _mm_cvtsi64_si32(BGRA_0);
                pixels[1] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_0, BGRA_0));
                pixels[2] = _mm_cvtsi64_si32(BGRA_1);
                pixels[3] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_1, BGRA_1));
                pixels[4] = _mm_cvtsi64_si32(BGRA_2);
                pixels[5] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_2, BGRA_2));
                pixels[6] = _mm_cvtsi64_si32(BGRA_3);
                pixels[7] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_3, BGRA_3));

                for (int k = 0; k < 8; ++k)
                {
                    *(int *)(ptr + k * 3) = pixels[k];
                }
            }
        }
        _mm_empty();
    }

    // Helper: Horizontal add of 32-bit integers in MMX register
    static inline int32_t hadd_and_extract_mmx(__m64 reg)
    {
        __m64 high = _mm_srli_si64(reg, 32);
        __m64 sum = _mm_add_pi32(reg, high);
        return _mm_cvtsi64_si32(sum);
    }

    // =========================================================
    // ARGB8888 -> YUV420 (MMX)
    // Uses 2x2 average sampling for U/V
    // =========================================================
    void RGB2YUV_ARGB8888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        // Coefficients for Y (Scale 256)
        // Y = 0.299R + 0.587G + 0.114B
        __m64 coeff_Y_RB = _mm_setr_pi16(29, 77, 29, 77); // [B, R, B, R]
        __m64 coeff_Y_GA = _mm_setr_pi16(150, 0, 150, 0); // [G, 0, G, 0]

        // Coefficients for U (Scale 256)
        // U = -0.1687R - 0.3313G + 0.5B + 128
        __m64 coeff_U_RB = _mm_setr_pi16(127, -43, 127, -43);
        __m64 coeff_U_GA = _mm_setr_pi16(-84, 0, -84, 0);

        // Coefficients for V (Scale 256)
        // V = 0.5R - 0.4187G - 0.0813B + 128
        __m64 coeff_V_RB = _mm_setr_pi16(-20, 127, -20, 127);
        __m64 coeff_V_GA = _mm_setr_pi16(-107, 0, -107, 0);

        __m64 zero = _mm_setzero_si64();

        // Process 2 rows at a time
        for (int y = 0; y < height; y += 2)
        {
            // Process 8 columns at a time
            for (int x = 0; x < width; x += 8)
            {
                // --- 1. Compute Y (for Row y and Row y+1) ---
                for (int r = 0; r < 2; ++r)
                {
                    int currY = y + r;
                    const uint8_t *pSrc = src.data + (currY * width + x) * 4;
                    uint8_t *pDstY = dst.Y + currY * width + x;

                    uint8_t y_buffer[8];

                    for (int k = 0; k < 4; ++k) // Process 2 pixels per iteration
                    {
                        // Load 2 pixels: [B0 G0 R0 A0 | B1 G1 R1 A1]
                        __m64 px = *(__m64 *)(pSrc + k * 8);

                        // Unpack to 16-bit
                        __m64 p0 = _mm_unpacklo_pi8(px, zero);
                        __m64 p1 = _mm_unpackhi_pi8(px, zero);

                        // Coefficients re-arranged for PMADD: [B*29, G*150, R*77, A*0]
                        __m64 cY = _mm_setr_pi16(29, 150, 77, 0);

                        // Multiply and pairwise add
                        __m64 res0 = _mm_madd_pi16(p0, cY);
                        __m64 res1 = _mm_madd_pi16(p1, cY);

                        // Horizontal Add to get final scalar sums
                        int y0_val = hadd_and_extract_mmx(res0) >> 8;
                        int y1_val = hadd_and_extract_mmx(res1) >> 8;

                        y_buffer[2 * k] = clamp(y0_val);
                        y_buffer[2 * k + 1] = clamp(y1_val);
                    }

                    *(__m64 *)pDstY = *(__m64 *)y_buffer;
                }

                // --- 2. Compute U/V (2x2 Average Sampling) ---
                const uint8_t *row0 = src.data + (y * width + x) * 4;
                const uint8_t *row1 = src.data + ((y + 1) * width + x) * 4;

                uint8_t u_buffer[4];
                uint8_t v_buffer[4];

                // Process 4 blocks of 2x2 pixels
                for (int k = 0; k < 4; ++k)
                {
                    // Load 2x2 block
                    __m64 r0_px = *(__m64 *)(row0 + k * 8); // Top row pair
                    __m64 r1_px = *(__m64 *)(row1 + k * 8); // Bottom row pair

                    __m64 p0 = _mm_unpacklo_pi8(r0_px, zero);
                    __m64 p1 = _mm_unpackhi_pi8(r0_px, zero);
                    __m64 p2 = _mm_unpacklo_pi8(r1_px, zero);
                    __m64 p3 = _mm_unpackhi_pi8(r1_px, zero);

                    // Average: (P0 + P1 + P2 + P3) / 4
                    __m64 sum = _mm_add_pi16(_mm_add_pi16(p0, p1), _mm_add_pi16(p2, p3));
                    __m64 avg = _mm_srli_pi16(sum, 2);

                    // Calculate U
                    // Coeff = [127, -84, -43, 0] (B, G, R, A)
                    __m64 cU = _mm_setr_pi16(127, -84, -43, 0);
                    __m64 u_parts = _mm_madd_pi16(avg, cU);
                    int u_val = (hadd_and_extract_mmx(u_parts) >> 8) + 128;
                    u_buffer[k] = clamp(u_val);

                    // Calculate V
                    // Coeff = [-20, -107, 127, 0] (B, G, R, A)
                    __m64 cV = _mm_setr_pi16(-20, -107, 127, 0);
                    __m64 v_parts = _mm_madd_pi16(avg, cV);
                    int v_val = (hadd_and_extract_mmx(v_parts) >> 8) + 128;
                    v_buffer[k] = clamp(v_val);
                }

                // Store subsampled U/V
                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                *(int *)(dst.U + uvIdx) = *(int *)u_buffer;
                *(int *)(dst.V + uvIdx) = *(int *)v_buffer;
            }
        }
        _mm_empty();
    }

    // =========================================================
    // RGB888 -> YUV420 (MMX)
    // Input format: BGR (3 bytes/pixel)
    // Uses 2x2 average sampling for U/V
    // =========================================================
    void RGB2YUV_RGB888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        // Coefficients Scaled by 64 (Bit shift 6) to fit MMX 16-bit limits
        // Y: 7*B + 38*G + 19*R
        __m64 coeff_Y = _mm_setr_pi16(7, 38, 19, 0);

        // U: 32*B - 21*G - 11*R
        __m64 coeff_U = _mm_setr_pi16(32, -21, -11, 0);

        // V: -5*B - 27*G + 32*R
        __m64 coeff_V = _mm_setr_pi16(-5, -27, 32, 0);

        __m64 zero = _mm_setzero_si64();

        for (int y = 0; y < height; y += 2)
        {
            // Process 2 columns (one 2x2 UV block) per iteration
            for (int x = 0; x < width; x += 2)
            {
                const uint8_t *r0 = src.data + (y * width + x) * 3;
                const uint8_t *r1 = src.data + ((y + 1) * width + x) * 3;

                // Mask high bits to isolate 24-bit pixel data from 32-bit read
                int p0_val = *(int *)(r0) & 0x00FFFFFF;
                int p1_val = *(int *)(r0 + 3) & 0x00FFFFFF;
                int p2_val = *(int *)(r1) & 0x00FFFFFF;
                int p3_val = *(int *)(r1 + 3) & 0x00FFFFFF;

                __m64 px0 = _mm_cvtsi32_si64(p0_val);
                __m64 px1 = _mm_cvtsi32_si64(p1_val);
                __m64 px2 = _mm_cvtsi32_si64(p2_val);
                __m64 px3 = _mm_cvtsi32_si64(p3_val);

                // Unpack to 16-bit: [B, G, R, 0]
                __m64 p0_16 = _mm_unpacklo_pi8(px0, zero);
                __m64 p1_16 = _mm_unpacklo_pi8(px1, zero);
                __m64 p2_16 = _mm_unpacklo_pi8(px2, zero);
                __m64 p3_16 = _mm_unpacklo_pi8(px3, zero);

                // --- Calculate Y (shift 6) ---
                int y0 = hadd_and_extract_mmx(_mm_madd_pi16(p0_16, coeff_Y)) >> 6;
                int y1 = hadd_and_extract_mmx(_mm_madd_pi16(p1_16, coeff_Y)) >> 6;
                int y2 = hadd_and_extract_mmx(_mm_madd_pi16(p2_16, coeff_Y)) >> 6;
                int y3 = hadd_and_extract_mmx(_mm_madd_pi16(p3_16, coeff_Y)) >> 6;

                dst.Y[y * width + x] = clamp(y0);
                dst.Y[y * width + x + 1] = clamp(y1);
                dst.Y[(y + 1) * width + x] = clamp(y2);
                dst.Y[(y + 1) * width + x + 1] = clamp(y3);

                // --- Calculate U/V (2x2 Average) ---
                __m64 sum = _mm_add_pi16(_mm_add_pi16(p0_16, p1_16), _mm_add_pi16(p2_16, p3_16));
                __m64 avg = _mm_srli_pi16(sum, 2);

                // U/V calculation (shift 6) + 128 offset
                int u_val = (hadd_and_extract_mmx(_mm_madd_pi16(avg, coeff_U)) >> 6) + 128;
                int v_val = (hadd_and_extract_mmx(_mm_madd_pi16(avg, coeff_V)) >> 6) + 128;

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                dst.U[uvIdx] = clamp(u_val);
                dst.V[uvIdx] = clamp(v_val);
            }
        }
        _mm_empty();
    }

    // ---------------------------------------------------------
    // MemOnly: 保留重叠写入逻辑 (Overlapping Store)
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_MemOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        uint8_t *dstPtr = dst.data;
        const uint8_t *YPtr = src.Y;

        for (int y = 0; y < height; y++)
        {
            // 强制读取 UV
            int uvOffset = (y / 2) * (width / 2);
            volatile int u_dummy = *(int*)(src.U + uvOffset);
            volatile int v_dummy = *(int*)(src.V + uvOffset);
            (void)u_dummy; (void)v_dummy;

            for (int x = 0; x < width; x += 8)
            {
                // 1. Load (8 pixels)
                __m64 y_raw = *(__m64 *)(YPtr + y * width + x);
                
                // 2. 构造 Dummy 数据 (模拟 BGRA 结构)
                // 直接使用 y_raw 填充，不进行计算
                __m64 BGRA_dummy = y_raw; 

                // 3. Store (完全复刻原版的重叠写入逻辑)
                uint8_t *ptr = dstPtr + (y * width + x) * 3;
                int pixels[8];
                
                // 模拟解包后的数据分布
                int val = _mm_cvtsi64_si32(BGRA_dummy);
                for(int k=0; k<8; ++k) pixels[k] = val;

                // 关键瓶颈点：8 次非对齐的 4 字节写入
                for (int k = 0; k < 8; ++k)
                {
                    *(int *)(ptr + k * 3) = pixels[k];
                }
            }
        }
        _mm_empty();
    }

    // ---------------------------------------------------------
    // ComputeOnly: 保留繁琐的 Unpack 和 pmadd
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_ComputeOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;

        __m64 c90 = _mm_set1_pi16(90);
        __m64 c22 = _mm_set1_pi16(22);
        __m64 c46 = _mm_set1_pi16(46);
        __m64 c113 = _mm_set1_pi16(113);
        __m64 zero = _mm_setzero_si64();
        __m64 alphaVec = _mm_set1_pi8(255);
        
        // 累加器
        __m64 accum = _mm_setzero_si64();

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            for (int x = 0; x < width; x += 8)
            {
                // Load & Upsample Logic... [保持原版逻辑]
                __m64 y_raw = *(__m64 *)(YPtr + y * width + x);
                int u_val = *(int *)(UPtr + uvOffset + (x / 2));
                int v_val = *(int *)(VPtr + uvOffset + (x / 2));
                __m64 u_raw_4 = _mm_cvtsi32_si64(u_val);
                __m64 v_raw_4 = _mm_cvtsi32_si64(v_val);
                __m64 u_raw = _mm_unpacklo_pi8(u_raw_4, u_raw_4);
                __m64 v_raw = _mm_unpacklo_pi8(v_raw_4, v_raw_4);

                // ... Process Low 4 pixels ...
                __m64 y_lo = _mm_unpacklo_pi8(y_raw, zero);
                __m64 u_lo = expand_and_sub128(u_raw, zero, false);
                __m64 v_lo = expand_and_sub128(v_raw, zero, false);

                __m64 r_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c90, v_lo), 6));
                __m64 g_part_lo = _mm_add_pi16(_mm_mullo_pi16(c22, u_lo), _mm_mullo_pi16(c46, v_lo));
                __m64 g_lo = _mm_sub_pi16(y_lo, _mm_srai_pi16(g_part_lo, 6));
                __m64 b_lo = _mm_add_pi16(y_lo, _mm_srai_pi16(_mm_mullo_pi16(c113, u_lo), 6));

                // ... Process High 4 pixels ...
                __m64 y_hi = _mm_unpackhi_pi8(y_raw, zero);
                __m64 u_hi = expand_and_sub128(u_raw, zero, true);
                __m64 v_hi = expand_and_sub128(v_raw, zero, true);

                __m64 r_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c90, v_hi), 6));
                __m64 g_part_hi = _mm_add_pi16(_mm_mullo_pi16(c22, u_hi), _mm_mullo_pi16(c46, v_hi));
                __m64 g_hi = _mm_sub_pi16(y_hi, _mm_srai_pi16(g_part_hi, 6));
                __m64 b_hi = _mm_add_pi16(y_hi, _mm_srai_pi16(_mm_mullo_pi16(c113, u_hi), 6));

                // Pack
                __m64 R = _mm_packs_pu16(r_lo, r_hi);
                __m64 G = _mm_packs_pu16(g_lo, g_hi);
                __m64 B = _mm_packs_pu16(b_lo, b_hi);
                
                // 虚假累加，防止优化
                accum = _mm_xor_si64(accum, R);
                accum = _mm_xor_si64(accum, G);
                accum = _mm_xor_si64(accum, B);
            }
        }
        // 防止 accum 被删
        volatile int keep_alive = _mm_cvtsi64_si32(accum);
        (void)keep_alive;
        _mm_empty();
    }

    // =========================================================
    // ShuffleOnly: 保留 Unpack/Pack 和 复杂的 Store 逻辑
    // =========================================================
    void YUV2RGB_RGB888_ShuffleOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        __m64 zero = _mm_setzero_si64();
        // Alpha 通道填充
        __m64 alphaVec = _mm_set1_pi8(255);

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 8)
            {
                // 1. Load (Layout)
                __m64 y_raw = *(__m64 *)(YPtr + y * width + x);

                int u_val = *(int *)(UPtr + uvOffset + (x / 2));
                int v_val = *(int *)(VPtr + uvOffset + (x / 2));
                __m64 u_raw_4 = _mm_cvtsi32_si64(u_val);
                __m64 v_raw_4 = _mm_cvtsi32_si64(v_val);
                
                // Upsample (Unpack is Layout)
                __m64 u_raw = _mm_unpacklo_pi8(u_raw_4, u_raw_4);
                __m64 v_raw = _mm_unpacklo_pi8(v_raw_4, v_raw_4);

                // 2. Math Removed (Use XOR to simulate dependency)
                // 用简单的逻辑运算替代繁重的乘加
                __m64 y_lo = _mm_unpacklo_pi8(y_raw, zero);
                __m64 y_hi = _mm_unpackhi_pi8(y_raw, zero);
                
                __m64 u_lo = _mm_unpacklo_pi8(u_raw, zero); 
                __m64 u_hi = _mm_unpackhi_pi8(u_raw, zero); // unpackhi 需要重新从寄存器取，算 Layout

                // 模拟结果
                __m64 r_lo = _mm_xor_si64(y_lo, u_lo);
                __m64 g_lo = y_lo;
                __m64 b_lo = u_lo;

                __m64 r_hi = _mm_xor_si64(y_hi, u_hi);
                __m64 g_hi = y_hi;
                __m64 b_hi = u_hi;

                // 3. Pack & Interleave (核心 Layout 开销)
                __m64 R = _mm_packs_pu16(r_lo, r_hi);
                __m64 G = _mm_packs_pu16(g_lo, g_hi);
                __m64 B = _mm_packs_pu16(b_lo, b_hi);
                __m64 A = alphaVec;

                // Planar -> BGRA 
                __m64 BG_lo = _mm_unpacklo_pi8(B, G);
                __m64 BG_hi = _mm_unpackhi_pi8(B, G);
                __m64 RA_lo = _mm_unpacklo_pi8(R, A);
                __m64 RA_hi = _mm_unpackhi_pi8(R, A);

                __m64 BGRA_0 = _mm_unpacklo_pi16(BG_lo, RA_lo);
                __m64 BGRA_1 = _mm_unpackhi_pi16(BG_lo, RA_lo);
                __m64 BGRA_2 = _mm_unpacklo_pi16(BG_hi, RA_hi);
                __m64 BGRA_3 = _mm_unpackhi_pi16(BG_hi, RA_hi);

                // 4. Store Logic (Layout: Register -> Stack -> Memory)
                uint8_t *ptr = dstPtr + (y * width + x) * 3;
                int pixels[8];
                pixels[0] = _mm_cvtsi64_si32(BGRA_0);
                pixels[1] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_0, BGRA_0));
                pixels[2] = _mm_cvtsi64_si32(BGRA_1);
                pixels[3] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_1, BGRA_1));
                pixels[4] = _mm_cvtsi64_si32(BGRA_2);
                pixels[5] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_2, BGRA_2));
                pixels[6] = _mm_cvtsi64_si32(BGRA_3);
                pixels[7] = _mm_cvtsi64_si32(_mm_unpackhi_pi32(BGRA_3, BGRA_3));

                for (int k = 0; k < 8; ++k)
                {
                    *(int *)(ptr + k * 3) = pixels[k];
                }
            }
        }
        _mm_empty();
    }
}