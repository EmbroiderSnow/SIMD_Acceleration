#include "Converter.h"
#include <emmintrin.h> // SSE2
#include <algorithm>

namespace SSE
{
    // =========================================================
    // Alpha Blending (SSE2)
    // Pixel = (Pixel * alpha) >> 8
    // =========================================================
    void AlphaBlend(RGBFrame &img, uint8_t alpha)
    {
        int totalBytes = img.width * img.height * 4;
        uint8_t *data = img.data;

        // Prepare Alpha vector (expanded to 16-bit)
        __m128i alpha_vec = _mm_set1_epi16((short)alpha);
        __m128i zero = _mm_setzero_si128();

        int i = 0;
        for (; i <= totalBytes - 16; i += 16)
        {
            // Load 16 bytes
            __m128i src = _mm_loadu_si128((__m128i *)(data + i));

            // Unpack 8-bit to 16-bit
            __m128i src_lo = _mm_unpacklo_epi8(src, zero);
            __m128i src_hi = _mm_unpackhi_epi8(src, zero);

            // Multiply
            src_lo = _mm_mullo_epi16(src_lo, alpha_vec);
            src_hi = _mm_mullo_epi16(src_hi, alpha_vec);

            // Shift (Divide by 256)
            src_lo = _mm_srli_epi16(src_lo, 8);
            src_hi = _mm_srli_epi16(src_hi, 8);

            // Pack back to 8-bit with saturation
            __m128i result = _mm_packus_epi16(src_lo, src_hi);

            // Store
            _mm_storeu_si128((__m128i *)(data + i), result);
        }

        // Scalar fallback
        for (; i < totalBytes; ++i)
        {
            data[i] = (data[i] * alpha) >> 8;
        }
    }

    // =========================================================
    // Image Overlay (SSE2)
    // Dst = (Src1 * (256-a) + Src2 * a) >> 8
    // =========================================================
    void ImageOverlay(const RGBFrame &src1, const RGBFrame &src2, RGBFrame &dst, uint8_t alpha)
    {
        int totalBytes = src1.width * src1.height * 3; // RGB888

        const uint8_t *pSrc1 = src1.data;
        const uint8_t *pSrc2 = src2.data;
        uint8_t *pDst = dst.data;

        // Prepare weights
        short w1_val = 256 - alpha;
        short w2_val = alpha;

        __m128i w1_vec = _mm_set1_epi16(w1_val);
        __m128i w2_vec = _mm_set1_epi16(w2_val);
        __m128i zero = _mm_setzero_si128();

        int i = 0;
        for (; i <= totalBytes - 16; i += 16)
        {
            // Load
            __m128i s1 = _mm_loadu_si128((const __m128i *)(pSrc1 + i));
            __m128i s2 = _mm_loadu_si128((const __m128i *)(pSrc2 + i));

            // Unpack
            __m128i s1_lo = _mm_unpacklo_epi8(s1, zero);
            __m128i s1_hi = _mm_unpackhi_epi8(s1, zero);

            __m128i s2_lo = _mm_unpacklo_epi8(s2, zero);
            __m128i s2_hi = _mm_unpackhi_epi8(s2, zero);

            // Weighted Sum
            s1_lo = _mm_mullo_epi16(s1_lo, w1_vec);
            s1_hi = _mm_mullo_epi16(s1_hi, w1_vec);

            s2_lo = _mm_mullo_epi16(s2_lo, w2_vec);
            s2_hi = _mm_mullo_epi16(s2_hi, w2_vec);

            __m128i res_lo = _mm_add_epi16(s1_lo, s2_lo);
            __m128i res_hi = _mm_add_epi16(s1_hi, s2_hi);

            // Shift
            res_lo = _mm_srli_epi16(res_lo, 8);
            res_hi = _mm_srli_epi16(res_hi, 8);

            // Pack & Store
            __m128i result = _mm_packus_epi16(res_lo, res_hi);
            _mm_storeu_si128((__m128i *)(pDst + i), result);
        }

        // Scalar fallback
        for (; i < totalBytes; ++i)
        {
            int val = (pSrc1[i] * w1_val + pSrc2[i] * w2_val) >> 8;
            pDst[i] = (uint8_t)val;
        }
    }

    // =========================================================
    // YUV420 -> ARGB8888 (SSE2)
    // =========================================================
    void YUV2RGB_ARGB8888(const YUVFrame &src, RGBFrame &dst, uint8_t alpha)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // Coefficients (Shift 6)
        __m128i c90 = _mm_set1_epi16(90);
        __m128i c22 = _mm_set1_epi16(22);
        __m128i c46 = _mm_set1_epi16(46);
        __m128i c113 = _mm_set1_epi16(113);

        __m128i zero = _mm_setzero_si128();
        __m128i c128 = _mm_set1_epi16(128);
        __m128i alphaVec = _mm_set1_epi8(alpha);

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            
            for (int x = 0; x < width; x += 16)
            {
                // Load Y (16 pixels)
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));

                // Load U/V (8 pixels)
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                // Upsample UV (8 -> 16)
                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                // Unpack to 16-bit and subtract 128
                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);

                __m128i u_lo = _mm_sub_epi16(_mm_unpacklo_epi8(u_raw, zero), c128);
                __m128i u_hi = _mm_sub_epi16(_mm_unpackhi_epi8(u_raw, zero), c128);

                __m128i v_lo = _mm_sub_epi16(_mm_unpacklo_epi8(v_raw, zero), c128);
                __m128i v_hi = _mm_sub_epi16(_mm_unpackhi_epi8(v_raw, zero), c128);

                // Calculate RGB (Shift 6)
                // R
                __m128i r_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c90, v_lo), 6));
                __m128i r_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c90, v_hi), 6));
                // B
                __m128i b_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c113, u_lo), 6));
                __m128i b_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c113, u_hi), 6));
                // G
                __m128i g_part_lo = _mm_add_epi16(_mm_mullo_epi16(c22, u_lo), _mm_mullo_epi16(c46, v_lo));
                __m128i g_part_hi = _mm_add_epi16(_mm_mullo_epi16(c22, u_hi), _mm_mullo_epi16(c46, v_hi));
                __m128i g_lo = _mm_sub_epi16(y_lo, _mm_srai_epi16(g_part_lo, 6));
                __m128i g_hi = _mm_sub_epi16(y_hi, _mm_srai_epi16(g_part_hi, 6));

                // Pack to 8-bit
                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec;

                // Interleave Planar to BGRA
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                __m128i result_0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i result_1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i result_2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i result_3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // Store
                __m128i *d = (__m128i *)(dstPtr + (y * width + x) * 4);
                _mm_storeu_si128(d + 0, result_0);
                _mm_storeu_si128(d + 1, result_1);
                _mm_storeu_si128(d + 2, result_2);
                _mm_storeu_si128(d + 3, result_3);
            }
        }
    }

    // =========================================================
    // YUV420 -> RGB888 (SSE2)
    // Uses overlap-write strategy for 3-byte pixels
    // =========================================================
    void YUV2RGB_RGB888(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        // Coefficients (Shift 6)
        __m128i c90 = _mm_set1_epi16(90);
        __m128i c22 = _mm_set1_epi16(22);
        __m128i c46 = _mm_set1_epi16(46);
        __m128i c113 = _mm_set1_epi16(113);

        __m128i zero = _mm_setzero_si128();
        __m128i c128 = _mm_set1_epi16(128);
        __m128i alphaVec = _mm_set1_epi8(0); 

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 16)
            {
                // Load & Upsample
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                // Unpack & Sub 128
                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);
                __m128i u_lo = _mm_sub_epi16(_mm_unpacklo_epi8(u_raw, zero), c128);
                __m128i u_hi = _mm_sub_epi16(_mm_unpackhi_epi8(u_raw, zero), c128);
                __m128i v_lo = _mm_sub_epi16(_mm_unpacklo_epi8(v_raw, zero), c128);
                __m128i v_hi = _mm_sub_epi16(_mm_unpackhi_epi8(v_raw, zero), c128);

                // Calculate RGB
                __m128i r_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c90, v_lo), 6));
                __m128i r_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c90, v_hi), 6));

                __m128i b_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c113, u_lo), 6));
                __m128i b_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c113, u_hi), 6));

                __m128i g_part_lo = _mm_add_epi16(_mm_mullo_epi16(c22, u_lo), _mm_mullo_epi16(c46, v_lo));
                __m128i g_part_hi = _mm_add_epi16(_mm_mullo_epi16(c22, u_hi), _mm_mullo_epi16(c46, v_hi));
                __m128i g_lo = _mm_sub_epi16(y_lo, _mm_srai_epi16(g_part_lo, 6));
                __m128i g_hi = _mm_sub_epi16(y_hi, _mm_srai_epi16(g_part_hi, 6));

                // Pack
                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec;

                // Interleave to BGRA
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                __m128i res0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i res1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i res2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i res3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // Buffer result on stack
                uint32_t buffer[16]; 
                _mm_storeu_si128((__m128i*)&buffer[0], res0);
                _mm_storeu_si128((__m128i*)&buffer[4], res1);
                _mm_storeu_si128((__m128i*)&buffer[8], res2);
                _mm_storeu_si128((__m128i*)&buffer[12], res3);

                // Overlap-write (writing 4 bytes per 3-byte pixel stride)
                uint8_t *ptr = dstPtr + (y * width + x) * 3;
                for (int k = 0; k < 16; ++k) {
                    *(uint32_t*)(ptr + k * 3) = buffer[k];
                }
            }
        }
    }

    // Helper: Horizontal Add of 32-bit Integers
    static inline int32_t hadd_epi32_sum(__m128i reg) {
        __m128i high = _mm_unpackhi_epi64(reg, reg);
        __m128i sum  = _mm_add_epi32(reg, high);
        __m128i sum_high = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 1, 1, 1));
        sum = _mm_add_epi32(sum, sum_high);
        return _mm_cvtsi128_si32(sum);
    }

    // =========================================================
    // ARGB8888 -> YUV420 (SSE2)
    // Process 2x8 block
    // =========================================================
    void RGB2YUV_ARGB8888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        // Y: 29*B + 150*G + 77*R
        __m128i cY = _mm_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0);
        // U: 127*B - 84*G - 43*R
        __m128i cU = _mm_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0);
        // V: -20*B - 107*G + 127*R
        __m128i cV = _mm_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0);

        __m128i zero = _mm_setzero_si128();

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 8)
            {
                // --- 1. Compute Y (for 2 rows) ---
                for (int r = 0; r < 2; ++r) 
                {
                    int currY = y + r;
                    const uint8_t* pSrc = src.data + (currY * width + x) * 4;
                    uint8_t* pDstY = dst.Y + currY * width + x;

                    // Load 8 pixels
                    __m128i px1 = _mm_loadu_si128((const __m128i*)pSrc);
                    __m128i px2 = _mm_loadu_si128((const __m128i*)(pSrc + 16));

                    // Unpack to 16-bit
                    __m128i p0 = _mm_unpacklo_epi8(px1, zero);
                    __m128i p1 = _mm_unpackhi_epi8(px1, zero);
                    __m128i p2 = _mm_unpacklo_epi8(px2, zero);
                    __m128i p3 = _mm_unpackhi_epi8(px2, zero);

                    // Madd and HAdd
                    __m128i y0_parts = _mm_madd_epi16(p0, cY); 
                    __m128i y1_parts = _mm_madd_epi16(p1, cY);
                    __m128i y2_parts = _mm_madd_epi16(p2, cY);
                    __m128i y3_parts = _mm_madd_epi16(p3, cY);

                    __m128i y01 = _mm_add_epi32(y0_parts, _mm_shuffle_epi32(y0_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y23 = _mm_add_epi32(y1_parts, _mm_shuffle_epi32(y1_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y45 = _mm_add_epi32(y2_parts, _mm_shuffle_epi32(y2_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y67 = _mm_add_epi32(y3_parts, _mm_shuffle_epi32(y3_parts, _MM_SHUFFLE(2, 3, 0, 1)));

                    // Extract and Store
                    int val0 = _mm_cvtsi128_si32(y01) >> 8;
                    int val1 = _mm_cvtsi128_si32(_mm_srli_si128(y01, 8)) >> 8;
                    int val2 = _mm_cvtsi128_si32(y23) >> 8;
                    int val3 = _mm_cvtsi128_si32(_mm_srli_si128(y23, 8)) >> 8;
                    int val4 = _mm_cvtsi128_si32(y45) >> 8;
                    int val5 = _mm_cvtsi128_si32(_mm_srli_si128(y45, 8)) >> 8;
                    int val6 = _mm_cvtsi128_si32(y67) >> 8;
                    int val7 = _mm_cvtsi128_si32(_mm_srli_si128(y67, 8)) >> 8;
                    
                    pDstY[0] = std::clamp(val0, 0, 255);
                    pDstY[1] = std::clamp(val1, 0, 255);
                    pDstY[2] = std::clamp(val2, 0, 255);
                    pDstY[3] = std::clamp(val3, 0, 255);
                    pDstY[4] = std::clamp(val4, 0, 255);
                    pDstY[5] = std::clamp(val5, 0, 255);
                    pDstY[6] = std::clamp(val6, 0, 255);
                    pDstY[7] = std::clamp(val7, 0, 255);
                }

                // --- 2. Compute UV (2x2 Average) ---
                __m128i r0_p1 = _mm_loadu_si128((const __m128i*)(src.data + (y * width + x) * 4));
                __m128i r0_p2 = _mm_loadu_si128((const __m128i*)(src.data + (y * width + x) * 4 + 16));
                __m128i r1_p1 = _mm_loadu_si128((const __m128i*)(src.data + ((y + 1) * width + x) * 4));
                __m128i r1_p2 = _mm_loadu_si128((const __m128i*)(src.data + ((y + 1) * width + x) * 4 + 16));

                __m128i r0_lo = _mm_unpacklo_epi8(r0_p1, zero);
                __m128i r0_hi = _mm_unpackhi_epi8(r0_p1, zero);
                __m128i r0_lo2 = _mm_unpacklo_epi8(r0_p2, zero);
                __m128i r0_hi2 = _mm_unpackhi_epi8(r0_p2, zero);

                __m128i r1_lo = _mm_unpacklo_epi8(r1_p1, zero);
                __m128i r1_hi = _mm_unpackhi_epi8(r1_p1, zero);
                __m128i r1_lo2 = _mm_unpacklo_epi8(r1_p2, zero);
                __m128i r1_hi2 = _mm_unpackhi_epi8(r1_p2, zero);

                // Vertical Sum
                __m128i sum01 = _mm_add_epi16(r0_lo, r1_lo);
                __m128i sum23 = _mm_add_epi16(r0_hi, r1_hi);
                __m128i sum45 = _mm_add_epi16(r0_lo2, r1_lo2);
                __m128i sum67 = _mm_add_epi16(r0_hi2, r1_hi2);

                // Horizontal Sum & Average
                __m128i avg0 = _mm_add_epi16(sum01, _mm_bsrli_si128(sum01, 8));
                __m128i avg1 = _mm_add_epi16(sum23, _mm_bsrli_si128(sum23, 8));
                __m128i avg2 = _mm_add_epi16(sum45, _mm_bsrli_si128(sum45, 8));
                __m128i avg3 = _mm_add_epi16(sum67, _mm_bsrli_si128(sum67, 8));

                avg0 = _mm_srli_epi16(avg0, 2);
                avg1 = _mm_srli_epi16(avg1, 2);
                avg2 = _mm_srli_epi16(avg2, 2);
                avg3 = _mm_srli_epi16(avg3, 2);

                // Calc U, V
                int u0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cU)) >> 8) + 128;
                int u1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cU)) >> 8) + 128;
                int u2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cU)) >> 8) + 128;
                int u3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cU)) >> 8) + 128;

                int v0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cV)) >> 8) + 128;
                int v1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cV)) >> 8) + 128;
                int v2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cV)) >> 8) + 128;
                int v3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cV)) >> 8) + 128;

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                dst.U[uvIdx + 0] = std::clamp(u0, 0, 255);
                dst.U[uvIdx + 1] = std::clamp(u1, 0, 255);
                dst.U[uvIdx + 2] = std::clamp(u2, 0, 255);
                dst.U[uvIdx + 3] = std::clamp(u3, 0, 255);

                dst.V[uvIdx + 0] = std::clamp(v0, 0, 255);
                dst.V[uvIdx + 1] = std::clamp(v1, 0, 255);
                dst.V[uvIdx + 2] = std::clamp(v2, 0, 255);
                dst.V[uvIdx + 3] = std::clamp(v3, 0, 255);
            }
        }
    }

    // Helper: Load 4 packed RGB pixels and convert to 32-bit integers
    static inline __m128i load_rgb_4px(const uint8_t* ptr) {
        __m128i raw = _mm_loadu_si128((const __m128i*)ptr);
        __m128i p0 = raw; 
        __m128i p1 = _mm_srli_si128(raw, 3);
        __m128i p2 = _mm_srli_si128(raw, 6);
        __m128i p3 = _mm_srli_si128(raw, 9);
        __m128i p01 = _mm_unpacklo_epi32(p0, p1);
        __m128i p23 = _mm_unpacklo_epi32(p2, p3);
        __m128i final_reg = _mm_unpacklo_epi64(p01, p23);
        __m128i mask = _mm_set1_epi32(0x00FFFFFF);
        return _mm_and_si128(final_reg, mask);
    }

    // =========================================================
    // RGB888 -> YUV420 (SSE2)
    // =========================================================
    void RGB2YUV_RGB888(const RGBFrame &src, YUVFrame &dst)
    {
        int width = src.width;
        int height = src.height;

        __m128i cY = _mm_setr_epi16(29, 150, 77, 0, 29, 150, 77, 0);
        __m128i cU = _mm_setr_epi16(127, -84, -43, 0, 127, -84, -43, 0);
        __m128i cV = _mm_setr_epi16(-20, -107, 127, 0, -20, -107, 127, 0);

        __m128i zero = _mm_setzero_si128();

        for (int y = 0; y < height; y += 2)
        {
            for (int x = 0; x < width; x += 8)
            {
                // --- 1. Compute Y ---
                for (int r = 0; r < 2; ++r) 
                {
                    int currY = y + r;
                    const uint8_t* pSrc = src.data + (currY * width + x) * 3;
                    uint8_t* pDstY = dst.Y + currY * width + x;

                    // Load 8 RGB pixels using helper
                    __m128i px1 = load_rgb_4px(pSrc);
                    __m128i px2 = load_rgb_4px(pSrc + 12);

                    // Unpack
                    __m128i p0 = _mm_unpacklo_epi8(px1, zero);
                    __m128i p1 = _mm_unpackhi_epi8(px1, zero);
                    __m128i p2 = _mm_unpacklo_epi8(px2, zero);
                    __m128i p3 = _mm_unpackhi_epi8(px2, zero);

                    // Calc Y
                    __m128i y0_parts = _mm_madd_epi16(p0, cY); 
                    __m128i y1_parts = _mm_madd_epi16(p1, cY);
                    __m128i y2_parts = _mm_madd_epi16(p2, cY);
                    __m128i y3_parts = _mm_madd_epi16(p3, cY);

                    __m128i y01 = _mm_add_epi32(y0_parts, _mm_shuffle_epi32(y0_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y23 = _mm_add_epi32(y1_parts, _mm_shuffle_epi32(y1_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y45 = _mm_add_epi32(y2_parts, _mm_shuffle_epi32(y2_parts, _MM_SHUFFLE(2, 3, 0, 1)));
                    __m128i y67 = _mm_add_epi32(y3_parts, _mm_shuffle_epi32(y3_parts, _MM_SHUFFLE(2, 3, 0, 1)));

                    // Store Y
                    int val0 = _mm_cvtsi128_si32(y01) >> 8;
                    int val1 = _mm_cvtsi128_si32(_mm_srli_si128(y01, 8)) >> 8;
                    int val2 = _mm_cvtsi128_si32(y23) >> 8;
                    int val3 = _mm_cvtsi128_si32(_mm_srli_si128(y23, 8)) >> 8;
                    int val4 = _mm_cvtsi128_si32(y45) >> 8;
                    int val5 = _mm_cvtsi128_si32(_mm_srli_si128(y45, 8)) >> 8;
                    int val6 = _mm_cvtsi128_si32(y67) >> 8;
                    int val7 = _mm_cvtsi128_si32(_mm_srli_si128(y67, 8)) >> 8;
                    
                    pDstY[0] = std::clamp(val0, 0, 255);
                    pDstY[1] = std::clamp(val1, 0, 255);
                    pDstY[2] = std::clamp(val2, 0, 255);
                    pDstY[3] = std::clamp(val3, 0, 255);
                    pDstY[4] = std::clamp(val4, 0, 255);
                    pDstY[5] = std::clamp(val5, 0, 255);
                    pDstY[6] = std::clamp(val6, 0, 255);
                    pDstY[7] = std::clamp(val7, 0, 255);
                }

                // --- 2. Compute UV ---
                __m128i r0_p1 = load_rgb_4px(src.data + (y * width + x) * 3);
                __m128i r0_p2 = load_rgb_4px(src.data + (y * width + x) * 3 + 12);
                __m128i r1_p1 = load_rgb_4px(src.data + ((y + 1) * width + x) * 3);
                __m128i r1_p2 = load_rgb_4px(src.data + ((y + 1) * width + x) * 3 + 12);

                __m128i r0_lo = _mm_unpacklo_epi8(r0_p1, zero);
                __m128i r0_hi = _mm_unpackhi_epi8(r0_p1, zero);
                __m128i r0_lo2 = _mm_unpacklo_epi8(r0_p2, zero);
                __m128i r0_hi2 = _mm_unpackhi_epi8(r0_p2, zero);

                __m128i r1_lo = _mm_unpacklo_epi8(r1_p1, zero);
                __m128i r1_hi = _mm_unpackhi_epi8(r1_p1, zero);
                __m128i r1_lo2 = _mm_unpacklo_epi8(r1_p2, zero);
                __m128i r1_hi2 = _mm_unpackhi_epi8(r1_p2, zero);

                // Vertical Sum
                __m128i sum01 = _mm_add_epi16(r0_lo, r1_lo);
                __m128i sum23 = _mm_add_epi16(r0_hi, r1_hi);
                __m128i sum45 = _mm_add_epi16(r0_lo2, r1_lo2);
                __m128i sum67 = _mm_add_epi16(r0_hi2, r1_hi2);

                // Horizontal Sum & Average
                __m128i avg0 = _mm_add_epi16(sum01, _mm_bsrli_si128(sum01, 8));
                __m128i avg1 = _mm_add_epi16(sum23, _mm_bsrli_si128(sum23, 8));
                __m128i avg2 = _mm_add_epi16(sum45, _mm_bsrli_si128(sum45, 8));
                __m128i avg3 = _mm_add_epi16(sum67, _mm_bsrli_si128(sum67, 8));

                avg0 = _mm_srli_epi16(avg0, 2);
                avg1 = _mm_srli_epi16(avg1, 2);
                avg2 = _mm_srli_epi16(avg2, 2);
                avg3 = _mm_srli_epi16(avg3, 2);

                // Calc U, V
                int u0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cU)) >> 8) + 128;
                int u1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cU)) >> 8) + 128;
                int u2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cU)) >> 8) + 128;
                int u3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cU)) >> 8) + 128;

                int v0 = (hadd_epi32_sum(_mm_madd_epi16(avg0, cV)) >> 8) + 128;
                int v1 = (hadd_epi32_sum(_mm_madd_epi16(avg1, cV)) >> 8) + 128;
                int v2 = (hadd_epi32_sum(_mm_madd_epi16(avg2, cV)) >> 8) + 128;
                int v3 = (hadd_epi32_sum(_mm_madd_epi16(avg3, cV)) >> 8) + 128;

                int uvIdx = (y / 2) * (width / 2) + (x / 2);
                dst.U[uvIdx + 0] = std::clamp(u0, 0, 255);
                dst.U[uvIdx + 1] = std::clamp(u1, 0, 255);
                dst.U[uvIdx + 2] = std::clamp(u2, 0, 255);
                dst.U[uvIdx + 3] = std::clamp(u3, 0, 255);

                dst.V[uvIdx + 0] = std::clamp(v0, 0, 255);
                dst.V[uvIdx + 1] = std::clamp(v1, 0, 255);
                dst.V[uvIdx + 2] = std::clamp(v2, 0, 255);
                dst.V[uvIdx + 3] = std::clamp(v3, 0, 255);
            }
        }
    }

    // ---------------------------------------------------------
    // MemOnly: 模拟 Stack Buffer 中转 + 重叠写入
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_MemOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        uint8_t *dstPtr = dst.data;
        const uint8_t *YPtr = src.Y;

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            volatile long long u_dummy = *(long long*)(src.U + uvOffset); // force read
            (void)u_dummy; 

            for (int x = 0; x < width; x += 16)
            {
                // 1. Load Y
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));

                // 2. 构造 Dummy 结果 (模拟 unpack/pack 后的结果)
                __m128i res_dummy = y_raw; 

                // 3. Buffer on Stack (复刻原版逻辑)
                uint32_t buffer[16]; 
                _mm_storeu_si128((__m128i*)&buffer[0], res_dummy);
                _mm_storeu_si128((__m128i*)&buffer[4], res_dummy);
                _mm_storeu_si128((__m128i*)&buffer[8], res_dummy);
                _mm_storeu_si128((__m128i*)&buffer[12], res_dummy);

                // 4. Overlap-write (瓶颈所在)
                uint8_t *ptr = dstPtr + (y * width + x) * 3;
                for (int k = 0; k < 16; ++k) {
                    *(uint32_t*)(ptr + k * 3) = buffer[k];
                }
            }
        }
    }

    // ---------------------------------------------------------
    // ComputeOnly: 主要是 16-bit 运算
    // ---------------------------------------------------------
    void YUV2RGB_RGB888_ComputeOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;

        __m128i c90 = _mm_set1_epi16(90);
        __m128i c22 = _mm_set1_epi16(22);
        __m128i c46 = _mm_set1_epi16(46);
        __m128i c113 = _mm_set1_epi16(113);
        __m128i zero = _mm_setzero_si128();
        __m128i c128 = _mm_set1_epi16(128);
        __m128i alphaVec = _mm_setzero_si128();

        __m128i accum = _mm_setzero_si128();

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);
            for (int x = 0; x < width; x += 16)
            {
                // Load & Compute Logic ... [保持原版]
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);
                __m128i u_lo = _mm_sub_epi16(_mm_unpacklo_epi8(u_raw, zero), c128);
                __m128i u_hi = _mm_sub_epi16(_mm_unpackhi_epi8(u_raw, zero), c128);
                __m128i v_lo = _mm_sub_epi16(_mm_unpacklo_epi8(v_raw, zero), c128);
                __m128i v_hi = _mm_sub_epi16(_mm_unpackhi_epi8(v_raw, zero), c128);

                __m128i r_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c90, v_lo), 6));
                __m128i r_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c90, v_hi), 6));
                __m128i b_lo = _mm_add_epi16(y_lo, _mm_srai_epi16(_mm_mullo_epi16(c113, u_lo), 6));
                __m128i b_hi = _mm_add_epi16(y_hi, _mm_srai_epi16(_mm_mullo_epi16(c113, u_hi), 6));

                __m128i g_part_lo = _mm_add_epi16(_mm_mullo_epi16(c22, u_lo), _mm_mullo_epi16(c46, v_lo));
                __m128i g_part_hi = _mm_add_epi16(_mm_mullo_epi16(c22, u_hi), _mm_mullo_epi16(c46, v_hi));
                __m128i g_lo = _mm_sub_epi16(y_lo, _mm_srai_epi16(g_part_lo, 6));
                __m128i g_hi = _mm_sub_epi16(y_hi, _mm_srai_epi16(g_part_hi, 6));

                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec;

                // Interleave 
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                __m128i res0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i res1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i res2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i res3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // 虚假累加
                accum = _mm_xor_si128(accum, res0);
                accum = _mm_xor_si128(accum, res1);
                accum = _mm_xor_si128(accum, res2);
                accum = _mm_xor_si128(accum, res3);
            }
        }
        volatile int keep_alive = _mm_cvtsi128_si32(accum);
        (void)keep_alive;
    }

    // =========================================================
    // ShuffleOnly: 保留 Interleave 和 Stack Buffer Store
    // =========================================================
    void YUV2RGB_RGB888_ShuffleOnly(const YUVFrame &src, RGBFrame &dst)
    {
        int width = src.width;
        int height = src.height;
        const uint8_t *YPtr = src.Y;
        const uint8_t *UPtr = src.U;
        const uint8_t *VPtr = src.V;
        uint8_t *dstPtr = dst.data;

        __m128i zero = _mm_setzero_si128();
        __m128i alphaVec = zero;

        for (int y = 0; y < height; y++)
        {
            int uvOffset = (y / 2) * (width / 2);

            for (int x = 0; x < width; x += 16)
            {
                // 1. Load & Upsample (Layout)
                __m128i y_raw = _mm_loadu_si128((const __m128i *)(YPtr + y * width + x));
                __m128i u_raw_8 = _mm_loadl_epi64((const __m128i *)(UPtr + uvOffset + (x / 2)));
                __m128i v_raw_8 = _mm_loadl_epi64((const __m128i *)(VPtr + uvOffset + (x / 2)));

                __m128i u_raw = _mm_unpacklo_epi8(u_raw_8, u_raw_8);
                __m128i v_raw = _mm_unpacklo_epi8(v_raw_8, v_raw_8);

                // 2. Math Removed (Simulate dependency)
                // 仅 unpack，不进行 calculate
                __m128i y_lo = _mm_unpacklo_epi8(y_raw, zero);
                __m128i y_hi = _mm_unpackhi_epi8(y_raw, zero);
                __m128i u_lo = _mm_unpacklo_epi8(u_raw, zero);
                __m128i u_hi = _mm_unpackhi_epi8(u_raw, zero);

                // 模拟结果
                __m128i r_lo = _mm_or_si128(y_lo, u_lo);
                __m128i r_hi = _mm_or_si128(y_hi, u_hi);
                __m128i g_lo = y_lo;
                __m128i g_hi = y_hi;
                __m128i b_lo = u_lo;
                __m128i b_hi = u_hi;

                // 3. Pack & Interleave (Layout)
                __m128i R = _mm_packus_epi16(r_lo, r_hi);
                __m128i G = _mm_packus_epi16(g_lo, g_hi);
                __m128i B = _mm_packus_epi16(b_lo, b_hi);
                __m128i A = alphaVec;

                // Planar -> BGRA Interleaving
                __m128i BG_lo = _mm_unpacklo_epi8(B, G);
                __m128i BG_hi = _mm_unpackhi_epi8(B, G);
                __m128i RA_lo = _mm_unpacklo_epi8(R, A);
                __m128i RA_hi = _mm_unpackhi_epi8(R, A);

                __m128i res0 = _mm_unpacklo_epi16(BG_lo, RA_lo);
                __m128i res1 = _mm_unpackhi_epi16(BG_lo, RA_lo);
                __m128i res2 = _mm_unpacklo_epi16(BG_hi, RA_hi);
                __m128i res3 = _mm_unpackhi_epi16(BG_hi, RA_hi);

                // 4. Store Logic (Layout: Reg -> Stack -> Mem)
                uint32_t buffer[16]; 
                _mm_storeu_si128((__m128i*)&buffer[0], res0);
                _mm_storeu_si128((__m128i*)&buffer[4], res1);
                _mm_storeu_si128((__m128i*)&buffer[8], res2);
                _mm_storeu_si128((__m128i*)&buffer[12], res3);

                uint8_t *ptr = dstPtr + (y * width + x) * 3;
                for (int k = 0; k < 16; ++k) {
                    *(uint32_t*)(ptr + k * 3) = buffer[k];
                }
            }
        }
    }
}