/*
 * mandelbrot.cc
 *
 * An improvement over the naive algorithm for rendering the mandelbrot
 * set by using OpenMP and SSE2 optimizations. There are several more advanced
 * algorithms that achieve faster approximations, the goal here was to speedup
 * the 'standard' approach. Although unlikely, maybe it will be of use to
 * someone.
 *
 * @author ciphron <ciphron@ciphron.org>
 * Written in April 2009
 *
 * ****************************************************************************
 *  This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 3
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
 *
 * ****************************************************************************
 */

#include <stdio.h>
#include <math.h>
#include <stdint.h>

// SSE Intrinsics
#include <xmmintrin.h>
#include <emmintrin.h>

#include <omp.h> // OpenMP

#include <SDL/SDL.h>

const int X_RES = 700;          // horizontal resolution
const int Y_RES = 700;          // vertical resolution

const int MAX_ITS = 500;       // Max iterations to check if a point escapes
const int MAX_DEPTH = 150;      // max depth of zoom
const float ZOOM_FACTOR = 1.07; // zoom between each frame

/* Part of the image to zoom in on */
const float PX = -0.702295281061;     // real component
const float PY = +0.350220783400;     // imaginary component

const __m128i     MAX_ITERATIONS_4 = _mm_set1_epi32(MAX_ITS);
const __m128    DIST_LIMIT_4 = _mm_set1_ps(4.0f);


/*
 * Color Palette
 */
unsigned char pal[] = {
    0, 0, 0,
    255, 180, 4,
    240, 156, 4,
    220, 124, 4,
    156, 71, 4,
    72, 20, 4,
    251, 180, 4,
    180, 74, 4,
    180, 70, 4,
    164, 91, 4,
    100, 28, 4,
    191, 82, 4,
    47, 5, 4,
    138, 39, 4,
    81, 27, 4,
    192, 89, 4,
    61, 27, 4,
    216, 148, 4,
    71, 14, 4,
    142, 48, 4,
    196, 102, 4,
    58, 9, 4,
    132, 45, 4,
    95, 15, 4,
    92, 21, 4,
    166, 59, 4,
    244, 178, 4,
    194, 121, 4,
    120, 41, 4,
    53, 14, 4,
    80, 15, 4,
    23, 3, 4,
    249, 204, 4,
    97, 25, 4,
    124, 30, 4,
    151, 57, 4,
    104, 36, 4,
    239, 171, 4,
    131, 57, 4,
    111, 23, 4,
    4, 2, 4,
    255, 180, 4,
    240, 156, 4,
    220, 124, 4,
    156, 71, 4,
    72, 20, 4,
    251, 180, 4,
    180, 74, 4,
    180, 70, 4,
    164, 91, 4,
    100, 28, 4,
    191, 82, 4,
    47, 5, 4,
    138, 39, 4,
    81, 27, 4,
    192, 89, 4,
    61, 27, 4,
    216, 148, 4,
    71, 14, 4,
    142, 48, 4,
    196, 102, 4,
    58, 9, 4,
    132, 45, 4,
    95, 15, 4,
    92, 21, 4,

};
const int PAL_SIZE = 40;        // Number of entries in the palette 



inline void putpixel(SDL_Surface *surface, int x, int y, uint8_t r, uint8_t g,
                     uint8_t b)
{
	uint32_t *pix_buf = (Uint32*)surface->pixels;
	const uint32_t pixel = (r << 16) + (g << 8) + b;
	const int line_offset = y * (surface->pitch >> 2);

	pix_buf[line_offset + x] = pixel;
}

/* 
 * Determine concurrently whether 4 points are members of the mandelbrot set
 * by checking whether they exceed a distance limit within a bounded number
 * of iterations. A packed integer vector is returned containing the number
 * of iterations executed for each point (in the corresponding position).
 */
inline __m128i member4(__m128 cx4, __m128 cy4)
{
    // We proceed in reverse down to 0
    __m128i iterations4 = MAX_ITERATIONS_4;

    __m128 x4 = cx4;
    __m128 y4 = cy4;
    __m128 x_sq4 = _mm_mul_ps(x4, x4);
    __m128 y_sq4 = _mm_mul_ps(y4, y4);
    __m128 aux4_a = _mm_add_ps(x_sq4, y_sq4);
    __m128i aux4_b;
    __m128i zero4 = _mm_setzero_si128();
    __m128i not_escape4;

    
    aux4_a = _mm_cmplt_ps(aux4_a, DIST_LIMIT_4);
    not_escape4 = (__m128i)aux4_a; // used to track

    /*
     * For all elements that have escaped, their corresponding value in
     * not_escape4 = 0. Otherwise is is 0xFFFFFFFF
     */

    // Mask away iterations of elements that have escaped
    aux4_b = _mm_and_si128(not_escape4, iterations4);

    // Check that the iterations have not reached 0
    aux4_b = _mm_cmpeq_epi32(aux4_b, zero4);
    int mask = _mm_movemask_epi8(aux4_b);
    while (mask != 0xFFFF) {
        /*
         * For all non-escaped points, 0xFFFFFFFF is in respective position
         * of not_escape4. This number is the 2's complement of -1. Adding to
         * iterations4 therefore decrements. But iteration count of escaped
         * points is preserved (since respective position of not_escape4 is 0)
         */
         
        iterations4 = _mm_add_epi32(iterations4, not_escape4);

        y4 = _mm_mul_ps(x4, y4); // x * y
        y4 = _mm_add_ps(y4, y4); // (x * y) + (x * y) = 2*x*y
        y4 = _mm_add_ps(y4, cy4); // 2*x*y + cy

        x4 = _mm_sub_ps(x_sq4, y_sq4); // x*x - y*y
        x4 = _mm_add_ps(x4, cx4); // (x*x - y*y) + cx

        x_sq4 = _mm_mul_ps(x4, x4); // x * x
        y_sq4 = _mm_mul_ps(y4, y4); // y * y

        aux4_a = _mm_add_ps(x_sq4, y_sq4); // x*x + y*y
        aux4_a = _mm_cmplt_ps(aux4_a, DIST_LIMIT_4); // (x*x + y*y) < 4 (limit)
        not_escape4 = _mm_and_si128((__m128i)aux4_a, not_escape4);

        aux4_b = _mm_and_si128(not_escape4, iterations4);
        aux4_b = _mm_cmpeq_epi32(aux4_b, zero4);
        mask = _mm_movemask_epi8(aux4_b);
    }

    return _mm_sub_epi32(MAX_ITERATIONS_4, iterations4);
}


/**
 * TODO: refactor (break up into smaller functions)
 */
void mandelbrot(SDL_Surface *surface)
{
    SDL_Event event; // for handling SDL events

    // Zoom (replace dividing by m with multiplying by 1 / zoom_factor)
    const float zoom_multiplier = 1.0f / ZOOM_FACTOR;
    const __m128 zoom_multiplier4 = _mm_set1_ps(zoom_multiplier);

    // Deltas
    const float delta_x = (1.0f / X_RES) * 4;
    __m128 delta_x4 = _mm_set1_ps(delta_x * 4); // skipping over 4 points
    float delta_y = (1.0f / Y_RES) * 4.0f;

    // Offsets
    const float center = -0.5f * 4.0f;
    float y_offset = center;
    __m128 x_offset4 = _mm_setr_ps(center, center + delta_x,
                                   center + 2*delta_x, center + 3*delta_x);

    // Center Translations
    const __m128 x_translate4 = _mm_set1_ps(PX);
    //const __m128 y_translate4 = _mm_set1_ps(PY);
    const __m128i increment4 = _mm_set1_epi32(1);

    // Masks
    const __m128i all_ones_mask4 = _mm_set1_epi32(0xFFFFFFFF);
    const __m128i mod_mask4 = _mm_set1_epi32(0x3F);


    int depth = 0;

    bool quit = false;

    while (!quit) {
        const float y_base = PY + y_offset; // translate y


        #pragma omp parallel for default(none), shared(surface, pal),\
                                 firstprivate(delta_y,\
                                              x_offset4, delta_x4),\
                                 schedule(guided, 50)
        
        for (int hy = 0; hy < Y_RES; hy++) {
            __m128 y4 = _mm_set1_ps(y_base + hy*delta_y);
            __m128 x4 = _mm_add_ps(x_translate4, x_offset4);

            for (int hx = 0; hx < X_RES; hx += 4) {
                /*
                 * Check if all 4 points are members of mandelbrot set
                 * member4 returns a vector containing the corresponding
                 * number of iterations executed, for each point
                 */
                __m128i iterations4 = member4(x4, y4);


                /*
                 * There is no neq or lt integer comparisons in SSE2
                 * Therefore must use eq, and invert using XOR
                 */
                __m128i max_mask4 = _mm_cmpeq_epi32(iterations4,
                                                    MAX_ITERATIONS_4);
                max_mask4 = _mm_xor_si128(max_mask4, all_ones_mask4);


                /*
                 * Mod 64 is equivilent to ANDing with 0x3F. I expanded the
                 * color table from 40 to 64 entries, and replicated the first
                 * 24 in the additonal 24. ANDing with 0x3F is then = mod 40
                 */
                iterations4 = _mm_and_si128(iterations4, mod_mask4);

                // Skip first color in the palette (which is black)
                iterations4 = _mm_add_epi32(iterations4, increment4);
                
                /*
                 * Finally, apply mask to retain nonzero color index if the
                 * iteration count was less than the escape limit
                 */
                iterations4 = _mm_and_si128(iterations4, max_mask4);
                  
                union {
                    __m128i v;
                    int color_index[4];
                } u;

                u.v = iterations4;

                for (int j = 0; j < 4; j++) {
                    int index = u.color_index[j] * 3;
                    
                    putpixel(surface, hx + j, hy, pal[index],
                             pal[index + 1], pal[index + 2]);
                }

                x4 = _mm_add_ps(x4, delta_x4);
            }
        }

        // Show the rendered fractal
        SDL_Flip(surface);


        if (depth < MAX_DEPTH) {

            depth++;

            // Zoom in
            delta_x4 = _mm_mul_ps(delta_x4, zoom_multiplier4);
            x_offset4 = _mm_mul_ps(x_offset4, zoom_multiplier4);
            delta_y *= zoom_multiplier;
            y_offset *= zoom_multiplier;
        }

        /*
         * We can handle events such SDL_KEYDOWN to allow navigation
         * around the fractal
         */
        if(SDL_PollEvent(&event)) {
            // Act on 
            switch (event.type) {
            case SDL_QUIT:
                quit = true;
            }
        }

    }

}



int main(int argc, char *argv[])
{
    char msg_buf[256];
    int bpp;
    uint32_t video_flags;
    SDL_Surface *surface;

    atexit(SDL_Quit);
    
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0) {
        sprintf(msg_buf, "Failed to initialize video: %s", SDL_GetError());
        perror(msg_buf);
        exit(EXIT_FAILURE);
    }
    
    SDL_EnableKeyRepeat(SDL_DEFAULT_REPEAT_DELAY ,SDL_DEFAULT_REPEAT_INTERVAL);
    
    /* Determine if display depth should be detected */
    if (SDL_GetVideoInfo()->vfmt->BitsPerPixel <= 8) {
        bpp = 8;
    } else {
        bpp = 32;
    }

    video_flags = SDL_HWSURFACE;
	
    if ((surface = SDL_SetVideoMode(X_RES, Y_RES, bpp, video_flags)) == NULL) {
        sprintf(msg_buf, "Failed to initialize video: %s", SDL_GetError());
        perror(msg_buf);
        exit(EXIT_FAILURE);
    }

    mandelbrot(surface);

    return 0;
}
