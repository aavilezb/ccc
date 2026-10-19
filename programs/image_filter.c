#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IMAGE_WIDTH 256U
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_WIDTH)

static volatile uint64_t sink;

static void initialize_image(uint16_t image[IMAGE_SIZE])
{
    uint32_t state = 0x12345678U;
    for (size_t i = 0; i < IMAGE_SIZE; i++)
    {
        state = state * 1664525U + 1013904223U;
        image[i] = (uint16_t)(state >> 24);
    }
}

static void filter_image(const uint16_t input[IMAGE_SIZE],
                         uint16_t output[IMAGE_SIZE],
                         unsigned long work_items)
{
    uint64_t checksum = 0;
    for (unsigned long i = 0; i < work_items; i++)
    {
        size_t position = i & (IMAGE_SIZE - 1U);
        size_t left = (position - 1U) & (IMAGE_SIZE - 1U);
        size_t right = (position + 1U) & (IMAGE_SIZE - 1U);
        size_t up = (position - IMAGE_WIDTH) & (IMAGE_SIZE - 1U);
        size_t down = (position + IMAGE_WIDTH) & (IMAGE_SIZE - 1U);
        int horizontal = (int)input[right] - (int)input[left];
        int vertical = (int)input[down] - (int)input[up];
        unsigned int magnitude =
            (unsigned int)(horizontal < 0 ? -horizontal : horizontal) +
            (unsigned int)(vertical < 0 ? -vertical : vertical);
        uint16_t pixel = magnitude > 160U ? 255U : (uint16_t)magnitude;
        output[position] = pixel;
        checksum += pixel;
    }
    sink = checksum;
}

int main(int argc, char **argv)
{
    unsigned long work_items =
        argc > 1 ? strtoul(argv[1], NULL, 10) : 5000000UL;
    size_t repetitions;
    if (pmc_measurement_repetitions(&repetitions) != 0)
        return 1;
    uint16_t *input = malloc(IMAGE_SIZE * sizeof(*input));
    uint16_t *output_image = malloc(IMAGE_SIZE * sizeof(*output_image));
    if (!input || !output_image)
    {
        fputs("failed to allocate image buffers\n", stderr);
        free(output_image);
        free(input);
        return 1;
    }
    initialize_image(input);

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("image_filter", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        filter_image(input, output_image, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    free(output_image);
    free(input);
    return failed ? 1 : 0;
}
