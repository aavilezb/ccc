#ifndef LIBPMC_PATCH_STRESS_OVERFLOW_CHECK_IMPL_H
#define LIBPMC_PATCH_STRESS_OVERFLOW_CHECK_IMPL_H

#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PATCHED_OVERFLOW_CHECK
#error "PATCHED_OVERFLOW_CHECK must be defined"
#endif

#define BUFFER_SIZE (1U << 16)

static volatile uint64_t sink;

static void parse_fields(const unsigned char buffer[BUFFER_SIZE],
                         unsigned long work_items)
{
    uint32_t state = 0x9e3779b9U;
    uint64_t checksum = 0;
    for (unsigned long i = 0; i < work_items; i++)
    {
        state = state * 1664525U + 1013904223U;
        volatile uint32_t supplied_offset = state & (BUFFER_SIZE - 257U);
        volatile uint32_t supplied_length = 1U + ((state >> 16) & 255U);
        uint32_t offset = supplied_offset;
        uint32_t length = supplied_length;
#if PATCHED_OVERFLOW_CHECK
        /* Security update: subtraction-based validation cannot wrap. */
        if (offset > BUFFER_SIZE || length > BUFFER_SIZE - offset)
            continue;
        uint32_t end = offset + length;
#else
        /* Historical check: offset + length could wrap for hostile values. */
        uint32_t end = offset + length;
        if (end > BUFFER_SIZE)
            continue;
#endif
        checksum += buffer[offset] + buffer[end - 1U] + length;
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
    unsigned char *buffer = malloc(BUFFER_SIZE);
    if (!buffer)
        return 1;
    for (size_t i = 0; i < BUFFER_SIZE; i++)
        buffer[i] = (unsigned char)(i * 131U + 17U);

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("overflow_check", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        parse_fields(buffer, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    free(buffer);
    return failed ? 1 : 0;
}

#endif
