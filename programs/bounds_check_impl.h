#ifndef LIBPMC_PATCH_STRESS_BOUNDS_CHECK_IMPL_H
#define LIBPMC_PATCH_STRESS_BOUNDS_CHECK_IMPL_H

#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef PATCHED_BOUNDS_CHECK
#error "PATCHED_BOUNDS_CHECK must be defined"
#endif

#define TABLE_SIZE (1U << 12)

static volatile uint64_t sink;

/* Model an index returned by a parser or device. The generated benchmark
 * indices are valid, so the patch does not change functional output. */
static size_t next_untrusted_index(uint64_t *state)
{
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (size_t)(*state & (TABLE_SIZE - 1U));
}

static void process_records(const uint64_t table[TABLE_SIZE],
                            unsigned long work_items)
{
    uint64_t state = 0x243f6a8885a308d3ULL;
    uint64_t checksum = 0;
    for (unsigned long i = 0; i < work_items; i++)
    {
        volatile size_t supplied_index = next_untrusted_index(&state);
        size_t index = supplied_index;
#if PATCHED_BOUNDS_CHECK
        /* Security update: reject an out-of-range parser result before use. */
        if (index >= TABLE_SIZE)
            continue;
#endif
        checksum = (checksum << 7) ^ (checksum >> 3) ^ table[index];
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
    uint64_t *table = malloc(TABLE_SIZE * sizeof(*table));
    if (!table)
        return 1;
    for (size_t i = 0; i < TABLE_SIZE; i++)
        table[i] = (uint64_t)i * 0x9e3779b97f4a7c15ULL;

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("bounds_check", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        process_records(table, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    free(table);
    return failed ? 1 : 0;
}

#endif
