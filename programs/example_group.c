#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static volatile uint64_t sink;

static void workload(unsigned long iterations)
{
    uint64_t value = sink + 1U;
    for (unsigned long i = 0; i < iterations; i++)
    {
        value = value * 6364136223846793005ULL + 1442695040888963407ULL;
        if ((value & 31U) == 0)
            value ^= i;
    }
    sink = value;
}

int main(int argc, char **argv)
{
    unsigned long iterations = argc > 1 ? strtoul(argv[1], NULL, 10) : 5000000UL;
    size_t repetitions;
    if (pmc_measurement_repetitions(&repetitions) != 0)
        return 1;
    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";

    pmc_group_t *group = pmc_group_begin_csv("example_workload", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        workload(iterations);

    if (pmc_group_end(group, output) != 0)
    {
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
        return 1;
    }
    return 0;
}
