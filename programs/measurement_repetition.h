#ifndef LIBPMC_MEASUREMENT_REPETITION_H
#define LIBPMC_MEASUREMENT_REPETITION_H

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define PMC_MEASUREMENT_REPETITIONS_ENV "PMC_MEASUREMENT_REPETITIONS"

/*
 * The collector sets PMC_MEASUREMENT_REPETITIONS after its one-invocation
 * scout. Keeping this helper in the workload harness (rather than libpmc)
 * makes the measurement boundary explicit: libpmc controls the counters, and
 * the harness controls how many times the target function is invoked while
 * those counters are enabled.
 */
static int pmc_measurement_repetitions(size_t *repetitions)
{
    const char *text = getenv(PMC_MEASUREMENT_REPETITIONS_ENV);
    if (!text || !*text)
    {
        *repetitions = 1U;
        return 0;
    }

    errno = 0;
    char *end = NULL;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || end == text || *end || parsed == 0 || parsed > SIZE_MAX)
    {
        fprintf(stderr, "%s must be a positive size_t value\n",
                PMC_MEASUREMENT_REPETITIONS_ENV);
        return -1;
    }
    *repetitions = (size_t)parsed;
    return 0;
}

#endif
