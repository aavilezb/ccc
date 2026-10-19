#ifndef LIBPMC_PATCH_STRESS_AUTH_COMPARE_IMPL_H
#define LIBPMC_PATCH_STRESS_AUTH_COMPARE_IMPL_H

#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef CONSTANT_TIME_COMPARE
#error "CONSTANT_TIME_COMPARE must be defined"
#endif

#define TOKEN_SIZE 32U

static volatile uint64_t sink;

static int verify_token(const unsigned char expected[TOKEN_SIZE],
                        const unsigned char supplied[TOKEN_SIZE])
{
#if CONSTANT_TIME_COMPARE
    /* Security update: inspect every byte before deciding equality. */
    unsigned int difference = 0;
    for (size_t i = 0; i < TOKEN_SIZE; i++)
        difference |= (unsigned int)(expected[i] ^ supplied[i]);
    return difference == 0;
#else
    /* Historical implementation returns as soon as a mismatch is found. */
    for (size_t i = 0; i < TOKEN_SIZE; i++)
        if (expected[i] != supplied[i])
            return 0;
    return 1;
#endif
}

static void authenticate(const unsigned char expected[TOKEN_SIZE],
                         const unsigned char matching[TOKEN_SIZE],
                         const unsigned char mismatch[TOKEN_SIZE],
                         unsigned long work_items)
{
    uint64_t accepted = 0;
    for (unsigned long i = 0; i < work_items; i++)
    {
        /* The mismatch is deliberately in the final byte, making this a
         * subtle early-exit versus constant-time comparison. */
        const unsigned char *candidate = (i & 7U) ? mismatch : matching;
        accepted += (uint64_t)verify_token(expected, candidate);
    }
    sink = accepted;
}

int main(int argc, char **argv)
{
    unsigned long work_items =
        argc > 1 ? strtoul(argv[1], NULL, 10) : 5000000UL;
    size_t repetitions;
    if (pmc_measurement_repetitions(&repetitions) != 0)
        return 1;
    unsigned char expected[TOKEN_SIZE];
    unsigned char matching[TOKEN_SIZE];
    unsigned char mismatch[TOKEN_SIZE];
    for (size_t i = 0; i < TOKEN_SIZE; i++)
        expected[i] = matching[i] = mismatch[i] = (unsigned char)(i * 29U + 7U);
    mismatch[TOKEN_SIZE - 1U] ^= 1U;

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("auth_compare", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        authenticate(expected, matching, mismatch, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    return failed ? 1 : 0;
}

#endif
