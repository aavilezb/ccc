#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define TABLE_SIZE (1U << 15)
#define KEY_COUNT (TABLE_SIZE / 2U)

struct entry
{
    uint64_t key;
    uint64_t value;
};

static volatile uint64_t sink;

static uint64_t mix64(uint64_t value)
{
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

static void insert(struct entry table[TABLE_SIZE], uint64_t key, uint64_t value)
{
    size_t slot = (size_t)key & (TABLE_SIZE - 1U);
    while (table[slot].key)
        slot = (slot + 1U) & (TABLE_SIZE - 1U);
    table[slot].key = key;
    table[slot].value = value;
}

static void run_queries(const struct entry table[TABLE_SIZE],
                        const uint64_t keys[KEY_COUNT],
                        unsigned long work_items)
{
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    uint64_t checksum = 0;
    for (unsigned long i = 0; i < work_items; i++)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        uint64_t key = (i & 1U)
            ? mix64(state | (1ULL << 63))
            : keys[state & (KEY_COUNT - 1U)];
        size_t slot = (size_t)key & (TABLE_SIZE - 1U);
        while (table[slot].key && table[slot].key != key)
            slot = (slot + 1U) & (TABLE_SIZE - 1U);
        if (table[slot].key == key)
            checksum += table[slot].value;
        else
            checksum ^= key;
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
    struct entry *table = calloc(TABLE_SIZE, sizeof(*table));
    uint64_t *keys = malloc(KEY_COUNT * sizeof(*keys));
    if (!table || !keys)
    {
        fputs("failed to allocate hash table\n", stderr);
        free(keys);
        free(table);
        return 1;
    }
    for (size_t i = 0; i < KEY_COUNT; i++)
    {
        keys[i] = mix64(i + 1U);
        if (!keys[i])
            keys[i] = i + 1U;
        insert(table, keys[i], mix64(keys[i]));
    }

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("hash_lookup", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        run_queries(table, keys, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    free(keys);
    free(table);
    return failed ? 1 : 0;
}
