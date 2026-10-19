#include "pmc.h"
#include "measurement_repetition.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_SIZE (1U << 16)

static volatile uint64_t sink;

static void initialize_input(unsigned char input[INPUT_SIZE])
{
    static const char record[] =
        "{\"user\":12345,\"active\":true,\"score\":98.75,"
        "\"tags\":[\"alpha\",\"beta\",\"gamma\"]}\n";
    const size_t record_size = sizeof(record) - 1U;
    for (size_t offset = 0; offset < INPUT_SIZE; offset++)
        input[offset] = (unsigned char)record[offset % record_size];
}

static void scan_json(const unsigned char input[INPUT_SIZE],
                      unsigned long work_items)
{
    uint64_t checksum = 0;
    unsigned int depth = 0;
    unsigned int in_string = 0;
    unsigned int escaped = 0;
    unsigned int numbers = 0;
    unsigned int records = 0;

    for (unsigned long i = 0; i < work_items; i++)
    {
        unsigned char byte = input[i & (INPUT_SIZE - 1U)];
        checksum = (checksum << 5) - checksum + byte;
        if (in_string)
        {
            if (escaped)
                escaped = 0;
            else if (byte == '\\')
                escaped = 1;
            else if (byte == '"')
                in_string = 0;
            continue;
        }

        if (byte == '"')
            in_string = 1;
        else if (byte == '{' || byte == '[')
            depth++;
        else if ((byte == '}' || byte == ']') && depth)
            depth--;
        else if (byte >= '0' && byte <= '9')
            numbers++;
        else if (byte == '\n')
            records++;
    }
    sink = checksum ^ ((uint64_t)depth << 48) ^
           ((uint64_t)numbers << 24) ^ records;
}

int main(int argc, char **argv)
{
    unsigned long work_items =
        argc > 1 ? strtoul(argv[1], NULL, 10) : 5000000UL;
    size_t repetitions;
    if (pmc_measurement_repetitions(&repetitions) != 0)
        return 1;
    unsigned char *input = malloc(INPUT_SIZE);
    if (!input)
    {
        fputs("failed to allocate JSON input\n", stderr);
        return 1;
    }
    initialize_input(input);

    const char *output = getenv("PMC_OUTPUT_FILE");
    if (!output || !*output)
        output = "pmc_group_results.json";
    pmc_group_t *group = pmc_group_begin_csv("json_scan", NULL);
    for (size_t repetition = 0; repetition < repetitions; repetition++)
        scan_json(input, work_items);

    int failed = pmc_group_end(group, output);
    if (failed)
        fprintf(stderr, "PMC end failed: %s\n", pmc_group_error());
    free(input);
    return failed ? 1 : 0;
}
