#include "context_mixer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MIXED_V1_BUFFER_SIZE (8U * 1024U * 1024U)
#define MIXED_V1_CACHE_LINE 64U
#define MIXED_V1_ALU_ITERATIONS 8192U
#define MIXED_V1_RANDOM_ACCESSES 4096U
#define MIXED_V1_TLB_ACCESSES 2048U
#define MIXED_V1_ICACHE_CALLS 1024U
#define MIXED_V1_BRANCHES 4096U
#define MIXED_V1_SEED UINT64_C(0x123456789abcdef)

struct context_mixer
{
  unsigned char *buffer;
  size_t buffer_size;
  size_t page_size;
};

static volatile uint64_t mixer_sink;

static void mixer_error(char *error, size_t error_size, const char *message)
{
  if (error && error_size)
    snprintf(error, error_size, "%s", message);
}

static uint64_t xorshift64(uint64_t *state)
{
  uint64_t value = *state;
  value ^= value << 13;
  value ^= value >> 7;
  value ^= value << 17;
  *state = value;
  return value;
}

#define DEFINE_MIXER_FUNCTION(number, multiplier, increment)                    \
  __attribute__((noinline)) static uint64_t mixer_function_##number(uint64_t x) \
  {                                                                             \
    x = x * UINT64_C(multiplier) + UINT64_C(increment);                          \
    return x ^ (x >> ((number % 13) + 1));                                       \
  }

DEFINE_MIXER_FUNCTION(0, 3, 1)
DEFINE_MIXER_FUNCTION(1, 5, 7)
DEFINE_MIXER_FUNCTION(2, 11, 13)
DEFINE_MIXER_FUNCTION(3, 17, 19)
DEFINE_MIXER_FUNCTION(4, 23, 29)
DEFINE_MIXER_FUNCTION(5, 31, 37)
DEFINE_MIXER_FUNCTION(6, 41, 43)
DEFINE_MIXER_FUNCTION(7, 47, 53)
DEFINE_MIXER_FUNCTION(8, 59, 61)
DEFINE_MIXER_FUNCTION(9, 67, 71)
DEFINE_MIXER_FUNCTION(10, 73, 79)
DEFINE_MIXER_FUNCTION(11, 83, 89)
DEFINE_MIXER_FUNCTION(12, 97, 101)
DEFINE_MIXER_FUNCTION(13, 103, 107)
DEFINE_MIXER_FUNCTION(14, 109, 113)
DEFINE_MIXER_FUNCTION(15, 127, 131)

typedef uint64_t (*mixer_function_t)(uint64_t);

static mixer_function_t mixer_functions[] = {
    mixer_function_0,  mixer_function_1,  mixer_function_2,  mixer_function_3,
    mixer_function_4,  mixer_function_5,  mixer_function_6,  mixer_function_7,
    mixer_function_8,  mixer_function_9,  mixer_function_10, mixer_function_11,
    mixer_function_12, mixer_function_13, mixer_function_14, mixer_function_15,
};

int context_mixer_prepare(const char *name, context_mixer_t **mixer_out,
                          char *error, size_t error_size)
{
  if (!mixer_out)
  {
    mixer_error(error, error_size, "context mixer output pointer is NULL");
    return -1;
  }
  *mixer_out = NULL;
  if (!name || !*name || strcmp(name, "none") == 0)
    return 0;
  if (strcmp(name, "mixed_v1") != 0)
  {
    mixer_error(error, error_size,
                "unknown context mixer; expected 'none' or 'mixed_v1'");
    return -1;
  }

  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
  {
    mixer_error(error, error_size, "cannot determine system page size");
    return -1;
  }

  context_mixer_t *mixer = calloc(1, sizeof(*mixer));
  if (!mixer)
  {
    mixer_error(error, error_size, "cannot allocate context mixer metadata");
    return -1;
  }
  mixer->buffer_size = MIXED_V1_BUFFER_SIZE;
  mixer->page_size = (size_t)page_size;
  mixer->buffer = malloc(mixer->buffer_size);
  if (!mixer->buffer)
  {
    mixer_error(error, error_size, "cannot allocate context mixer buffer");
    free(mixer);
    return -1;
  }

  /* Initialize and prefault every page before the final mixer run. */
  for (size_t i = 0; i < mixer->buffer_size; i++)
    mixer->buffer[i] = (unsigned char)((i * 131U + 17U) & 0xffU);

  *mixer_out = mixer;
  return 0;
}

int context_mixer_run(context_mixer_t *mixer, char *error, size_t error_size)
{
  (void)error;
  (void)error_size;
  if (!mixer)
    return 0;

  uint64_t state = MIXED_V1_SEED;
  uint64_t accumulator = mixer_sink ^ state;

  /* Establish repeatable integer-pipeline activity. */
  for (size_t i = 0; i < MIXED_V1_ALU_ITERATIONS; i++)
  {
    accumulator = accumulator * UINT64_C(6364136223846793005) +
                  UINT64_C(1442695040888963407);
    accumulator ^= accumulator >> 13;
  }

  /* Sweep an 8 MiB footprint at cache-line granularity. */
  for (size_t i = 0; i < mixer->buffer_size; i += MIXED_V1_CACHE_LINE)
  {
    accumulator += mixer->buffer[i];
    mixer->buffer[i] ^= (unsigned char)accumulator;
  }

  /* Add a deterministic, non-sequential cache access stream. */
  size_t cache_lines = mixer->buffer_size / MIXED_V1_CACHE_LINE;
  for (size_t i = 0; i < MIXED_V1_RANDOM_ACCESSES; i++)
  {
    size_t line = (size_t)(xorshift64(&state) % cache_lines);
    size_t offset = line * MIXED_V1_CACHE_LINE;
    accumulator += mixer->buffer[offset];
    mixer->buffer[offset] ^= (unsigned char)(state >> 8);
  }

  /* Touch deterministic pseudo-random pages to condition address translation. */
  size_t pages = mixer->buffer_size / mixer->page_size;
  for (size_t i = 0; i < MIXED_V1_TLB_ACCESSES; i++)
  {
    size_t page = (size_t)(xorshift64(&state) % pages);
    size_t offset = page * mixer->page_size;
    accumulator += mixer->buffer[offset];
    mixer->buffer[offset] ^= (unsigned char)(state >> 16);
  }

  /* Exercise several distinct instruction and indirect-branch targets. */
  size_t function_count = sizeof(mixer_functions) / sizeof(mixer_functions[0]);
  for (size_t i = 0; i < MIXED_V1_ICACHE_CALLS; i++)
  {
    size_t index = (size_t)(xorshift64(&state) % function_count);
    accumulator = mixer_functions[index](accumulator);
  }

  /* Finish with a repeatable data-dependent conditional-branch history. */
  for (size_t i = 0; i < MIXED_V1_BRANCHES; i++)
  {
    uint64_t value = xorshift64(&state);
    if (value & UINT64_C(1))
      accumulator += value;
    else
      accumulator ^= value;
  }

  mixer_sink = accumulator;
  return 0;
}

const char *context_mixer_name(const context_mixer_t *mixer)
{
  return mixer ? "mixed_v1" : "none";
}

void context_mixer_destroy(context_mixer_t *mixer)
{
  if (!mixer)
    return;
  free(mixer->buffer);
  free(mixer);
}
