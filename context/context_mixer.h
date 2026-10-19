#ifndef LIBPMC_CONTEXT_MIXER_H
#define LIBPMC_CONTEXT_MIXER_H

#include <stddef.h>

typedef struct context_mixer context_mixer_t;

/*
 * Prepare all memory used by a mixer before the final preconditioning step.
 * "none" produces a NULL context and is a successful no-op.
 */
int context_mixer_prepare(const char *name, context_mixer_t **mixer_out,
                          char *error, size_t error_size);

/* Run the prepared, allocation-free preconditioning sequence. */
int context_mixer_run(context_mixer_t *mixer, char *error, size_t error_size);

const char *context_mixer_name(const context_mixer_t *mixer);
void context_mixer_destroy(context_mixer_t *mixer);

#endif
