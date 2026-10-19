#ifndef LIBPMC_GROUP_PMC_H
#define LIBPMC_GROUP_PMC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define PMC_GROUP_MAX_MEMBERS 3U
#define PMC_GROUP_MAX_EVENTS (1U + PMC_GROUP_MAX_MEMBERS)
#define PMC_GROUP_DEFAULT_RING_BUFFER_PAGES 128U
#define PMC_GROUP_MAX_RING_BUFFER_PAGES (1U << 16)

  typedef struct pmc_group pmc_group_t;

  typedef enum
  {
    PMC_EVENT_RAW = 0,
    PMC_EVENT_HARDWARE = 1,
  } pmc_event_type_t;

  typedef enum
  {
    PMC_SAMPLE_PERIOD = 0,
    PMC_SAMPLE_FREQUENCY = 1,
  } pmc_sample_mode_t;

  typedef struct
  {
    uint32_t index;
    const char *event_name;
    pmc_event_type_t type;
    uint64_t config;
    int exclude_kernel;
    int exclude_hv;
  } pmc_event_config_t;

  typedef struct
  {
    pmc_event_config_t event;
    pmc_sample_mode_t sample_mode;
    uint64_t sample_value;
    int precise_ip;
    unsigned int ring_buffer_pages;
    int pinned;
  } pmc_leader_config_t;

  typedef struct
  {
    uint32_t index;
    const char *event_name;
    uint64_t id;
    uint64_t total_count;
  } pmc_group_event_result_t;

  typedef struct
  {
    uint64_t ip;
    uint32_t pid;
    uint32_t tid;
    uint64_t time;
    uint32_t cpu;
    uint32_t reserved;
    uint64_t period;
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t values[PMC_GROUP_MAX_EVENTS];
  } pmc_group_sample_t;

  /*
   * Result memory is owned by pmc_group_t. Event slot 0 is always the
   * sampling leader.
   */
  typedef struct
  {
    const char *label;
    size_t num_events;
    pmc_group_event_result_t events[PMC_GROUP_MAX_EVENTS];
    pmc_group_sample_t *samples;
    size_t num_samples;
    uint64_t lost_samples;
    uint64_t time_enabled;
    uint64_t time_running;
  } pmc_group_result_t;

  /* Return practical defaults for a sampling leader. */
  pmc_leader_config_t pmc_group_default_leader(const char *event_name,
                                               uint32_t index,
                                               pmc_event_type_t type,
                                               uint64_t config,
                                               uint64_t sample_period);

  /*
   * Load the eight-column group CSV documented in README.md. The single leader
   * row is always included. PMC_MEMBER_INDICES must select one to three member
   * rows, for example PMC_MEMBER_INDICES="4,7,9".
   * PMC_RUN_MODE may be "sampling" (the default) or "scout". Scout mode opens
   * the leader as a counting event, does not map a sample ring, and still
   * returns the final grouped counts. PMC_SAMPLE_PERIOD optionally overrides a
   * period-mode CSV leader for a sampling run.
   * PMC_RING_BUFFER_PAGES optionally overrides the 128-page sampling-ring
   * default and must be a power of two no larger than 65536.
   * PMC_CPU_AFFINITY optionally pins the calling process to one logical CPU.
   * PMC_CONTEXT_MIXER may be "none" (the default) or "mixed_v1". The mixer
   * runs after PMU reset and before PMU enable, so it is not counted.
  */
  pmc_group_t *pmc_group_begin_csv(const char *label, const char *csv_path);

  /* Convenience stop/export/destroy operation. NULL json_path skips export. */
  int pmc_group_end(pmc_group_t *group, const char *json_path);

  /* Thread-local description of the most recent library error. */
  const char *pmc_group_error(void);

#ifdef __cplusplus
}
#endif

#endif
