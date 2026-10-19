#define _GNU_SOURCE
#include "pmc.h"
#include "context/context_mixer.h"

#include <asm/unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

struct pmc_ring
{
  void *base;
  size_t mmap_len;
  size_t data_size;
  size_t page_size;
  struct perf_event_mmap_page *meta;
};




/*
fds — file descriptors from perf_event_open.

fds[0] is the leader; members open with group_fd = fds[0] so they form one perf group.
Used for enable/disable/reset, reading group counts (read(fds[0], ...)), mapping the sample ring, 
and closing on destroy.

ids — kernel event IDs from ioctl(..., PERF_EVENT_IOC_ID, ...).

Stable identifiers for each counter in the group.
Used to match values in group reads and sample records back to the right slot via event_slot_for_id, 
because the kernel returns (value, id) pairs, not fixed array positions.
*/
struct pmc_group
{
  char *label;
  pmc_leader_config_t leader;
  pmc_event_config_t members[PMC_GROUP_MAX_MEMBERS];
  size_t num_members;
  int fds[PMC_GROUP_MAX_EVENTS]; // file descriptors from perf_event_open
  uint64_t ids[PMC_GROUP_MAX_EVENTS]; // kernel event IDs
  struct pmc_ring ring;
  int scout_mode;
  pmc_sample_mode_t configured_sample_mode;
  uint64_t configured_sample_value;
  int started;
  int collected;
  context_mixer_t *context_mixer;
  int cpu_affinity;
  pmc_group_result_t result; //stores the result from read()
};

struct csv_event
{
  pmc_event_config_t event;
  int is_leader;
  pmc_sample_mode_t sample_mode;
  uint64_t sample_value;
};

static _Thread_local char error_buffer[512];

static void pmc_group_destroy(pmc_group_t *group);

static void set_error(const char *fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  vsnprintf(error_buffer, sizeof(error_buffer), fmt, args);
  va_end(args);
}

const char *pmc_group_error(void) { return error_buffer; }

static long perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                            int group_fd, unsigned long flags)
{
  return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int is_power_of_two(unsigned int value)
{
  return value != 0 && (value & (value - 1U)) == 0;
}

// malloc a copy of the event name
static char *duplicate_name(const char *name)
{
  if (!name || !*name)
  {
    set_error("event name must not be empty");
    return NULL;
  }
  char *copy = strdup(name);
  if (!copy)
  {
    set_error("failed to allocate event name");
  }
  return copy;
}

static void free_config_names(struct pmc_group *group)
{
  free((char *)group->leader.event.event_name);
  group->leader.event.event_name = NULL;
  for (size_t i = 0; i < group->num_members; i++)
  {
    free((char *)group->members[i].event_name);
    group->members[i].event_name = NULL;
  }
}


static void free_csv_events(struct csv_event *rows, size_t count)
{
  if (!rows)
    return;
  for (size_t i = 0; i < count; i++)
    free((char *)rows[i].event.event_name);
  free(rows);
}


pmc_leader_config_t pmc_group_default_leader(const char *event_name,
                                             uint32_t index,
                                             pmc_event_type_t type,
                                             uint64_t config,
                                             uint64_t sample_period)
{
  pmc_leader_config_t leader;
  memset(&leader, 0, sizeof(leader));
  leader.event.index = index;
  leader.event.event_name = event_name;
  leader.event.type = type;
  leader.event.config = config;
  leader.event.exclude_kernel = 1;
  leader.event.exclude_hv = 1;
  leader.sample_mode = PMC_SAMPLE_PERIOD; // by default, use period mode. may be overriden later
  leader.sample_value = sample_period;
  leader.precise_ip = 0;
  leader.ring_buffer_pages = PMC_GROUP_DEFAULT_RING_BUFFER_PAGES;
  leader.pinned = 1; // see perf_event_open man page 
  return leader;
}


/*sanity check for the configuration of the group that we just parsed and processed from the CSV file*/
static int validate_config(const char *label, const pmc_leader_config_t *leader,
                           const pmc_event_config_t *members,
                           size_t num_members)
{
  if (!label || !*label || !leader || !leader->event.event_name)
  {
    set_error("label and leader configuration are required");
    return -1;
  }
  if (!members || num_members == 0 || num_members > PMC_GROUP_MAX_MEMBERS)
  {
    set_error("a group requires one to %u counting members",
              PMC_GROUP_MAX_MEMBERS);
    return -1;
  }
  if (leader->sample_value == 0)
  {
    set_error("leader sample period/frequency must be greater than zero");
    return -1;
  }
  if (leader->event.type != PMC_EVENT_RAW &&
      leader->event.type != PMC_EVENT_HARDWARE)
  {
    set_error("leader has an invalid event type");
    return -1;
  }
  if (leader->sample_mode != PMC_SAMPLE_PERIOD &&
      leader->sample_mode != PMC_SAMPLE_FREQUENCY)
  {
    set_error("leader has an invalid sampling mode");
    return -1;
  }
  if (leader->precise_ip < 0 || leader->precise_ip > 3)
  {
    set_error("precise_ip must be between 0 and 3");
    return -1;
  }
  if (!is_power_of_two(leader->ring_buffer_pages) ||
      leader->ring_buffer_pages > PMC_GROUP_MAX_RING_BUFFER_PAGES)
  {
    set_error("ring_buffer_pages must be a nonzero power of two no larger "
              "than %u", PMC_GROUP_MAX_RING_BUFFER_PAGES);
    return -1;
  }
  for (size_t i = 0; i < num_members; i++)
  {
    if (!members[i].event_name || !*members[i].event_name)
    {
      set_error("member %zu has no event name", i);
      return -1;
    }
    if (members[i].type != PMC_EVENT_RAW &&
        members[i].type != PMC_EVENT_HARDWARE)
    {
      set_error("member %zu has an invalid event type", i);
      return -1;
    }
    if (members[i].index == leader->event.index)
    {
      set_error("leader and member indices must be unique");
      return -1;
    }
    for (size_t j = 0; j < i; j++)
    {
      if (members[i].index == members[j].index)
      {
        set_error("duplicate member index %u", members[i].index);
        return -1;
      }
    }
  }
  return 0;
}

static void fill_common_attr(struct perf_event_attr *attr,
                             const pmc_event_config_t *event)
{
  memset(attr, 0, sizeof(*attr));
  attr->size = sizeof(*attr);
  attr->type =
      event->type == PMC_EVENT_RAW ? PERF_TYPE_RAW : PERF_TYPE_HARDWARE;
  attr->config = event->config;
  attr->exclude_kernel = event->exclude_kernel ? 1U : 0U;
  attr->exclude_hv = event->exclude_hv ? 1U : 0U;
  attr->exclude_idle = 1;
}

static int ring_map(int fd, struct pmc_ring *ring, unsigned int data_pages)
{
  // get the page size of the system
  long page_size = sysconf(_SC_PAGESIZE);
  if (page_size <= 0)
  {
    set_error("failed to determine system page size");
    return -1;
  }
  if (data_pages > PMC_GROUP_MAX_RING_BUFFER_PAGES)
  {
    set_error("ring buffer is too large");
    return -1;
  }
  size_t data_size = (size_t)page_size * data_pages; 
  size_t mmap_len = (size_t)page_size + data_size; // unit in bytes, as mmap() takes size in bytes
  void *base = mmap(NULL, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (base == MAP_FAILED)
  {
    set_error("failed to mmap sampling ring: %s", strerror(errno));
    return -1;
  }
  ring->base = base;
  ring->mmap_len = mmap_len;
  ring->data_size = data_size;
  ring->page_size = (size_t)page_size;
  ring->meta = (struct perf_event_mmap_page *)base;
  return 0;
}

static void ring_unmap(struct pmc_ring *ring)
{
  if (ring->base && ring->base != MAP_FAILED)
  {
    munmap(ring->base, ring->mmap_len);
  }
  memset(ring, 0, sizeof(*ring));
}

static void ring_copy(const struct pmc_ring *ring, uint64_t position,
                      void *destination, size_t length)
{
  const char *data = (const char *)ring->base + ring->page_size; //skip the metadata page and stop right at the header portion
  // The perf kernel keeps data_head and position/data_tail increasing 
  // instead of resetting them when they reach the end of the physical buffer
  // so we need to find out the offset in the data portion

  size_t offset = (size_t)(position & (ring->data_size - 1U));
  if (offset + length <= ring->data_size)
  {
    memcpy(destination, data + offset, length);
    return;
  }
  // case when wrap around. won't happen in this project but keep it here for completeness
  size_t first = ring->data_size - offset;
  memcpy(destination, data + offset, first); // copy "first" bytes from the address of data+offset
  memcpy((char *)destination + first, data, length - first); // copy the rest of the bytes from beginning of the header portion
}

/*check if an id from read results matches any id in the group*/
/*if it does, return the slot index of the event*/
/*if it doesn't, return -1*/
static int event_slot_for_id(const struct pmc_group *group, uint64_t id)
{
  size_t num_events = group->num_members + 1U;
  for (size_t i = 0; i < num_events; i++)
  {
    if (group->ids[i] == id)
    {
      return (int)i;
    }
  }
  return -1;
}

static int read_group_counts(struct pmc_group *group, int save_results)
{
  // 1 word for the number of events in the group
  // 1 word for time enabled
  // 1 word for time running
  // 2 words for each event value and id in the group
  // consult read_format section and Reading results section
  // in perf_event_open man page for more details
  uint64_t words[3U + 2U * PMC_GROUP_MAX_EVENTS]; 
  ssize_t bytes = read(group->fds[0], words, sizeof(words));
  if (bytes == 0)
  {
    set_error("pinned perf group could not be scheduled; free PMU counters "
              "or disable the NMI watchdog");
    return -1;
  }
  if (bytes < 0)
  {
    set_error("failed to read perf group: %s", strerror(errno));
    return -1;
  }
  if (bytes < (ssize_t)(3U * sizeof(uint64_t)))
  {
    set_error("short perf group read: %zd bytes", bytes);
    return -1;
  }
  // below we just cross check if the actual read matches the expectation
  uint64_t nr = words[0];
  size_t num_events = group->num_members + 1U; //total number of events = members + leader
  size_t expected = (3U + 2U * (size_t)nr) * sizeof(uint64_t); //expected number of words 
  if (nr != num_events || nr > PMC_GROUP_MAX_EVENTS ||
      (size_t)bytes != expected)
  {
    set_error("unexpected perf group read layout: nr=%llu bytes=%zd",
              (unsigned long long)nr, bytes);
    return -1;
  }
  int seen[PMC_GROUP_MAX_EVENTS] = {0};
  for (size_t i = 0; i < (size_t)nr; i++)
  {
    uint64_t value = words[3U + 2U * i];
    uint64_t id = words[3U + 2U * i + 1U];
    int slot = event_slot_for_id(group, id);
    if (slot < 0 || seen[slot])
    {
      set_error("perf group returned an unknown or duplicate event id");
      return -1;
    }
    seen[slot] = 1;
    if (save_results)
    {
      group->result.events[slot].total_count = value; // record the total count of this event
    }
  }
  if (save_results)
  {
    group->result.time_enabled = words[1];
    group->result.time_running = words[2];
  }
  return 0;
}



/*copy the configurations from CSV parser structures "leader and members" to the group structure so 
we can free the parser structures later*/
static int copy_group_config(struct pmc_group *group,
                             const pmc_leader_config_t *leader,
                             const pmc_event_config_t *members,
                             size_t num_members)
{
  group->leader = *leader; // shallow copy all scalar configuration fields

  // replaces the borrowed pointer with an owned string copy
  group->leader.event.event_name = duplicate_name(leader->event.event_name);
  if (!group->leader.event.event_name)
  {
    return -1;
  }
  group->num_members = num_members;
  for (size_t i = 0; i < num_members; i++)
  {
    group->members[i] = members[i];
    group->members[i].event_name = duplicate_name(members[i].event_name);
    if (!group->members[i].event_name)
    {
      return -1;
    }
  }
  return 0;
}

static pmc_group_t *pmc_group_begin_internal(
    const char *label, const pmc_leader_config_t *leader,
    const pmc_event_config_t *members, size_t num_members, int scout_mode,
    pmc_sample_mode_t configured_sample_mode,
    uint64_t configured_sample_value,
    const char *context_mixer_name,
    int cpu_affinity,
    struct csv_event *owned_csv_rows,
    size_t owned_csv_row_count)
{
  error_buffer[0] = '\0';

  // sanity check
  if (validate_config(label, leader, members, num_members) != 0)
  {
    free_csv_events(owned_csv_rows, owned_csv_row_count);
    return NULL;
  }
  // allocate a group structure to store all the information about the group
  struct pmc_group *group = calloc(1, sizeof(*group));
  if (!group)
  {
    set_error("failed to allocate perf group");
    free_csv_events(owned_csv_rows, owned_csv_row_count);
    return NULL;
  }

  // initialize all file descriptors to -1
  for (size_t i = 0; i < PMC_GROUP_MAX_EVENTS; i++)
  {
    group->fds[i] = -1;
  }
  group->label = strdup(label);
  if (!group->label ||
      copy_group_config(group, leader, members, num_members) != 0)
  {
    if (!error_buffer[0])
      set_error("failed to allocate group metadata");
    free_csv_events(owned_csv_rows, owned_csv_row_count);
    pmc_group_destroy(group);
    return NULL;
  }

  /* The group now owns independent copies of all selected event names. */
  free_csv_events(owned_csv_rows, owned_csv_row_count);


  group->scout_mode = scout_mode;
  group->configured_sample_mode = configured_sample_mode;
  group->configured_sample_value = configured_sample_value;
  group->cpu_affinity = cpu_affinity;

  char mixer_error[160] = {0};
  if (context_mixer_prepare(context_mixer_name, &group->context_mixer,
                            mixer_error, sizeof(mixer_error)) != 0)
  {
    set_error("failed to prepare context mixer: %s", mixer_error);
    pmc_group_destroy(group);
    return NULL;
  }

  // Now we start configuring the structure seen by the Linux kernel
  struct perf_event_attr attr;
  // use leader event information to configure attr
  fill_common_attr(&attr, &group->leader.event);
  attr.disabled = 1;
  attr.pinned = group->leader.pinned ? 1U : 0U;
  if (!group->scout_mode)
  {
    attr.precise_ip = (unsigned int)group->leader.precise_ip;
    // frequency mode
    if (group->leader.sample_mode == PMC_SAMPLE_FREQUENCY)
    {
      attr.freq = 1;
      attr.sample_freq = group->leader.sample_value;
    }
    // period mode
    else
    {
      attr.freq = 0;
      attr.sample_period = group->leader.sample_value;
    }
    attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME |
                       PERF_SAMPLE_CPU | PERF_SAMPLE_PERIOD | PERF_SAMPLE_READ;
    attr.wakeup_events = 1;
  }
  attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
                     PERF_FORMAT_TOTAL_TIME_ENABLED |
                     PERF_FORMAT_TOTAL_TIME_RUNNING;


  /*performance counter values are exposed in two ways: 1. reading from the ringbuffer.
  as every time an overflow occurs, the kernel put the counter value plus other metadata specified 
  by attr.sample_type on ring buffer; 2.  reading the file descriptor,
  which asks the kernel for the current cumulative values of the entire group. the information format 
  is configured by attr.read_format.*/

  group->fds[0] = (int)perf_event_open(&attr, 0, -1, -1, 0);
  if (group->fds[0] < 0)
  {
    set_error("failed to open %s leader '%s': %s",
              group->scout_mode ? "scout" : "sampling",
              group->leader.event.event_name, strerror(errno));
    pmc_group_destroy(group);
    return NULL;
  }
  // get the id of the leader event
  if (ioctl(group->fds[0], PERF_EVENT_IOC_ID, &group->ids[0]) != 0)
  {
    set_error("failed to get leader event id: %s", strerror(errno));
    pmc_group_destroy(group);
    return NULL;
  }

  for (size_t i = 0; i < num_members; i++)
  {
    fill_common_attr(&attr, &group->members[i]);
    attr.disabled = 0;
    attr.pinned = 0;
    size_t slot = i + 1U;
    group->fds[slot] = (int)perf_event_open(&attr, 0, -1, group->fds[0], 0);
    if (group->fds[slot] < 0)
    {
      set_error("failed to add counting member '%s': %s",
                group->members[i].event_name, strerror(errno));
      pmc_group_destroy(group);
      return NULL;
    }
    // get the id of the member event
    if (ioctl(group->fds[slot], PERF_EVENT_IOC_ID, &group->ids[slot]) != 0)
    {
      set_error("failed to get event id for '%s': %s",
                group->members[i].event_name, strerror(errno));
      pmc_group_destroy(group);
      return NULL;
    }
  }

  // 8192 pages
  if (!group->scout_mode &&
      ring_map(group->fds[0], &group->ring,
               group->leader.ring_buffer_pages) != 0)
  {
    pmc_group_destroy(group);
    return NULL;
  }

  group->result.label = group->label;
  group->result.num_events = num_members + 1U;
  group->result.events[0].index = group->leader.event.index;
  group->result.events[0].event_name = group->leader.event.event_name;
  group->result.events[0].id = group->ids[0];
  for (size_t i = 0; i < num_members; i++)
  {
    size_t slot = i + 1U;
    group->result.events[slot].index = group->members[i].index;
    group->result.events[slot].event_name = group->members[i].event_name;
    group->result.events[slot].id = group->ids[slot];
  }

  // reset the group to start counting from 0
  if (ioctl(group->fds[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0)
  {
    set_error("failed to reset perf group: %s", strerror(errno));
    pmc_group_destroy(group);
    return NULL;
  }

  /* Precondition the selected CPU while every PMU event remains disabled. */
  if (context_mixer_run(group->context_mixer, mixer_error,
                        sizeof(mixer_error)) != 0)
  {
    set_error("failed to run context mixer: %s", mixer_error);
    pmc_group_destroy(group);
    return NULL;
  }

  // start measuring
  if (ioctl(group->fds[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0)
  {
    set_error("failed to enable perf group: %s", strerror(errno));
    pmc_group_destroy(group);
    return NULL;
  }

  group->started = 1;
  return group;
}

static int pmc_group_stop(pmc_group_t *group)
{

  if (!group->started)
  {
    return 0;
  }
  if (ioctl(group->fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0)
  {
    set_error("failed to disable perf group: %s", strerror(errno));
    return -1;
  }
  group->started = 0;
  return 0;
}

static int read_u64_field(const struct pmc_ring *ring, uint64_t *cursor,
                          uint64_t record_end, uint64_t *value)
{
  if (record_end - *cursor < sizeof(*value))
    return -1;
  ring_copy(ring, *cursor, value, sizeof(*value));
  *cursor += sizeof(*value);
  return 0;
}

static int read_u32_field(const struct pmc_ring *ring, uint64_t *cursor,
                          uint64_t record_end, uint32_t *value)
{
  if (record_end - *cursor < sizeof(*value))
    return -1;
  ring_copy(ring, *cursor, value, sizeof(*value));
  *cursor += sizeof(*value);
  return 0;
}

/*append a new sample to the result array
if the number of samples we have collected reaches the capacity, we need to resize the array*/
static int append_sample(struct pmc_group *group,
                         const pmc_group_sample_t *sample, size_t *capacity)
{
  // if the number of samples we have collected reaches the capacity, we need to resize the array
  if (group->result.num_samples == *capacity)
  {
    size_t next = *capacity ? *capacity * 2U : 128U;
    pmc_group_sample_t *resized =
        realloc(group->result.samples, next * sizeof(*resized));
    if (!resized)
    {
      set_error("failed to allocate sample storage");
      return -1;
    }
    group->result.samples = resized;
    *capacity = next;
  }

  group->result.samples[group->result.num_samples++] = *sample; // append the new sample to the array
  return 0;
}

static int parse_sample_record(struct pmc_group *group, uint64_t cursor,
                               uint64_t record_end, size_t *capacity)
{
  // consult PERF_RECORD_SAMPLE in perf_event_open man page for the ordering of the fields
  pmc_group_sample_t sample;
  memset(&sample, 0, sizeof(sample));
  if (read_u64_field(&group->ring, &cursor, record_end, &sample.ip) != 0 ||
      read_u32_field(&group->ring, &cursor, record_end, &sample.pid) != 0 ||
      read_u32_field(&group->ring, &cursor, record_end, &sample.tid) != 0 ||
      read_u64_field(&group->ring, &cursor, record_end, &sample.time) != 0 ||
      read_u32_field(&group->ring, &cursor, record_end, &sample.cpu) != 0 ||
      read_u32_field(&group->ring, &cursor, record_end, &sample.reserved) !=
          0 ||
      read_u64_field(&group->ring, &cursor, record_end, &sample.period) != 0)
  {
    set_error("truncated fixed fields in PERF_RECORD_SAMPLE");
    return -1;
  }

  // here we want to read counter values for each event in the group
  // so we include PERF_SAMPLE_READ for attr.sample_type
  // therefore, just like how we read from read(), we read like the following:
  uint64_t nr;
  if (read_u64_field(&group->ring, &cursor, record_end, &nr) != 0 ||
      read_u64_field(&group->ring, &cursor, record_end, &sample.time_enabled) !=
          0 ||
      read_u64_field(&group->ring, &cursor, record_end, &sample.time_running) !=
          0)
  {
    set_error("truncated grouped PERF_SAMPLE_READ header");
    return -1;
  }

  /* 
  sample.time
      = timestamp at which the sample was generated

  sample.time_enabled
      = cumulative duration the perf event was enabled

  sample.time_running
      = cumulative duration it was actually scheduled on the PMU
  */
  size_t num_events = group->num_members + 1U;
  if (nr != num_events || nr > PMC_GROUP_MAX_EVENTS)
  {
    set_error("sample contains %llu counters; expected %zu",
              (unsigned long long)nr, num_events);
    return -1;
  }
  int seen[PMC_GROUP_MAX_EVENTS] = {0};
  for (size_t i = 0; i < (size_t)nr; i++)
  {
    uint64_t value, id;
    if (read_u64_field(&group->ring, &cursor, record_end, &value) != 0 ||
        read_u64_field(&group->ring, &cursor, record_end, &id) != 0)
    {
      set_error("truncated grouped counter value");
      return -1;
    }
    int slot = event_slot_for_id(group, id);
    if (slot < 0 || seen[slot])
    {
      set_error("sample contains an unknown or duplicate event id");
      return -1;
    }
    seen[slot] = 1;
    sample.values[slot] = value;
  }
  if (cursor != record_end)
  {
    set_error("unexpected trailing bytes in PERF_RECORD_SAMPLE");
    return -1;
  }
  return append_sample(group, &sample, capacity);
}



/*
 * perf mmap layout:
 *
 * ring->base
 * ring->meta
 *     |
 *     v
 * +-------------------------+--------------------------------------------+
 * | metadata page           | circular data area                        |
 * |                         |                                            |
 * | data_head: producer     | +----------+-----------+----------+-----+  |
 * | data_tail: consumer     | | header 1 | payload 1 | header 2 | ... |  |
 * | data_offset             | +----------+-----------+----------+-----+  |
 * | data_size               | ^                                          |
 * +-------------------------+-|------------------------------------------+
 *                             data = ring->base + ring->page_size
 *
 * Logical unread byte stream:
 *
 * data_tail -> [header 1][payload 1][header 2][payload 2] ... <- data_head
 *
 * (scenario below won't happen in this project but keep it here for completeness)
 * data_head and data_tail are monotonically increasing byte positions, which means they will keep 
 * increasing and can be larger than the data area and never wrap around.
 * Therefore, we need to find out the offset in the data portion to get the actual position of the head and tail.
 *
 *     offset = head or tail & (ring->data_size - 1)
 *
 * A record can be split between the physical end and beginning of the data
 * area even though it remains contiguous in the logical byte stream. 
 * 
 */

static int consume_ring(struct pmc_group *group)
{
  // The ring buffer starts with a metadata page containing the logical head
  // and tail positions. ring->data_size is the capacity of the data area.
  struct pmc_ring *ring = &group->ring;
  uint64_t head = __atomic_load_n(&ring->meta->data_head, __ATOMIC_ACQUIRE);
  uint64_t tail = ring->meta->data_tail;
  // Tail identifies the beginning of the next unread record header.
  if (head - tail > ring->data_size)
  {
    set_error("sampling ring overrun: unread data exceeds ring capacity");
    return -1;
  }
  size_t capacity = 0;
  while (tail < head)
  {
    // every record starts with a header then followed by the actual payload
    // we read the header first so then we can determine the type of the record
    // each type of record has a different format so we need to handle them differently
    // Every record starts with the same header structure:
    /*
    struct perf_event_header {
    __u32 type;
    __u16 misc;
    __u16 size;
    };
    */
    struct perf_event_header header;
    if (head - tail < sizeof(header))
    {
      set_error("truncated perf record header");
      return -1;
    }
    ring_copy(ring, tail, &header, sizeof(header));
    if (header.size < sizeof(header) || header.size > head - tail)
    {
      set_error("invalid perf record size %u", header.size);
      return -1;
    }
    uint64_t record_end = tail + header.size; // points to the end of the payload
    uint64_t payload = tail + sizeof(header); // points to the beginning of the payload
    if (header.type == PERF_RECORD_SAMPLE)
    {
      if (parse_sample_record(group, payload, record_end, &capacity) != 0)
      {
        return -1;
      }
    }
    else if (header.type == PERF_RECORD_LOST)
    {
      uint64_t id, lost;
      if (read_u64_field(ring, &payload, record_end, &id) != 0 ||
          read_u64_field(ring, &payload, record_end, &lost) != 0)
      {
        set_error("truncated PERF_RECORD_LOST record");
        return -1;
      }
      (void)id;
      group->result.lost_samples += lost;
    }
    // so the current implementation will silently skips every 
    // record type other than PERF_RECORD_SAMPLE and PERF_RECORD_LOST
    // we may need PERF_RECORD_THROTTLE and PERF_RECORD_UNTHROTTLE later
    tail = record_end;
  }
  __atomic_store_n(&ring->meta->data_tail, head, __ATOMIC_RELEASE);
  return 0;
}

/*collect final aggregated results from read() and samples ringbuffer*/
static int pmc_group_collect(pmc_group_t *group)
{
  if (group->collected)
  {
    return 0;
  }
  // retrive final aggregated counts from read() and samples from ring buffer
  if (read_group_counts(group, 1) != 0 ||
      (!group->scout_mode && consume_ring(group) != 0))
  {
    return -1;
  }
  if (group->result.time_running == 0)
  {
    set_error(
        "perf group never ran; verify PMU availability and event constraints");
    return -1;
  }
  group->collected = 1;
  return 0;
}

static void json_string(FILE *file, const char *value)
{
  fputc('"', file);
  for (const unsigned char *p = (const unsigned char *)value; *p; p++)
  {
    switch (*p)
    {
    case '"':
      fputs("\\\"", file);
      break;
    case '\\':
      fputs("\\\\", file);
      break;
    case '\b':
      fputs("\\b", file);
      break;
    case '\f':
      fputs("\\f", file);
      break;
    case '\n':
      fputs("\\n", file);
      break;
    case '\r':
      fputs("\\r", file);
      break;
    case '\t':
      fputs("\\t", file);
      break;
    default:
      if (*p < 0x20)
        fprintf(file, "\\u%04x", *p);
      else
        fputc(*p, file);
    }
  }
  fputc('"', file);
}

static void write_measurement(FILE *file, const struct pmc_group *group)
{
  const pmc_group_result_t *result = &group->result;
  fputs("    {\n      \"label\": ", file);
  json_string(file, result->label);
  fprintf(file, ",\n      \"run_mode\": \"%s\"",
          group->scout_mode ? "scout" : "sampling");
  fputs(",\n      \"context_mixer\": ", file);
  json_string(file, context_mixer_name(group->context_mixer));
  if (group->cpu_affinity >= 0)
    fprintf(file, ",\n      \"cpu_affinity\": %d", group->cpu_affinity);
  else
    fputs(",\n      \"cpu_affinity\": null", file);
  fputs(",\n      \"leader\": {\n        \"index\": ", file);
  fprintf(file, "%u,\n        \"event_name\": ", result->events[0].index);
  json_string(file, result->events[0].event_name);
  fprintf(file,
          ",\n        \"configured_sample_mode\": \"%s\",\n"
          "        \"configured_sample_value\": %llu,\n"
          "        \"sample_mode\": \"%s\",\n"
          "        \"sample_value\": %llu,\n"
          "        \"ring_buffer_pages\": %u\n      },\n",
          group->configured_sample_mode == PMC_SAMPLE_FREQUENCY ? "frequency"
                                                                 : "period",
          (unsigned long long)group->configured_sample_value,
          group->scout_mode
              ? "counting"
              : (group->leader.sample_mode == PMC_SAMPLE_FREQUENCY
                     ? "frequency"
                     : "period"),
          group->scout_mode ? 0ULL
                            : (unsigned long long)group->leader.sample_value,
          group->scout_mode ? 0U : group->leader.ring_buffer_pages);

  fputs("      \"members\": [\n", file);
  for (size_t i = 1; i < result->num_events; i++)
  {
    fprintf(file, "        {\"index\": %u, \"event_name\": ",
            result->events[i].index);
    json_string(file, result->events[i].event_name);
    fprintf(file, ", \"total_count\": %llu}",
            (unsigned long long)result->events[i].total_count);
    fputs(i + 1U < result->num_events ? ",\n" : "\n", file);
  }
  fprintf(file,
          "      ],\n      \"leader_total_count\": %llu,\n"
          "      \"time_enabled\": %llu,\n"
          "      \"time_running\": %llu,\n"
          "      \"lost_samples\": %llu,\n"
          "      \"num_samples\": %zu,\n"
          "      \"samples\": [\n",
          (unsigned long long)result->events[0].total_count,
          (unsigned long long)result->time_enabled,
          (unsigned long long)result->time_running,
          (unsigned long long)result->lost_samples, result->num_samples);

  for (size_t i = 0; i < result->num_samples; i++)
  {
    const pmc_group_sample_t *sample = &result->samples[i];
    fprintf(file,
            "        {\"ip\": \"0x%llx\", \"pid\": %u, "
            "\"tid\": %u, \"time\": %llu, \"cpu\": %u, "
            "\"period\": %llu, \"time_enabled\": %llu, "
            "\"time_running\": %llu, \"counts\": {",
            (unsigned long long)sample->ip, sample->pid, sample->tid,
            (unsigned long long)sample->time, sample->cpu,
            (unsigned long long)sample->period,
            (unsigned long long)sample->time_enabled,
            (unsigned long long)sample->time_running);
    for (size_t j = 0; j < result->num_events; j++)
    {
      fprintf(file, "\"%u\": %llu", result->events[j].index,
              (unsigned long long)sample->values[j]);
      if (j + 1U < result->num_events)
        fputs(", ", file);
    }
    fputs("}}", file);
    fputs(i + 1U < result->num_samples ? ",\n" : "\n", file);
  }
  fputs("      ]\n    }", file);
}

static int append_json_document(const char *path,
                                const struct pmc_group *group)
{
  int fd = open(path, O_RDWR | O_CREAT | O_CLOEXEC, 0644);
  if (fd < 0)
  {
    set_error("failed to open JSON output '%s': %s", path, strerror(errno));
    return -1;
  }
  if (flock(fd, LOCK_EX) != 0)
  {
    set_error("failed to lock JSON output: %s", strerror(errno));
    close(fd);
    return -1;
  }
  struct stat stat_buffer;
  if (fstat(fd, &stat_buffer) != 0)
  {
    set_error("failed to inspect JSON output: %s", strerror(errno));
    close(fd);
    return -1;
  }

  off_t append_position = 0;
  int existing = stat_buffer.st_size > 0;
  if (existing)
  {
    char *content = malloc((size_t)stat_buffer.st_size + 1U);
    if (!content)
    {
      set_error("failed to allocate JSON validation buffer");
      close(fd);
      return -1;
    }
    ssize_t got = pread(fd, content, (size_t)stat_buffer.st_size, 0);
    if (got != stat_buffer.st_size)
    {
      set_error("failed to read existing JSON output");
      free(content);
      close(fd);
      return -1;
    }
    content[got] = '\0';
    if (!strstr(content, "\"schema_version\": 2"))
    {
      set_error("existing output is not a libpmc group schema-version-2 file");
      free(content);
      close(fd);
      return -1;
    }
    char *closing = strrchr(content, ']');
    if (!closing)
    {
      set_error("existing JSON output has no measurements-array terminator");
      free(content);
      close(fd);
      return -1;
    }
    append_position = (off_t)(closing - content);
    free(content);
    if (ftruncate(fd, append_position) != 0 ||
        lseek(fd, append_position, SEEK_SET) < 0)
    {
      set_error("failed to prepare JSON output for append: %s",
                strerror(errno));
      close(fd);
      return -1;
    }
  }

  FILE *file = fdopen(fd, "r+");
  if (!file)
  {
    set_error("failed to create JSON output stream: %s", strerror(errno));
    close(fd);
    return -1;
  }
  if (!existing)
  {
    fputs("{\n  \"schema_version\": 2,\n  \"measurements\": [\n", file);
  }
  else
  {
    fputs(",\n", file);
  }
  write_measurement(file, group);
  fputs("\n  ]\n}\n", file);
  int failed = ferror(file) || fflush(file) != 0 || fsync(fd) != 0;
  if (failed)
  {
    set_error("failed while writing JSON output: %s", strerror(errno));
  }
  fclose(file);
  return failed ? -1 : 0;
}

static int pmc_group_export_json(pmc_group_t *group, const char *json_path)
{
  if (!json_path || !*json_path)
  {
    set_error("JSON path is required");
    return -1;
  }
  if (pmc_group_collect(group) != 0)
  {
    return -1;
  }
  return append_json_document(json_path, group);
}

int pmc_group_end(pmc_group_t *group, const char *json_path)
{
  if (!group)
  {
    /* Preserve the actionable begin/open/mmap error when pmc begin returned group = NULL. */
    if (!error_buffer[0])
      set_error("group is NULL");
    return -1;
  }
  int result = pmc_group_stop(group);
  if (result == 0 && json_path)
  {
    result = pmc_group_export_json(group, json_path);
  }
  pmc_group_destroy(group);
  return result;
}

static void pmc_group_destroy(pmc_group_t *group)
{
  if (!group)
    return;
  if (group->started && group->fds[0] >= 0)
  {
    ioctl(group->fds[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
  }
  ring_unmap(&group->ring);
  for (size_t i = 0; i < PMC_GROUP_MAX_EVENTS; i++)
  {
    if (group->fds[i] >= 0)
      close(group->fds[i]);
  }
  free(group->result.samples);
  context_mixer_destroy(group->context_mixer);
  free_config_names(group);
  free(group->label);
  free(group);
}


/* trim strips leading and trailing whitespace from a mutable C string, 
in place, and returns a pointer to the start of the trimmed text. 
Example: " hello\n" becomes a pointer into that same buffer that reads as "hello".
*/
static char *trim(char *text)
{
  while (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n')
    text++;
  size_t length = strlen(text);
  while (length && (text[length - 1U] == ' ' || text[length - 1U] == '\t' ||
                    text[length - 1U] == '\r' || text[length - 1U] == '\n'))
  {
    text[--length] = '\0';
  }
  return text;
}

/*split_csv splits a CSV line into fields, using commas as delimiters.
Each CSV row has 8 columns: index, event_name, event_number, umask, mode, sample_period, type, role.
Store each individual column value in the fields array.
*/
static int split_csv(char *line, char **fields, size_t expected)
{
  size_t count = 0;
  char *save = NULL;
  for (char *field = strtok_r(line, ",", &save); field;
       field = strtok_r(NULL, ",", &save))
  {
    if (count == expected)
      return -1;
    fields[count++] = trim(field);
  }
  return count == expected ? 0 : -1;
}

/*based on the name of the four events that can be put on fixed PMCs, set the correct config value
so linux kernel can recognize it*/
static int hardware_config_for_name(const char *name, uint64_t *config)
{
  if (strstr(name, "INST_RETIRED") || strstr(name, "INSTRUCTIONS"))
  {
    *config = PERF_COUNT_HW_INSTRUCTIONS;
  }
  else if (strstr(name, "REF_CYCLES") || strstr(name, "REF_TSC"))
  {
    *config = PERF_COUNT_HW_REF_CPU_CYCLES;
  }
  else if (strstr(name, "CPU_CLK_UNHALTED") || strstr(name, "CPU_CYCLES") ||
           strstr(name, "UNHALTED"))
  {
    *config = PERF_COUNT_HW_CPU_CYCLES;
  }
  else
  {
    set_error("unsupported hardware/fixed event name '%s'", name);
    return -1;
  }
  return 0;
}

static int parse_u32(const char *text, uint32_t *value)
{
  char *end;
  errno = 0;
  unsigned long parsed = strtoul(text, &end, 0);
  if (errno || *text == '\0' || *end != '\0' || parsed > UINT32_MAX)
    return -1;
  *value = (uint32_t)parsed;
  return 0;
}

static int parse_u64(const char *text, uint64_t *value)
{
  char *end;
  errno = 0;
  unsigned long long parsed = strtoull(text, &end, 0);
  if (errno || *text == '\0' || *end != '\0')
    return -1;
  *value = (uint64_t)parsed;
  return 0;
}

static int apply_cpu_affinity_environment(int *cpu_out)
{
  *cpu_out = -1;
  const char *environment = getenv("PMC_CPU_AFFINITY");
  if (!environment || !*environment)
    return 0;

  uint32_t cpu;
  if (parse_u32(environment, &cpu) != 0 || cpu >= CPU_SETSIZE)
  {
    set_error("PMC_CPU_AFFINITY must be a valid nonnegative CPU number");
    return -1;
  }

  cpu_set_t allowed;
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0)
  {
    set_error("failed to read CPU affinity: %s", strerror(errno));
    return -1;
  }
  if (!CPU_ISSET((int)cpu, &allowed))
  {
    set_error("CPU %u is outside the process's allowed affinity mask", cpu);
    return -1;
  }

  cpu_set_t selected;
  CPU_ZERO(&selected);
  CPU_SET((int)cpu, &selected);
  if (sched_setaffinity(0, sizeof(selected), &selected) != 0)
  {
    set_error("failed to pin process to CPU %u: %s", cpu, strerror(errno));
    return -1;
  }
  *cpu_out = (int)cpu;
  return 0;
}

/*ring buffer size can also be overridden by the environment variable PMC_RING_BUFFER_PAGES
collect_pmc_groups.py script will set this environment variable to the desired ring buffer 
size.(usually 8192 pages is big enough)*/
static int apply_ring_buffer_pages_environment(pmc_leader_config_t *leader)
{
  const char *environment = getenv("PMC_RING_BUFFER_PAGES");
  if (!environment || !*environment)
    return 0;

  uint32_t pages;
  if (parse_u32(environment, &pages) != 0 || !is_power_of_two(pages) ||
      pages > PMC_GROUP_MAX_RING_BUFFER_PAGES)
  {
    set_error("PMC_RING_BUFFER_PAGES must be a nonzero power of two no larger "
              "than %u", PMC_GROUP_MAX_RING_BUFFER_PAGES);
    return -1;
  }
  leader->ring_buffer_pages = pages;
  return 0;
}

/*check if is scout mode or the actual sampling mode*/
static int parse_run_mode_environment(int *scout_mode)
{
  const char *environment = getenv("PMC_RUN_MODE");
  // if PM_RUN_MODE is not provided or PMC_RUN_MODE is sampling, set scout_mode to 0
  if (!environment || !*environment || !strcasecmp(environment, "sampling"))
  {
    *scout_mode = 0;
    return 0;
  }
  // if PM_RUN_MODE is scout, set scout_mode to 1
  if (!strcasecmp(environment, "scout"))
  {
    *scout_mode = 1;
    return 0;
  }
  set_error("PMC_RUN_MODE must be 'scout' or 'sampling'");
  return -1;
}


/*we need environment variable PMC_SAMPLE_PERIOD to override the sample period specified in the CSV file because
of the adaptive sampling algorithm. collect_pmc_groups.py script will set this environment variable to the 
desired sample period based on golden counts from the scout mode.*/
static int apply_sample_period_environment(pmc_leader_config_t *leader)
{
  const char *environment = getenv("PMC_SAMPLE_PERIOD");
  if (!environment || !*environment)
    return 0;
  if (leader->sample_mode != PMC_SAMPLE_PERIOD)
  {
    set_error("PMC_SAMPLE_PERIOD cannot override a leader that is in sampling_freq mode");
    return -1;
  }
  uint64_t period;
  if (parse_u64(environment, &period) != 0 || period == 0)
  {
    set_error("PMC_SAMPLE_PERIOD must be a positive integer");
    return -1;
  }
  leader->sample_value = period;
  return 0;
}

/*parse values from one row from the CSV file into a storage structure "row" */
static int parse_csv_event(char **fields, struct csv_event *row,
                           size_t line_number)
{
  uint32_t event_number, umask;
  uint64_t sample_value;
  memset(row, 0, sizeof(*row));
  if (parse_u32(fields[0], &row->event.index) != 0 ||
      parse_u32(fields[2], &event_number) != 0 ||
      parse_u32(fields[3], &umask) != 0 ||
      parse_u64(fields[5], &sample_value) != 0)
  {
    set_error("invalid numeric field on CSV line %zu", line_number);
    return -1;
  }
  row->event.event_name = strdup(fields[1]);
  if (!row->event.event_name)
  {
    set_error("failed to allocate CSV event name");
    return -1;
  }
  // set default values to exclude_kernel and exclude_hv
  row->event.exclude_kernel = 1;
  row->event.exclude_hv = 1;
  if (!strcasecmp(fields[6], "raw"))
  {
    row->event.type = PMC_EVENT_RAW;
    // for raw type, use the event number and umask to set the config value
    // see perf_event_open man page for more details
    row->event.config = (uint64_t)event_number | ((uint64_t)umask << 8U);
  }
  else if (!strcasecmp(fields[6], "fixed") ||
           !strcasecmp(fields[6], "hardware"))
  {
    row->event.type = PMC_EVENT_HARDWARE;
    if (hardware_config_for_name(fields[1], &row->event.config) != 0)
      return -1;
  }
  else
  {
    set_error("invalid event type '%s' on CSV line %zu", fields[6],
              line_number);
    return -1;
  }
  if (!strcasecmp(fields[7], "leader"))
  {
    row->is_leader = 1;
    row->sample_value = sample_value;
    if (!strcasecmp(fields[4], "sampling"))
    {
      row->sample_mode = PMC_SAMPLE_PERIOD;
    }
    else if (!strcasecmp(fields[4], "sampling_freq"))
    {
      row->sample_mode = PMC_SAMPLE_FREQUENCY;
    }
    else
    {
      set_error("leader on CSV line %zu must use sampling or sampling_freq",
                line_number);
      return -1;
    }
    if (!sample_value)
    {
      set_error("leader on CSV line %zu has a zero sample value", line_number);
      return -1;
    }
  }
  else if (!strcasecmp(fields[7], "member"))
  {
    if (strcasecmp(fields[4], "counting"))
    {
      set_error("member on CSV line %zu must use counting mode", line_number);
      return -1;
    }
  }
  else
  {
    set_error("invalid role '%s' on CSV line %zu", fields[7], line_number);
    return -1;
  }
  return 0;
}


/*load and store all rows from the csv file into an array*/
static int load_csv(const char *path, struct csv_event **rows_out,
                    size_t *count_out)
{
  FILE *file = fopen(path, "r");
  if (!file)
  {
    set_error("failed to open group CSV '%s': %s", path, strerror(errno));
    return -1;
  }
  struct csv_event *rows = NULL;
  size_t count = 0, capacity = 0, line_number = 0;
  char *line = NULL;
  size_t line_capacity = 0;
  int header_seen = 0;

  // *line and *line_capacity will contain buffer address and allocated size
  while (getline(&line, &line_capacity, file) >= 0)
  {
    line_number++;
    char *clean = trim(line);
    // if the line is empty or starts with a #, skip it
    if (!*clean || *clean == '#')
      continue;
    // if the header has not been seen, check if the line is the header
    if (!header_seen)
    {
      const char *expected =
          "index,event_name,event_number,umask,mode,sample_period,type,role";
      if (strcmp(clean, expected))
      {
        set_error("CSV header must be: %s", expected);
        goto fail;
      }
      header_seen = 1;
      continue;
    }
    char *fields[8];
    if (split_csv(clean, fields, 8) != 0)
    {
      set_error("CSV line %zu must contain exactly eight fields", line_number);
      goto fail;
    }
    // allocate memory for rows
    if (count == capacity)
    {
      size_t next = capacity ? capacity * 2U : 64U;
      struct csv_event *resized = realloc(rows, next * sizeof(*resized));
      if (!resized)
      {
        set_error("failed to allocate CSV event table");
        goto fail;
      }
      rows = resized;
      capacity = next;
    }
    if (parse_csv_event(fields, &rows[count], line_number) != 0)
    {
      free((char *)rows[count].event.event_name);
      rows[count].event.event_name = NULL;
      goto fail;
    }
    count++;
  }

  if (ferror(file) || !header_seen || count < 2U)
  {
    set_error("group CSV must contain a header, one leader, and members");
    goto fail;
  }
  free(line);
  fclose(file);
  *rows_out = rows;
  *count_out = count;
  return 0;

fail:
  free(line);
  fclose(file);
  free_csv_events(rows, count);
  return -1;
}


/*parse event indices from the PMC_MEMBER_INDICES environment variable and count and store 
the number of indices from the environment variable*/

static int parse_selected_indices(uint32_t indices[PMC_GROUP_MAX_MEMBERS],
                                  size_t *count_out)
{
  const char *environment = getenv("PMC_MEMBER_INDICES");
  if (!environment || !*environment)
  {
    set_error("PMC_MEMBER_INDICES must select one to three member indices");
    return -1;
  }
  char *copy = strdup(environment);
  if (!copy)
  {
    set_error("failed to allocate member-index parser");
    return -1;
  }
  size_t count = 0;
  char *save = NULL;
  for (char *token = strtok_r(copy, ",", &save); token;
       token = strtok_r(NULL, ",", &save))
  {
    // put the trimmed token into the indices array
    // &indices[count] is a pointer to the count-th element of the indices array
    if (count == PMC_GROUP_MAX_MEMBERS ||
        parse_u32(trim(token), &indices[count]) != 0)
    {
      set_error("PMC_MEMBER_INDICES must contain one to three valid integers");
      free(copy);
      return -1;
    }

    // validates uniqueness inside the requested member-index list
    for (size_t i = 0; i < count; i++)
    {
      if (indices[i] == indices[count])
      {
        set_error("PMC_MEMBER_INDICES contains duplicate index %u",
                  indices[count]);
        free(copy);
        return -1;
      }
    }
    count++;
  }
  free(copy);
  if (count == 0)
  {
    set_error("PMC_MEMBER_INDICES did not contain any indices");
    return -1;
  }
  *count_out = count;
  return 0;
}

pmc_group_t *pmc_group_begin_csv(const char *label, const char *csv_path)
{
  error_buffer[0] = '\0';
  int cpu_affinity = -1;
  if (apply_cpu_affinity_environment(&cpu_affinity) != 0)
    return NULL;
  const char *mixer_name = getenv("PMC_CONTEXT_MIXER");
  if (!mixer_name || !*mixer_name)
    mixer_name = "none";

  const char *path = csv_path;
  // a short C idiom for "path is missing or empty"
  // if csv_path is not provided, use PM_EVENTS_CSV environment variable
  if (!path || !*path){
    path = getenv("PMC_EVENTS_CSV");
  }
  // if PM_EVENTS_CSV is not provided, use default path
  if (!path || !*path) {
    path = "pmc_group_events.csv";
  }
  struct csv_event *rows = NULL;
  size_t row_count = 0;
  uint32_t selected[PMC_GROUP_MAX_MEMBERS];
  size_t selected_count = 0;
  int scout_mode = 0;
  if (parse_run_mode_environment(&scout_mode) != 0 ||
      parse_selected_indices(selected, &selected_count) != 0 ||
      load_csv(path, &rows, &row_count) != 0)
  {
    return NULL;
  }

  struct csv_event *leader_row = NULL;
  size_t leader_count = 0;
  for (size_t i = 0; i < row_count; i++)
  {
    // find the leader in the CSV file
    if (rows[i].is_leader)
    {
      leader_row = &rows[i];
      leader_count++;
    }
    // make sure there are no duplicate event indices in the CSV file
    for (size_t j = 0; j < i; j++)
    {
      if (rows[i].event.index == rows[j].event.index)
      {
        set_error("duplicate event index %u in group CSV", rows[i].event.index);
        free_csv_events(rows, row_count);
        return NULL;
      }
    }
  }
  // make sure there is exactly one leader in the CSV file
  if (leader_count != 1U)
  {
    set_error("group CSV must define exactly one leader; found %zu",
              leader_count);
    free_csv_events(rows, row_count);
    return NULL;
  }

  // configure members
  pmc_event_config_t members[PMC_GROUP_MAX_MEMBERS];
  memset(members, 0, sizeof(members));
  // selected array stores the event indices from environment variable
  // selected_count is the number of indices from the environment variable
  for (size_t i = 0; i < selected_count; i++)
  {
    struct csv_event *match = NULL;
    // detect if a event index from the environment variable presents in the CSV file too
    for (size_t j = 0; j < row_count; j++)
    {
      if (rows[j].event.index == selected[i])
      {
        // if found, store the pointer in match
        match = &rows[j];
        break;
      }
    }
    if (!match)
    {
      set_error("selected member index %u is not in the group CSV",
                selected[i]);
      free_csv_events(rows, row_count);
      return NULL;
    }
    if (match->is_leader)
    {
      set_error("leader index %u must not appear in PMC_MEMBER_INDICES",
                selected[i]);
      free_csv_events(rows, row_count);
      return NULL;
    }
    // we only keep the common hardware-event configuration not the whole csv_event structure
    // we don't need sample_mode and sample_value from csv_event structures as we are processing
    // members. Leader will be processed separately down below.
    members[i] = match->event;
  }
  
  // Now we configure leader 
  pmc_leader_config_t leader = pmc_group_default_leader(
      leader_row->event.event_name, leader_row->event.index,
      leader_row->event.type, leader_row->event.config,
      leader_row->sample_value);
  leader.event.exclude_kernel = leader_row->event.exclude_kernel;
  leader.event.exclude_hv = leader_row->event.exclude_hv;
  leader.sample_mode = leader_row->sample_mode;

  // read environment variables from outside (collect_pmc_groups.py)
  if ((!scout_mode && apply_sample_period_environment(&leader) != 0) ||
      (!scout_mode && apply_ring_buffer_pages_environment(&leader) != 0))
  {
    free_csv_events(rows, row_count);
    return NULL;
  }
  pmc_group_t *group = pmc_group_begin_internal(
      label, &leader, members, selected_count, scout_mode,
      leader_row->sample_mode, leader_row->sample_value, mixer_name,
      cpu_affinity, rows, row_count);

  return group;
}
