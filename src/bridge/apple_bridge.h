/*
 * apple_bridge.h — the only ABI between the Zig engine core and Apple
 * frameworks (ADR-0001). Plain C: opaque handles, fixed-width integers, POD
 * structs and status codes; no Objective-C types. Every function returns an
 * ami_status and writes results through out-parameters, which it sets to a
 * defined value (NULL / zero) on failure.
 */
#ifndef AMINFER_APPLE_BRIDGE_H
#define AMINFER_APPLE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AMI_PAGE_SIZE 16384u
#define AMI_MAX_BUFFER_BINDINGS 16u
#define AMI_MAX_FRAMES_IN_FLIGHT 4u
#define AMI_UNIFORM_STRIDE 256u
#define AMI_UNIFORM_SLOTS_PER_FRAME 256u

typedef int32_t ami_status;

enum {
    AMI_OK = 0,
    AMI_ERR_INVALID_ARGUMENT = 1,
    AMI_ERR_NO_DEVICE = 2,
    AMI_ERR_OUT_OF_MEMORY = 3,
    AMI_ERR_LIBRARY = 4,
    AMI_ERR_PIPELINE = 5,
    AMI_ERR_BUFFER = 6,
    AMI_ERR_COMPUTE = 7,
    AMI_ERR_TIMEOUT = 8,
    AMI_ERR_UNSUPPORTED = 9,
};

enum {
    AMI_CONST_U32 = 0,
    AMI_CONST_BOOL = 1,
};

enum {
    AMI_DISPATCH_THREADS = 0,
    AMI_DISPATCH_THREADGROUPS = 1,
};

/* Owns one id<MTLDevice>. */
typedef struct ami_device ami_device;

/* Owns one id<MTLLibrary>. */
typedef struct ami_library ami_library;

/* Owns one zero-copy id<MTLBuffer> wrapping caller-owned wired memory. */
typedef struct ami_buffer ami_buffer;

/* Owns one id<MTLComputePipelineState>. */
typedef struct ami_pipeline ami_pipeline;

/* Owns the reusable multi-frame Metal 4 submission objects (ADR-0011). */
typedef struct ami_compute ami_compute;

typedef struct ami_device_info {
    char name[64];                             /* NUL-terminated, truncated */
    uint64_t recommended_max_working_set_size; /* bytes */
    uint64_t max_buffer_length;                /* bytes */
    uint32_t has_unified_memory;               /* 0 or 1 */
    uint32_t supports_metal4;                  /* 0 or 1 */
    uint32_t apple_gpu_family;                 /* highest MTLGPUFamilyAppleN, 0 if none */
    uint32_t ane_core_count;                   /* 0 when CoreML reports no Neural Engine */
} ami_device_info;

/* Function constant entry for startup pipeline specialization (ADR-0014). */
typedef struct ami_fn_constant {
    uint32_t index;
    uint32_t type; /* AMI_CONST_U32 or AMI_CONST_BOOL */
    uint32_t value;
} ami_fn_constant;

/* Sub-range binding of a resident MTLBuffer at `buffer->gpuAddress + offset`. */
typedef struct ami_buffer_binding {
    const ami_buffer *buffer;
    uint64_t offset;
} ami_buffer_binding;

/* Opens the system default Metal device. */
ami_status ami_device_create(ami_device **out_device);
void ami_device_destroy(ami_device *device);
ami_status ami_device_query(const ami_device *device, ami_device_info *out_info);

/*
 * Loads a .metallib image without copying it; `bytes` must stay valid for the
 * library's lifetime (the engine passes its embedded, static metallib).
 */
ami_status ami_library_create(ami_device *device, const void *bytes, size_t length,
                              ami_library **out_library);
void ami_library_destroy(ami_library *library);

/* Sets *out_present to 1 when the library contains a function named `name`. */
ami_status ami_library_has_function(const ami_library *library, const char *name,
                                    uint32_t *out_present);

/*
 * Wraps caller-owned, 16 KiB-aligned memory in a shared MTLBuffer without
 * copying (ADR-0002). Both `ptr` and `length` must be multiples of
 * AMI_PAGE_SIZE and `length` must be non-zero. The caller keeps the backing
 * pages mapped until the buffer is removed from every residency set and
 * destroyed.
 */
ami_status ami_buffer_wrap(ami_device *device, void *ptr, size_t length,
                           ami_buffer **out_buffer);
void ami_buffer_destroy(ami_buffer *buffer);
uint64_t ami_buffer_gpu_address(const ami_buffer *buffer);

/*
 * Compiles a compute pipeline for `function_name` from `library` using the
 * Metal 4 compiler, optionally specializing with function constants (ADR-0014).
 * Startup only; never call on the hot path.
 */
ami_status ami_pipeline_create(ami_device *device, const ami_library *library,
                               const char *function_name, ami_pipeline **out_pipeline);
ami_status ami_pipeline_create_specialized(ami_device *device, const ami_library *library,
                                           const char *function_name,
                                           const ami_fn_constant *constants,
                                           size_t constant_count,
                                           ami_pipeline **out_pipeline);
void ami_pipeline_destroy(ami_pipeline *pipeline);
uint32_t ami_pipeline_max_threads_per_threadgroup(const ami_pipeline *pipeline);

/*
 * Creates a Metal 4 compute context with `frames_in_flight` reusable allocator /
 * command-buffer / argument-table / uniform-buffer slots (1..AMI_MAX_FRAMES_IN_FLIGHT),
 * a user residency set, and a shared event (ADR-0011).
 */
ami_status ami_compute_create(ami_device *device, ami_compute **out_compute);
ami_status ami_compute_create_with_frames(ami_device *device, uint32_t frames_in_flight,
                                          ami_compute **out_compute);
void ami_compute_destroy(ami_compute *compute);

/* Manages caller buffers in the context's user residency set. */
ami_status ami_compute_make_resident(ami_compute *compute, ami_buffer *const *buffers,
                                     size_t count);
ami_status ami_compute_remove_resident(ami_compute *compute, ami_buffer *const *buffers,
                                       size_t count);
ami_status ami_compute_clear_residency(ami_compute *compute);

/*
 * Multi-dispatch Metal 4 frame encoding (zero heap allocation on the hot path).
 * Call sequence: begin_frame -> 1+ encode_dispatch -> commit_frame -> wait.
 */
ami_status ami_compute_begin_frame(ami_compute *compute, uint64_t timeout_ms,
                                   uint32_t *out_slot);
void ami_compute_abort_frame(ami_compute *compute);

/*
 * Encodes one compute dispatch into the active frame's encoder.
 * `bindings[0..binding_count)` bind to `[[buffer(0..binding_count-1)]]`.
 * When `uniform_len > 0`, `uniform_bytes` (<= AMI_UNIFORM_STRIDE) is copied
 * into the frame's preallocated uniform ring and bound at `[[buffer(binding_count)]]`.
 */
ami_status ami_compute_encode_dispatch(ami_compute *compute, const ami_pipeline *pipeline,
                                       const ami_buffer_binding *bindings, size_t binding_count,
                                       const void *uniform_bytes, size_t uniform_len,
                                       uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                                       uint64_t tg_x, uint64_t tg_y, uint64_t tg_z,
                                       uint32_t dispatch_mode, uint32_t insert_barrier_before);

ami_status ami_compute_commit_frame(ami_compute *compute, uint64_t *out_signal);
ami_status ami_compute_wait(ami_compute *compute, uint64_t signal, uint64_t timeout_ms);

/*
 * Convenience single-dispatch synchronous helper built on the reusable frame
 * path. `buffers[i]` binds to `[[buffer(i)]]`.
 */
ami_status ami_compute_dispatch_1d_sync(ami_compute *compute, const ami_pipeline *pipeline,
                                        ami_buffer *const *buffers, size_t buffer_count,
                                        uint64_t threads, uint64_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* AMINFER_APPLE_BRIDGE_H */
