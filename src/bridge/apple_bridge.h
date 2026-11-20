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
#define AMI_MAX_TEXTURE_BINDINGS 16u
#define AMI_MAX_TEXTURE_DIM 16384u
#define AMI_KV_TILE_ROW_ALIGN_F16 32u
#define AMI_MAX_FRAMES_IN_FLIGHT 4u
#define AMI_UNIFORM_STRIDE 256u
#define AMI_UNIFORM_SLOTS_PER_FRAME 2048u
#define AMI_ANE_MAX_DIMS 5u
#define AMI_ANE_MAX_BINDINGS 16u

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
    AMI_ERR_TILE = 10,
    AMI_ERR_ANE = 11,
};

enum {
    AMI_ANE_COMPUTE_CPU_AND_NE = 0,
    AMI_ANE_COMPUTE_ALL = 1,
    AMI_ANE_COMPUTE_CPU_ONLY = 2,
    AMI_ANE_COMPUTE_CPU_AND_GPU = 3,
};

enum {
    AMI_TEX_VIEW_RGBA16F = 0,
    AMI_TEX_VIEW_R16F = 1,
};

enum {
    AMI_KV_TILE_FLAG_NONE = 0u,
    AMI_KV_TILE_FLAG_ALIAS_BUFFER = 1u << 0,
    AMI_KV_TILE_FLAG_MLOCK = 1u << 1,
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

/*
 * Owns one IOSurface-backed CVPixelBuffer (OneComponent16Half), zero-copy
 * RGBA16Float and R16Float MTLTexture views, and an optional zero-copy
 * MTLBuffer alias over the same 16 KiB-aligned pages (ADR-0004).
 */
typedef struct ami_kv_tile ami_kv_tile;

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

/* Binding of a resident KvTile's zero-copy MTLTexture view at `[[texture(i)]]`. */
typedef struct ami_texture_binding {
    const ami_kv_tile *tile;
    uint32_t view; /* AMI_TEX_VIEW_RGBA16F or AMI_TEX_VIEW_R16F */
    uint32_t reserved;
} ami_texture_binding;

/* Metadata and GPU resource handles of an IOSurface-backed KV tile. */
typedef struct ami_kv_tile_info {
    uint32_t row_f16;            /* width in fp16 elements (multiple of 32) */
    uint32_t rows;               /* height in rows (1..16384) */
    uint32_t bytes_per_row;      /* verified multiple of 64 */
    uint32_t has_alias_buffer;   /* 0 or 1 */
    uint32_t is_mlocked;         /* 0 or 1 */
    uint32_t reserved;
    uint64_t data_bytes;         /* rows * bytes_per_row */
    uint64_t alloc_bytes;        /* explicit 16 KiB-aligned allocation size */
    uint64_t surface_alloc_size; /* raw IOSurfaceGetAllocSize(surface) */
    uint64_t rgba_resource_id;   /* RGBA16Float MTLTexture.gpuResourceID._impl */
    uint64_t r16_resource_id;    /* R16Float MTLTexture.gpuResourceID._impl */
    uint64_t alias_gpu_address;  /* MTLBuffer.gpuAddress if alias created, else 0 */
} ami_kv_tile_info;

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
 * Creates an IOSurface-backed fp16 CVPixelBuffer (OneComponent16Half) with
 * `row_f16` (multiple of 32, <= 16384) and `rows` (1..16384), explicit 16 KiB-
 * aligned kIOSurfaceAllocSize, zero-copy RGBA16Float (width = row_f16 / 4) and
 * R16Float (width = row_f16) MTLTexture views, and optional zero-copy MTLBuffer
 * alias (AMI_KV_TILE_FLAG_ALIAS_BUFFER) or mlock (AMI_KV_TILE_FLAG_MLOCK).
 */
ami_status ami_kv_tile_create(ami_device *device, uint32_t row_f16, uint32_t rows,
                              uint32_t flags, ami_kv_tile **out_tile);
void ami_kv_tile_destroy(ami_kv_tile *tile);
ami_status ami_kv_tile_query(const ami_kv_tile *tile, ami_kv_tile_info *out_info);
uint64_t ami_kv_tile_rgba_resource_id(const ami_kv_tile *tile);
uint64_t ami_kv_tile_r16_resource_id(const ami_kv_tile *tile);
/* Borrowed handle owned by `tile`; NULL unless AMI_KV_TILE_FLAG_ALIAS_BUFFER was set. */
ami_buffer *ami_kv_tile_alias_buffer(const ami_kv_tile *tile);

/* CPU lock/unlock for test setup and inspection; never used on the GPU hot path. */
ami_status ami_kv_tile_lock(ami_kv_tile *tile, uint32_t read_only,
                            void **out_base_ptr, size_t *out_bytes_per_row,
                            size_t *out_data_bytes);
ami_status ami_kv_tile_unlock(ami_kv_tile *tile, uint32_t read_only);

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

/* Manages caller buffers and KV tiles in the context's user residency set. */
ami_status ami_compute_make_resident(ami_compute *compute, ami_buffer *const *buffers,
                                     size_t count);
ami_status ami_compute_remove_resident(ami_compute *compute, ami_buffer *const *buffers,
                                       size_t count);
ami_status ami_compute_make_tiles_resident(ami_compute *compute, ami_kv_tile *const *tiles,
                                           size_t count);
ami_status ami_compute_remove_tiles_resident(ami_compute *compute, ami_kv_tile *const *tiles,
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

/*
 * Extended dispatch encoding that also binds `textures[0..texture_count)` to
 * `[[texture(0..texture_count-1)]]` in the frame's Metal 4 argument table.
 */
ami_status ami_compute_encode_dispatch_ex(ami_compute *compute, const ami_pipeline *pipeline,
                                          const ami_buffer_binding *bindings, size_t binding_count,
                                          const ami_texture_binding *textures, size_t texture_count,
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

/* -------------------------------------------------------------------------
 * CoreML / Apple Neural Engine zero-copy IOSurface bridge (ADR-0004, D5, S4)
 * ------------------------------------------------------------------------- */

/* Owns one compiled & loaded CoreML MLModel and its MLComputePlan summary. */
typedef struct ami_ane_model ami_ane_model;

/* Owns one zero-copy MLMultiArray wrapping an IOSurface-backed KvTile. */
typedef struct ami_ane_tensor ami_ane_tensor;

/* Owns preallocated input MLDictionaryFeatureProvider + MLPredictionOptions outputBackings. */
typedef struct ami_ane_plan ami_ane_plan;

typedef struct ami_ane_binding {
    const char *name;
    const ami_ane_tensor *tensor;
} ami_ane_binding;

typedef struct ami_ane_model_info {
    uint32_t compute_units;
    uint32_t total_layers;
    uint32_t ane_preferred_layers;
    uint32_t gpu_preferred_layers;
    uint32_t cpu_preferred_layers;
    uint32_t reserved;
} ami_ane_model_info;

typedef struct ami_ane_predict_info {
    uint32_t outputs_zero_copy_matched;
    uint32_t outputs_total;
} ami_ane_predict_info;

/*
 * Compiles (.mlmodel / .mlpackage) and loads, or directly loads (.mlmodelc),
 * a CoreML model from disk with the requested compute units. Startup only.
 */
ami_status ami_ane_model_load_path(const char *path, uint32_t compute_units,
                                   ami_ane_model **out_model);

/*
 * Compiles and loads a CoreML .mlmodel protobuf directly from in-memory bytes
 * via MLModelAsset (zero temporary disk files). Startup only.
 */
ami_status ami_ane_model_load_spec_data(const void *spec_bytes, size_t spec_len,
                                        uint32_t compute_units,
                                        ami_ane_model **out_model);

void ami_ane_model_destroy(ami_ane_model *model);
ami_status ami_ane_model_query(const ami_ane_model *model, ami_ane_model_info *out_info);

/*
 * Wraps an IOSurface-backed KvTile's CVPixelBuffer (OneComponent16Half) as a
 * zero-copy fp16 MLMultiArray with `shape[0..rank)` (ADR-0004).
 * Requires shape[rank - 1] == tile.row_f16 and product(shape[0..rank-1]) == tile.rows.
 */
ami_status ami_ane_tensor_from_kv_tile(const ami_kv_tile *tile, const uint32_t *shape,
                                       size_t rank, ami_ane_tensor **out_tensor);
void ami_ane_tensor_destroy(ami_ane_tensor *tensor);

/*
 * Preallocates input feature provider and outputBackings MLPredictionOptions
 * so repeated ANE predictions execute with zero per-call dictionary creation.
 */
ami_status ami_ane_plan_create(const ami_ane_binding *inputs, size_t input_count,
                               const ami_ane_binding *outputs, size_t output_count,
                               ami_ane_plan **out_plan);
void ami_ane_plan_destroy(ami_ane_plan *plan);

/*
 * Runs synchronous CoreML prediction using preallocated zero-copy IOSurface
 * inputs and outputBackings. `out_info` (optional) reports zero-copy backing hits.
 */
ami_status ami_ane_model_predict(const ami_ane_model *model, const ami_ane_plan *plan,
                                 ami_ane_predict_info *out_info);

/* One-shot prediction convenience helper (builds a transient plan). */
ami_status ami_ane_model_predict_oneshot(const ami_ane_model *model,
                                         const ami_ane_binding *inputs, size_t input_count,
                                         const ami_ane_binding *outputs, size_t output_count,
                                         ami_ane_predict_info *out_info);

#ifdef __cplusplus
}
#endif

#endif /* AMINFER_APPLE_BRIDGE_H */
