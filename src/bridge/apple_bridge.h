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

typedef int32_t ami_status;

enum {
    AMI_OK = 0,
    AMI_ERR_INVALID_ARGUMENT = 1,
    AMI_ERR_NO_DEVICE = 2,
    AMI_ERR_OUT_OF_MEMORY = 3,
    AMI_ERR_LIBRARY = 4,
};

/* Owns one id<MTLDevice>. */
typedef struct ami_device ami_device;

/* Owns one id<MTLLibrary>. */
typedef struct ami_library ami_library;

typedef struct ami_device_info {
    char name[64];                             /* NUL-terminated, truncated */
    uint64_t recommended_max_working_set_size; /* bytes */
    uint64_t max_buffer_length;                /* bytes */
    uint32_t has_unified_memory;               /* 0 or 1 */
    uint32_t supports_metal4;                  /* 0 or 1 */
    uint32_t apple_gpu_family;                 /* highest MTLGPUFamilyAppleN, 0 if none */
    uint32_t ane_core_count;                   /* 0 when CoreML reports no Neural Engine */
} ami_device_info;

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

#ifdef __cplusplus
}
#endif

#endif /* AMINFER_APPLE_BRIDGE_H */
