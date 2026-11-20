// apple_bridge.m — Objective-C (ARC) implementation of apple_bridge.h.
//
// Handles own their Objective-C objects through __bridge_retained pointers and
// release them in the matching *_destroy function.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdlib.h>
#include <string.h>

#include "apple_bridge.h"

struct ami_device {
    void *mtl_device; // id<MTLDevice>, retained
};

struct ami_library {
    void *mtl_library; // id<MTLLibrary>, retained
};

static id<MTLDevice> device_of(const ami_device *device) {
    return (__bridge id<MTLDevice>)device->mtl_device;
}

static uint32_t highest_apple_family(id<MTLDevice> device) {
    // MTLGPUFamilyApple1 = 1001 ... probe upward-compatible values numerically
    // so that families newer than the SDK enum are still reported.
    for (NSInteger family = 1020; family >= 1001; family--) {
        if ([device supportsFamily:(MTLGPUFamily)family]) {
            return (uint32_t)(family - 1000);
        }
    }
    return 0;
}

static uint32_t neural_engine_cores(void) {
    uint32_t cores = 0;
    for (id compute_device in MLAllComputeDevices()) {
        if ([compute_device isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
            cores = (uint32_t)((MLNeuralEngineComputeDevice *)compute_device).totalCoreCount;
        }
    }
    return cores;
}

ami_status ami_device_create(ami_device **out_device) {
    if (out_device == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_device = NULL;
    id<MTLDevice> mtl_device = MTLCreateSystemDefaultDevice();
    if (mtl_device == nil) {
        return AMI_ERR_NO_DEVICE;
    }
    ami_device *device = calloc(1, sizeof *device);
    if (device == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }
    device->mtl_device = (__bridge_retained void *)mtl_device;
    *out_device = device;
    return AMI_OK;
}

void ami_device_destroy(ami_device *device) {
    if (device == NULL) {
        return;
    }
    CFRelease(device->mtl_device);
    free(device);
}

ami_status ami_device_query(const ami_device *device, ami_device_info *out_info) {
    if (device == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_info, 0, sizeof *out_info);
    @autoreleasepool {
        id<MTLDevice> mtl_device = device_of(device);
        const char *name = mtl_device.name.UTF8String;
        if (name != NULL) {
            strlcpy(out_info->name, name, sizeof out_info->name);
        }
        out_info->recommended_max_working_set_size = mtl_device.recommendedMaxWorkingSetSize;
        out_info->max_buffer_length = (uint64_t)mtl_device.maxBufferLength;
        out_info->has_unified_memory = mtl_device.hasUnifiedMemory ? 1 : 0;
        out_info->supports_metal4 =
            [mtl_device respondsToSelector:@selector(newMTL4CommandQueue)] ? 1 : 0;
        out_info->apple_gpu_family = highest_apple_family(mtl_device);
        out_info->ane_core_count = neural_engine_cores();
    }
    return AMI_OK;
}

ami_status ami_library_create(ami_device *device, const void *bytes, size_t length,
                              ami_library **out_library) {
    if (out_library == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_library = NULL;
    if (device == NULL || bytes == NULL || length == 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        // The no-op destructor keeps dispatch from copying the caller's bytes.
        dispatch_data_t data = dispatch_data_create(bytes, length, NULL, ^{
        });
        if (data == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        NSError *error = nil;
        id<MTLLibrary> mtl_library = [device_of(device) newLibraryWithData:data error:&error];
        if (mtl_library == nil) {
            return AMI_ERR_LIBRARY;
        }
        ami_library *library = calloc(1, sizeof *library);
        if (library == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        library->mtl_library = (__bridge_retained void *)mtl_library;
        *out_library = library;
    }
    return AMI_OK;
}

void ami_library_destroy(ami_library *library) {
    if (library == NULL) {
        return;
    }
    CFRelease(library->mtl_library);
    free(library);
}

ami_status ami_library_has_function(const ami_library *library, const char *name,
                                    uint32_t *out_present) {
    if (out_present == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_present = 0;
    if (library == NULL || name == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        NSString *function_name = [NSString stringWithUTF8String:name];
        if (function_name == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        id<MTLLibrary> mtl_library = (__bridge id<MTLLibrary>)library->mtl_library;
        *out_present = [mtl_library.functionNames containsObject:function_name] ? 1 : 0;
    }
    return AMI_OK;
}
