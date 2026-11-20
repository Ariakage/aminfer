// apple_bridge.m — Objective-C (ARC) implementation of apple_bridge.h.
//
// Handles own their Objective-C objects through __bridge_retained pointers and
// release them in the matching *_destroy function.

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "apple_bridge.h"

struct ami_device {
    void *mtl_device; // id<MTLDevice>, retained
};

struct ami_library {
    void *mtl_library; // id<MTLLibrary>, retained
};

struct ami_buffer {
    void *mtl_buffer; // id<MTLBuffer>, retained
};

struct ami_pipeline {
    void *mtl_pipeline; // id<MTLComputePipelineState>, retained
};

struct ami_compute {
    void *mtl_queue;          // id<MTL4CommandQueue>, retained
    void *mtl_allocator;      // id<MTL4CommandAllocator>, retained
    void *mtl_command_buffer; // id<MTL4CommandBuffer>, retained
    void *mtl_arg_table;      // id<MTL4ArgumentTable>, retained
    void *mtl_residency_set;  // id<MTLResidencySet>, retained
    void *mtl_event;          // id<MTLSharedEvent>, retained
    uint64_t next_signal;
    uint64_t last_signal;
    bool allocator_busy;
};

static id<MTLDevice> device_of(const ami_device *device) {
    return (__bridge id<MTLDevice>)device->mtl_device;
}

static id<MTLLibrary> library_of(const ami_library *library) {
    return (__bridge id<MTLLibrary>)library->mtl_library;
}

static id<MTLBuffer> buffer_of(const ami_buffer *buffer) {
    return (__bridge id<MTLBuffer>)buffer->mtl_buffer;
}

static id<MTLComputePipelineState> pipeline_of(const ami_pipeline *pipeline) {
    return (__bridge id<MTLComputePipelineState>)pipeline->mtl_pipeline;
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
        *out_present = [library_of(library).functionNames containsObject:function_name] ? 1 : 0;
    }
    return AMI_OK;
}

ami_status ami_buffer_wrap(ami_device *device, void *ptr, size_t length,
                           ami_buffer **out_buffer) {
    if (out_buffer == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_buffer = NULL;
    if (device == NULL || ptr == NULL || length == 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    // Defense in depth: Metal does not reliably reject non-16 KiB NoCopy spans.
    if (((uintptr_t)ptr % AMI_PAGE_SIZE) != 0 || (length % AMI_PAGE_SIZE) != 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        id<MTLBuffer> mtl_buffer = [device_of(device) newBufferWithBytesNoCopy:ptr
                                                                        length:length
                                                                       options:MTLResourceStorageModeShared
                                                                   deallocator:nil];
        if (mtl_buffer == nil) {
            return AMI_ERR_BUFFER;
        }
        ami_buffer *buffer = calloc(1, sizeof *buffer);
        if (buffer == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        buffer->mtl_buffer = (__bridge_retained void *)mtl_buffer;
        *out_buffer = buffer;
    }
    return AMI_OK;
}

void ami_buffer_destroy(ami_buffer *buffer) {
    if (buffer == NULL) {
        return;
    }
    CFRelease(buffer->mtl_buffer);
    free(buffer);
}

uint64_t ami_buffer_gpu_address(const ami_buffer *buffer) {
    if (buffer == NULL) {
        return 0;
    }
    return (uint64_t)buffer_of(buffer).gpuAddress;
}

ami_status ami_pipeline_create(ami_device *device, const ami_library *library,
                               const char *function_name, ami_pipeline **out_pipeline) {
    if (out_pipeline == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_pipeline = NULL;
    if (device == NULL || library == NULL || function_name == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        NSString *name = [NSString stringWithUTF8String:function_name];
        if (name == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        id<MTLDevice> mtl_device = device_of(device);
        id<MTLLibrary> mtl_library = library_of(library);
        if (![mtl_library.functionNames containsObject:name]) {
            return AMI_ERR_PIPELINE;
        }

        NSError *error = nil;
        MTL4CompilerDescriptor *compiler_desc = [[MTL4CompilerDescriptor alloc] init];
        id<MTL4Compiler> compiler = [mtl_device newCompilerWithDescriptor:compiler_desc
                                                                    error:&error];
        if (compiler == nil) {
            return AMI_ERR_PIPELINE;
        }

        MTL4LibraryFunctionDescriptor *fn_desc = [[MTL4LibraryFunctionDescriptor alloc] init];
        fn_desc.library = mtl_library;
        fn_desc.name = name;

        MTL4ComputePipelineDescriptor *pipe_desc = [[MTL4ComputePipelineDescriptor alloc] init];
        pipe_desc.computeFunctionDescriptor = fn_desc;

        id<MTLComputePipelineState> state =
            [compiler newComputePipelineStateWithDescriptor:pipe_desc
                                        compilerTaskOptions:nil
                                                      error:&error];
        if (state == nil) {
            return AMI_ERR_PIPELINE;
        }

        ami_pipeline *pipeline = calloc(1, sizeof *pipeline);
        if (pipeline == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        pipeline->mtl_pipeline = (__bridge_retained void *)state;
        *out_pipeline = pipeline;
    }
    return AMI_OK;
}

void ami_pipeline_destroy(ami_pipeline *pipeline) {
    if (pipeline == NULL) {
        return;
    }
    CFRelease(pipeline->mtl_pipeline);
    free(pipeline);
}

ami_status ami_compute_create(ami_device *device, ami_compute **out_compute) {
    if (out_compute == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_compute = NULL;
    if (device == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        id<MTLDevice> mtl_device = device_of(device);
        if (![mtl_device respondsToSelector:@selector(newMTL4CommandQueue)]) {
            return AMI_ERR_UNSUPPORTED;
        }

        id<MTL4CommandQueue> queue = [mtl_device newMTL4CommandQueue];
        if (queue == nil) {
            return AMI_ERR_COMPUTE;
        }
        id<MTL4CommandAllocator> allocator = [mtl_device newCommandAllocator];
        if (allocator == nil) {
            return AMI_ERR_COMPUTE;
        }
        id<MTL4CommandBuffer> command_buffer = [mtl_device newCommandBuffer];
        if (command_buffer == nil) {
            return AMI_ERR_COMPUTE;
        }

        NSError *error = nil;
        MTL4ArgumentTableDescriptor *table_desc = [[MTL4ArgumentTableDescriptor alloc] init];
        table_desc.maxBufferBindCount = AMI_MAX_BUFFER_BINDINGS;
        table_desc.initializeBindings = YES;
        id<MTL4ArgumentTable> arg_table = [mtl_device newArgumentTableWithDescriptor:table_desc
                                                                               error:&error];
        if (arg_table == nil) {
            return AMI_ERR_COMPUTE;
        }

        MTLResidencySetDescriptor *res_desc = [[MTLResidencySetDescriptor alloc] init];
        res_desc.initialCapacity = AMI_MAX_BUFFER_BINDINGS;
        id<MTLResidencySet> residency_set = [mtl_device newResidencySetWithDescriptor:res_desc
                                                                                error:&error];
        if (residency_set == nil) {
            return AMI_ERR_COMPUTE;
        }
        [residency_set commit];
        [residency_set requestResidency];
        [queue addResidencySet:residency_set];

        id<MTLSharedEvent> event = [mtl_device newSharedEvent];
        if (event == nil) {
            [queue removeResidencySet:residency_set];
            [residency_set endResidency];
            return AMI_ERR_COMPUTE;
        }

        ami_compute *compute = calloc(1, sizeof *compute);
        if (compute == NULL) {
            [queue removeResidencySet:residency_set];
            [residency_set endResidency];
            return AMI_ERR_OUT_OF_MEMORY;
        }
        compute->mtl_queue = (__bridge_retained void *)queue;
        compute->mtl_allocator = (__bridge_retained void *)allocator;
        compute->mtl_command_buffer = (__bridge_retained void *)command_buffer;
        compute->mtl_arg_table = (__bridge_retained void *)arg_table;
        compute->mtl_residency_set = (__bridge_retained void *)residency_set;
        compute->mtl_event = (__bridge_retained void *)event;
        compute->next_signal = 1;
        compute->last_signal = 0;
        compute->allocator_busy = false;
        *out_compute = compute;
    }
    return AMI_OK;
}

void ami_compute_destroy(ami_compute *compute) {
    if (compute == NULL) {
        return;
    }
    @autoreleasepool {
        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)compute->mtl_queue;
        id<MTLResidencySet> residency_set =
            (__bridge id<MTLResidencySet>)compute->mtl_residency_set;
        [queue removeResidencySet:residency_set];
        [residency_set removeAllAllocations];
        [residency_set commit];
        [residency_set endResidency];
    }
    CFRelease(compute->mtl_event);
    CFRelease(compute->mtl_residency_set);
    CFRelease(compute->mtl_arg_table);
    CFRelease(compute->mtl_command_buffer);
    CFRelease(compute->mtl_allocator);
    CFRelease(compute->mtl_queue);
    free(compute);
}

ami_status ami_compute_make_resident(ami_compute *compute, ami_buffer *const *buffers,
                                     size_t count) {
    if (compute == NULL || (count > 0 && buffers == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return AMI_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (buffers[i] == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTLResidencySet> residency_set =
            (__bridge id<MTLResidencySet>)compute->mtl_residency_set;
        for (size_t i = 0; i < count; i++) {
            [residency_set addAllocation:buffer_of(buffers[i])];
        }
        [residency_set commit];
        [residency_set requestResidency];
    }
    return AMI_OK;
}

ami_status ami_compute_clear_residency(ami_compute *compute) {
    if (compute == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        id<MTLResidencySet> residency_set =
            (__bridge id<MTLResidencySet>)compute->mtl_residency_set;
        [residency_set removeAllAllocations];
        [residency_set commit];
    }
    return AMI_OK;
}

ami_status ami_compute_dispatch_1d_sync(ami_compute *compute, const ami_pipeline *pipeline,
                                        ami_buffer *const *buffers, size_t buffer_count,
                                        uint64_t threads, uint64_t timeout_ms) {
    if (compute == NULL || pipeline == NULL || threads == 0 ||
        buffer_count > AMI_MAX_BUFFER_BINDINGS || (buffer_count > 0 && buffers == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < buffer_count; i++) {
        if (buffers[i] == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)compute->mtl_queue;
        id<MTL4CommandAllocator> allocator =
            (__bridge id<MTL4CommandAllocator>)compute->mtl_allocator;
        id<MTL4CommandBuffer> command_buffer =
            (__bridge id<MTL4CommandBuffer>)compute->mtl_command_buffer;
        id<MTL4ArgumentTable> arg_table =
            (__bridge id<MTL4ArgumentTable>)compute->mtl_arg_table;
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)compute->mtl_event;
        id<MTLComputePipelineState> state = pipeline_of(pipeline);

        // Never reset an allocator whose previous submission is still in flight.
        if (compute->allocator_busy) {
            if (![event waitUntilSignaledValue:compute->last_signal timeoutMS:timeout_ms]) {
                return AMI_ERR_TIMEOUT;
            }
            [allocator reset];
            compute->allocator_busy = false;
        }

        for (size_t i = 0; i < buffer_count; i++) {
            [arg_table setAddress:buffer_of(buffers[i]).gpuAddress atIndex:(NSUInteger)i];
        }

        [command_buffer beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (encoder == nil) {
            [command_buffer endCommandBuffer];
            [allocator reset];
            return AMI_ERR_COMPUTE;
        }

        [encoder setComputePipelineState:state];
        [encoder setArgumentTable:arg_table];

        NSUInteger max_tg = state.maxTotalThreadsPerThreadgroup;
        NSUInteger tg_width =
            (threads < (uint64_t)max_tg) ? (NSUInteger)threads : max_tg;
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)threads, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(tg_width, 1, 1)];
        [encoder endEncoding];
        [command_buffer endCommandBuffer];

        const id<MTL4CommandBuffer> batch[1] = {command_buffer};
        [queue commit:batch count:1];

        uint64_t signal_value = compute->next_signal++;
        compute->last_signal = signal_value;
        compute->allocator_busy = true;
        [queue signalEvent:event value:signal_value];

        if (![event waitUntilSignaledValue:signal_value timeoutMS:timeout_ms]) {
            return AMI_ERR_TIMEOUT;
        }
        [allocator reset];
        compute->allocator_busy = false;
    }
    return AMI_OK;
}
