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

typedef struct ami_frame_slot {
    void *mtl_allocator;      // id<MTL4CommandAllocator>, retained
    void *mtl_command_buffer; // id<MTL4CommandBuffer>, retained
    void *mtl_arg_table;      // id<MTL4ArgumentTable>, retained
    void *mtl_uniform_buffer; // id<MTLBuffer>, retained (preallocated uniform ring)
    uint8_t *uniform_base;    // cached mtl_uniform_buffer.contents
    uint64_t uniform_gpu;     // cached mtl_uniform_buffer.gpuAddress
    uint64_t last_signal;
    bool in_flight;
} ami_frame_slot;

struct ami_compute {
    void *mtl_queue;                 // id<MTL4CommandQueue>, retained
    void *mtl_user_residency_set;    // id<MTLResidencySet>, retained
    void *mtl_internal_residency_set;// id<MTLResidencySet>, retained (uniform rings)
    void *mtl_event;                 // id<MTLSharedEvent>, retained
    void *active_encoder;            // id<MTL4ComputeCommandEncoder>, retained while frame open
    uint32_t frame_count;
    uint32_t next_slot;
    int32_t active_slot;             // -1 when no frame is open
    uint32_t uniform_cursor;
    uint64_t next_signal;
    ami_frame_slot slots[AMI_MAX_FRAMES_IN_FLIGHT];
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
    return ami_pipeline_create_specialized(device, library, function_name, NULL, 0, out_pipeline);
}

ami_status ami_pipeline_create_specialized(ami_device *device, const ami_library *library,
                                           const char *function_name,
                                           const ami_fn_constant *constants,
                                           size_t constant_count,
                                           ami_pipeline **out_pipeline) {
    if (out_pipeline == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_pipeline = NULL;
    if (device == NULL || library == NULL || function_name == NULL ||
        (constant_count > 0 && constants == NULL)) {
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

        MTL4LibraryFunctionDescriptor *base_desc = [[MTL4LibraryFunctionDescriptor alloc] init];
        base_desc.library = mtl_library;
        base_desc.name = name;

        MTL4FunctionDescriptor *fn_desc = base_desc;
        if (constant_count > 0) {
            MTLFunctionConstantValues *values = [[MTLFunctionConstantValues alloc] init];
            for (size_t i = 0; i < constant_count; i++) {
                if (constants[i].type == AMI_CONST_U32) {
                    uint32_t v = constants[i].value;
                    [values setConstantValue:&v
                                        type:MTLDataTypeUInt
                                     atIndex:(NSUInteger)constants[i].index];
                } else if (constants[i].type == AMI_CONST_BOOL) {
                    bool b = (constants[i].value != 0);
                    [values setConstantValue:&b
                                        type:MTLDataTypeBool
                                     atIndex:(NSUInteger)constants[i].index];
                } else {
                    return AMI_ERR_INVALID_ARGUMENT;
                }
            }
            MTL4SpecializedFunctionDescriptor *spec_desc =
                [[MTL4SpecializedFunctionDescriptor alloc] init];
            spec_desc.functionDescriptor = base_desc;
            spec_desc.constantValues = values;
            fn_desc = spec_desc;
        }

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

uint32_t ami_pipeline_max_threads_per_threadgroup(const ami_pipeline *pipeline) {
    if (pipeline == NULL) {
        return 0;
    }
    return (uint32_t)pipeline_of(pipeline).maxTotalThreadsPerThreadgroup;
}

static void release_frame_slot(ami_frame_slot *slot) {
    if (slot->mtl_uniform_buffer != NULL) CFRelease(slot->mtl_uniform_buffer);
    if (slot->mtl_arg_table != NULL) CFRelease(slot->mtl_arg_table);
    if (slot->mtl_command_buffer != NULL) CFRelease(slot->mtl_command_buffer);
    if (slot->mtl_allocator != NULL) CFRelease(slot->mtl_allocator);
    memset(slot, 0, sizeof *slot);
}

ami_status ami_compute_create(ami_device *device, ami_compute **out_compute) {
    return ami_compute_create_with_frames(device, 2, out_compute);
}

ami_status ami_compute_create_with_frames(ami_device *device, uint32_t frames_in_flight,
                                          ami_compute **out_compute) {
    if (out_compute == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_compute = NULL;
    if (device == NULL || frames_in_flight == 0 || frames_in_flight > AMI_MAX_FRAMES_IN_FLIGHT) {
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

        NSError *error = nil;
        MTLResidencySetDescriptor *user_res_desc = [[MTLResidencySetDescriptor alloc] init];
        user_res_desc.initialCapacity = AMI_MAX_BUFFER_BINDINGS;
        id<MTLResidencySet> user_res = [mtl_device newResidencySetWithDescriptor:user_res_desc
                                                                           error:&error];
        if (user_res == nil) {
            return AMI_ERR_COMPUTE;
        }
        [user_res commit];
        [user_res requestResidency];

        MTLResidencySetDescriptor *int_res_desc = [[MTLResidencySetDescriptor alloc] init];
        int_res_desc.initialCapacity = frames_in_flight;
        id<MTLResidencySet> int_res = [mtl_device newResidencySetWithDescriptor:int_res_desc
                                                                          error:&error];
        if (int_res == nil) {
            [user_res endResidency];
            return AMI_ERR_COMPUTE;
        }

        id<MTLSharedEvent> event = [mtl_device newSharedEvent];
        if (event == nil) {
            [user_res endResidency];
            return AMI_ERR_COMPUTE;
        }

        ami_compute *compute = calloc(1, sizeof *compute);
        if (compute == NULL) {
            [user_res endResidency];
            return AMI_ERR_OUT_OF_MEMORY;
        }
        compute->mtl_queue = (__bridge_retained void *)queue;
        compute->mtl_user_residency_set = (__bridge_retained void *)user_res;
        compute->mtl_internal_residency_set = (__bridge_retained void *)int_res;
        compute->mtl_event = (__bridge_retained void *)event;
        compute->frame_count = frames_in_flight;
        compute->next_slot = 0;
        compute->active_slot = -1;
        compute->uniform_cursor = 0;
        compute->next_signal = 1;

        const size_t uniform_bytes = AMI_UNIFORM_STRIDE * AMI_UNIFORM_SLOTS_PER_FRAME;
        for (uint32_t i = 0; i < frames_in_flight; i++) {
            id<MTL4CommandAllocator> allocator = [mtl_device newCommandAllocator];
            id<MTL4CommandBuffer> cb = [mtl_device newCommandBuffer];

            MTL4ArgumentTableDescriptor *table_desc = [[MTL4ArgumentTableDescriptor alloc] init];
            table_desc.maxBufferBindCount = AMI_MAX_BUFFER_BINDINGS;
            table_desc.initializeBindings = YES;
            id<MTL4ArgumentTable> arg_table = [mtl_device newArgumentTableWithDescriptor:table_desc
                                                                                   error:&error];
            id<MTLBuffer> uniform_buf =
                [mtl_device newBufferWithLength:uniform_bytes
                                        options:MTLResourceStorageModeShared];
            if (allocator == nil || cb == nil || arg_table == nil || uniform_buf == nil) {
                ami_compute_destroy(compute);
                return AMI_ERR_COMPUTE;
            }
            [int_res addAllocation:uniform_buf];
            compute->slots[i].mtl_allocator = (__bridge_retained void *)allocator;
            compute->slots[i].mtl_command_buffer = (__bridge_retained void *)cb;
            compute->slots[i].mtl_arg_table = (__bridge_retained void *)arg_table;
            compute->slots[i].mtl_uniform_buffer = (__bridge_retained void *)uniform_buf;
            compute->slots[i].uniform_base = (uint8_t *)uniform_buf.contents;
            compute->slots[i].uniform_gpu = (uint64_t)uniform_buf.gpuAddress;
        }

        [int_res commit];
        [int_res requestResidency];
        [queue addResidencySet:user_res];
        [queue addResidencySet:int_res];

        *out_compute = compute;
    }
    return AMI_OK;
}

void ami_compute_destroy(ami_compute *compute) {
    if (compute == NULL) {
        return;
    }
    ami_compute_abort_frame(compute);
    @autoreleasepool {
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)compute->mtl_event;
        uint64_t max_sig = 0;
        for (uint32_t i = 0; i < compute->frame_count; i++) {
            if (compute->slots[i].in_flight && compute->slots[i].last_signal > max_sig) {
                max_sig = compute->slots[i].last_signal;
            }
        }
        if (max_sig > 0 && event != nil) {
            [event waitUntilSignaledValue:max_sig timeoutMS:5000];
        }

        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)compute->mtl_queue;
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        id<MTLResidencySet> int_res =
            (__bridge id<MTLResidencySet>)compute->mtl_internal_residency_set;
        if (queue != nil && user_res != nil) [queue removeResidencySet:user_res];
        if (queue != nil && int_res != nil) [queue removeResidencySet:int_res];
        if (user_res != nil) {
            [user_res removeAllAllocations];
            [user_res commit];
            [user_res endResidency];
        }
        if (int_res != nil) {
            [int_res removeAllAllocations];
            [int_res commit];
            [int_res endResidency];
        }
    }
    for (uint32_t i = 0; i < AMI_MAX_FRAMES_IN_FLIGHT; i++) {
        release_frame_slot(&compute->slots[i]);
    }
    if (compute->mtl_event != NULL) CFRelease(compute->mtl_event);
    if (compute->mtl_internal_residency_set != NULL) CFRelease(compute->mtl_internal_residency_set);
    if (compute->mtl_user_residency_set != NULL) CFRelease(compute->mtl_user_residency_set);
    if (compute->mtl_queue != NULL) CFRelease(compute->mtl_queue);
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
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            [user_res addAllocation:buffer_of(buffers[i])];
        }
        [user_res commit];
        [user_res requestResidency];
    }
    return AMI_OK;
}

ami_status ami_compute_remove_resident(ami_compute *compute, ami_buffer *const *buffers,
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
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            [user_res removeAllocation:buffer_of(buffers[i])];
        }
        [user_res commit];
    }
    return AMI_OK;
}

ami_status ami_compute_clear_residency(ami_compute *compute) {
    if (compute == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        [user_res removeAllAllocations];
        [user_res commit];
    }
    return AMI_OK;
}

ami_status ami_compute_begin_frame(ami_compute *compute, uint64_t timeout_ms,
                                   uint32_t *out_slot) {
    if (compute == NULL || compute->active_slot >= 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    uint32_t slot_idx = compute->next_slot;
    ami_frame_slot *slot = &compute->slots[slot_idx];

    @autoreleasepool {
        id<MTL4CommandAllocator> allocator =
            (__bridge id<MTL4CommandAllocator>)slot->mtl_allocator;
        id<MTL4CommandBuffer> cb = (__bridge id<MTL4CommandBuffer>)slot->mtl_command_buffer;
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)compute->mtl_event;

        if (slot->in_flight) {
            if (![event waitUntilSignaledValue:slot->last_signal timeoutMS:timeout_ms]) {
                return AMI_ERR_TIMEOUT;
            }
            [allocator reset];
            slot->in_flight = false;
        }

        [cb beginCommandBufferWithAllocator:allocator];
        id<MTL4ComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        if (encoder == nil) {
            [cb endCommandBuffer];
            [allocator reset];
            return AMI_ERR_COMPUTE;
        }
        compute->active_encoder = (__bridge_retained void *)encoder;
        compute->active_slot = (int32_t)slot_idx;
        compute->uniform_cursor = 0;
        if (out_slot != NULL) {
            *out_slot = slot_idx;
        }
    }
    return AMI_OK;
}

void ami_compute_abort_frame(ami_compute *compute) {
    if (compute == NULL || compute->active_slot < 0) {
        return;
    }
    ami_frame_slot *slot = &compute->slots[compute->active_slot];
    @autoreleasepool {
        if (compute->active_encoder != NULL) {
            id<MTL4ComputeCommandEncoder> encoder =
                (__bridge_transfer id<MTL4ComputeCommandEncoder>)compute->active_encoder;
            compute->active_encoder = NULL;
            [encoder endEncoding];
        }
        id<MTL4CommandBuffer> cb = (__bridge id<MTL4CommandBuffer>)slot->mtl_command_buffer;
        id<MTL4CommandAllocator> allocator =
            (__bridge id<MTL4CommandAllocator>)slot->mtl_allocator;
        [cb endCommandBuffer];
        [allocator reset];
    }
    compute->active_slot = -1;
    compute->uniform_cursor = 0;
}

ami_status ami_compute_encode_dispatch(ami_compute *compute, const ami_pipeline *pipeline,
                                       const ami_buffer_binding *bindings, size_t binding_count,
                                       const void *uniform_bytes, size_t uniform_len,
                                       uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                                       uint64_t tg_x, uint64_t tg_y, uint64_t tg_z,
                                       uint32_t dispatch_mode, uint32_t insert_barrier_before) {
    if (compute == NULL || compute->active_slot < 0 || compute->active_encoder == NULL ||
        pipeline == NULL || grid_x == 0 || grid_y == 0 || grid_z == 0 ||
        tg_x == 0 || tg_y == 0 || tg_z == 0 ||
        (binding_count > 0 && bindings == NULL) ||
        (uniform_len > 0 && uniform_bytes == NULL) ||
        uniform_len > AMI_UNIFORM_STRIDE ||
        (dispatch_mode != AMI_DISPATCH_THREADS && dispatch_mode != AMI_DISPATCH_THREADGROUPS)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    size_t total_bindings = binding_count + (uniform_len > 0 ? 1u : 0u);
    if (total_bindings > AMI_MAX_BUFFER_BINDINGS) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < binding_count; i++) {
        if (bindings[i].buffer == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }

    ami_frame_slot *slot = &compute->slots[compute->active_slot];
    if (uniform_len > 0 && compute->uniform_cursor >= AMI_UNIFORM_SLOTS_PER_FRAME) {
        return AMI_ERR_OUT_OF_MEMORY;
    }

    @autoreleasepool {
        id<MTL4ComputeCommandEncoder> encoder =
            (__bridge id<MTL4ComputeCommandEncoder>)compute->active_encoder;
        id<MTL4ArgumentTable> arg_table = (__bridge id<MTL4ArgumentTable>)slot->mtl_arg_table;
        id<MTLComputePipelineState> state = pipeline_of(pipeline);

        if (insert_barrier_before != 0) {
            [encoder barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
        }

        for (size_t i = 0; i < binding_count; i++) {
            MTLGPUAddress addr = buffer_of(bindings[i].buffer).gpuAddress + bindings[i].offset;
            [arg_table setAddress:addr atIndex:(NSUInteger)i];
        }

        if (uniform_len > 0) {
            size_t u_off = (size_t)compute->uniform_cursor * AMI_UNIFORM_STRIDE;
            memcpy(slot->uniform_base + u_off, uniform_bytes, uniform_len);
            [arg_table setAddress:(slot->uniform_gpu + u_off)
                          atIndex:(NSUInteger)binding_count];
            compute->uniform_cursor++;
        }

        [encoder setComputePipelineState:state];
        [encoder setArgumentTable:arg_table];

        MTLSize grid = MTLSizeMake((NSUInteger)grid_x, (NSUInteger)grid_y, (NSUInteger)grid_z);
        MTLSize tg = MTLSizeMake((NSUInteger)tg_x, (NSUInteger)tg_y, (NSUInteger)tg_z);
        if (dispatch_mode == AMI_DISPATCH_THREADS) {
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        } else {
            [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tg];
        }
    }
    return AMI_OK;
}

ami_status ami_compute_commit_frame(ami_compute *compute, uint64_t *out_signal) {
    if (compute == NULL || compute->active_slot < 0 || compute->active_encoder == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    uint32_t slot_idx = (uint32_t)compute->active_slot;
    ami_frame_slot *slot = &compute->slots[slot_idx];
    @autoreleasepool {
        id<MTL4ComputeCommandEncoder> encoder =
            (__bridge_transfer id<MTL4ComputeCommandEncoder>)compute->active_encoder;
        compute->active_encoder = NULL;
        [encoder endEncoding];

        id<MTL4CommandBuffer> cb = (__bridge id<MTL4CommandBuffer>)slot->mtl_command_buffer;
        id<MTL4CommandQueue> queue = (__bridge id<MTL4CommandQueue>)compute->mtl_queue;
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)compute->mtl_event;

        [cb endCommandBuffer];
        const id<MTL4CommandBuffer> batch[1] = {cb};
        [queue commit:batch count:1];

        uint64_t sig = compute->next_signal++;
        slot->last_signal = sig;
        slot->in_flight = true;
        [queue signalEvent:event value:sig];

        compute->active_slot = -1;
        compute->uniform_cursor = 0;
        compute->next_slot = (slot_idx + 1u) % compute->frame_count;
        if (out_signal != NULL) {
            *out_signal = sig;
        }
    }
    return AMI_OK;
}

ami_status ami_compute_wait(ami_compute *compute, uint64_t signal, uint64_t timeout_ms) {
    if (compute == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (signal == 0) {
        return AMI_OK;
    }
    @autoreleasepool {
        id<MTLSharedEvent> event = (__bridge id<MTLSharedEvent>)compute->mtl_event;
        if (![event waitUntilSignaledValue:signal timeoutMS:timeout_ms]) {
            return AMI_ERR_TIMEOUT;
        }
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
    ami_buffer_binding bindings[AMI_MAX_BUFFER_BINDINGS];
    for (size_t i = 0; i < buffer_count; i++) {
        if (buffers[i] == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        bindings[i].buffer = buffers[i];
        bindings[i].offset = 0;
    }
    ami_status st = ami_compute_begin_frame(compute, timeout_ms, NULL);
    if (st != AMI_OK) {
        return st;
    }
    uint64_t max_tg = ami_pipeline_max_threads_per_threadgroup(pipeline);
    uint64_t tg = threads < max_tg ? threads : max_tg;
    st = ami_compute_encode_dispatch(compute, pipeline, bindings, buffer_count, NULL, 0,
                                     threads, 1, 1, tg, 1, 1, AMI_DISPATCH_THREADS, 0);
    if (st != AMI_OK) {
        ami_compute_abort_frame(compute);
        return st;
    }
    uint64_t sig = 0;
    st = ami_compute_commit_frame(compute, &sig);
    if (st != AMI_OK) {
        return st;
    }
    return ami_compute_wait(compute, sig, timeout_ms);
}
