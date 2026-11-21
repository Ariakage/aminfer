// apple_bridge.m — Objective-C (ARC) implementation of apple_bridge.h.
//
// Handles own their Objective-C objects through __bridge_retained pointers and
// release them in the matching *_destroy function.

#define ACCELERATE_NEW_LAPACK 1
#import <Accelerate/Accelerate.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <Metal/Metal.h>

#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

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

struct ami_kv_tile {
    CVPixelBufferRef pixel_buffer; // retained via CoreVideo
    IOSurfaceRef surface;          // borrowed from pixel_buffer
    void *mtl_rgba_texture;        // id<MTLTexture> (RGBA16Float), retained
    void *mtl_r16_texture;         // id<MTLTexture> (R16Float), retained
    ami_buffer *alias_buffer;      // optional zero-copy MTLBuffer, owned
    void *base_addr;               // 16 KiB-aligned IOSurface base address
    uint32_t row_f16;
    uint32_t rows;
    uint32_t bytes_per_row;
    uint32_t is_mlocked;
    uint64_t data_bytes;
    uint64_t alloc_bytes;
    uint64_t surface_alloc_size;
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

static id<MTLTexture> tile_texture_of(const ami_kv_tile *tile, uint32_t view) {
    if (view == AMI_TEX_VIEW_R16F) {
        return (__bridge id<MTLTexture>)tile->mtl_r16_texture;
    }
    return (__bridge id<MTLTexture>)tile->mtl_rgba_texture;
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

ami_status ami_kv_tile_create(ami_device *device, uint32_t row_f16, uint32_t rows,
                              uint32_t flags, ami_kv_tile **out_tile) {
    if (out_tile == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_tile = NULL;
    const uint32_t valid_flags = AMI_KV_TILE_FLAG_ALIAS_BUFFER | AMI_KV_TILE_FLAG_MLOCK;
    if (device == NULL || row_f16 == 0 || rows == 0 ||
        (row_f16 % AMI_KV_TILE_ROW_ALIGN_F16) != 0 ||
        row_f16 > AMI_MAX_TEXTURE_DIM || rows > AMI_MAX_TEXTURE_DIM ||
        (flags & ~valid_flags) != 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const uint64_t bytes_per_row = (uint64_t)row_f16 * 2u;
    const uint64_t data_bytes = bytes_per_row * (uint64_t)rows;
    const uint64_t alloc_bytes =
        ((data_bytes + AMI_PAGE_SIZE - 1u) / AMI_PAGE_SIZE) * AMI_PAGE_SIZE;

    @autoreleasepool {
        NSDictionary *surface_props = @{
            (__bridge id)kIOSurfaceWidth: @(row_f16),
            (__bridge id)kIOSurfaceHeight: @(rows),
            (__bridge id)kIOSurfaceBytesPerElement: @2,
            (__bridge id)kIOSurfaceBytesPerRow: @(bytes_per_row),
            (__bridge id)kIOSurfaceAllocSize: @(alloc_bytes),
            (__bridge id)kIOSurfacePixelFormat: @(kCVPixelFormatType_OneComponent16Half),
        };
        IOSurfaceRef created_surface = IOSurfaceCreate((__bridge CFDictionaryRef)surface_props);
        if (created_surface == NULL) {
            return AMI_ERR_TILE;
        }

        NSDictionary *pb_attrs = @{
            (__bridge id)kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        CVPixelBufferRef pixel_buffer = NULL;
        CVReturn cv_ret = CVPixelBufferCreateWithIOSurface(
            kCFAllocatorDefault,
            created_surface,
            (__bridge CFDictionaryRef)pb_attrs,
            &pixel_buffer);
        CFRelease(created_surface);
        if (cv_ret != kCVReturnSuccess || pixel_buffer == NULL) {
            return AMI_ERR_TILE;
        }

        IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixel_buffer);
        if (surface == NULL ||
            CVPixelBufferGetPixelFormatType(pixel_buffer) != kCVPixelFormatType_OneComponent16Half) {
            CVPixelBufferRelease(pixel_buffer);
            return AMI_ERR_TILE;
        }

        const size_t actual_bpr = IOSurfaceGetBytesPerRow(surface);
        const size_t surface_alloc = IOSurfaceGetAllocSize(surface);
        void *base_addr = IOSurfaceGetBaseAddress(surface);
        if (base_addr == NULL ||
            ((uintptr_t)base_addr % AMI_PAGE_SIZE) != 0 ||
            actual_bpr != bytes_per_row ||
            (actual_bpr % 64u) != 0 ||
            surface_alloc < alloc_bytes) {
            CVPixelBufferRelease(pixel_buffer);
            return AMI_ERR_TILE;
        }

        id<MTLDevice> mtl_device = device_of(device);

        MTLTextureDescriptor *rgba_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                               width:(NSUInteger)(row_f16 / 4u)
                                                              height:(NSUInteger)rows
                                                           mipmapped:NO];
        rgba_desc.storageMode = MTLStorageModeShared;
        rgba_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> rgba_tex = [mtl_device newTextureWithDescriptor:rgba_desc
                                                             iosurface:surface
                                                                 plane:0];
        if (rgba_tex == nil) {
            CVPixelBufferRelease(pixel_buffer);
            return AMI_ERR_TILE;
        }

        MTLTextureDescriptor *r16_desc =
            [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatR16Float
                                                               width:(NSUInteger)row_f16
                                                              height:(NSUInteger)rows
                                                           mipmapped:NO];
        r16_desc.storageMode = MTLStorageModeShared;
        r16_desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        id<MTLTexture> r16_tex = [mtl_device newTextureWithDescriptor:r16_desc
                                                            iosurface:surface
                                                                plane:0];
        if (r16_tex == nil) {
            CVPixelBufferRelease(pixel_buffer);
            return AMI_ERR_TILE;
        }

        uint32_t is_mlocked = 0;
        if ((flags & AMI_KV_TILE_FLAG_MLOCK) != 0) {
            if (mlock(base_addr, (size_t)alloc_bytes) != 0) {
                CVPixelBufferRelease(pixel_buffer);
                return AMI_ERR_TILE;
            }
            is_mlocked = 1;
        }

        ami_buffer *alias_buffer = NULL;
        if ((flags & AMI_KV_TILE_FLAG_ALIAS_BUFFER) != 0) {
            ami_status st = ami_buffer_wrap(device, base_addr, (size_t)alloc_bytes, &alias_buffer);
            if (st != AMI_OK) {
                if (is_mlocked != 0) {
                    munlock(base_addr, (size_t)alloc_bytes);
                }
                CVPixelBufferRelease(pixel_buffer);
                return st;
            }
        }

        ami_kv_tile *tile = calloc(1, sizeof *tile);
        if (tile == NULL) {
            if (alias_buffer != NULL) {
                ami_buffer_destroy(alias_buffer);
            }
            if (is_mlocked != 0) {
                munlock(base_addr, (size_t)alloc_bytes);
            }
            CVPixelBufferRelease(pixel_buffer);
            return AMI_ERR_OUT_OF_MEMORY;
        }

        tile->pixel_buffer = pixel_buffer;
        tile->surface = surface;
        tile->mtl_rgba_texture = (__bridge_retained void *)rgba_tex;
        tile->mtl_r16_texture = (__bridge_retained void *)r16_tex;
        tile->alias_buffer = alias_buffer;
        tile->base_addr = base_addr;
        tile->row_f16 = row_f16;
        tile->rows = rows;
        tile->bytes_per_row = (uint32_t)actual_bpr;
        tile->is_mlocked = is_mlocked;
        tile->data_bytes = data_bytes;
        tile->alloc_bytes = alloc_bytes;
        tile->surface_alloc_size = (uint64_t)surface_alloc;
        *out_tile = tile;
    }
    return AMI_OK;
}

void ami_kv_tile_destroy(ami_kv_tile *tile) {
    if (tile == NULL) {
        return;
    }
    if (tile->alias_buffer != NULL) {
        ami_buffer_destroy(tile->alias_buffer);
    }
    if (tile->mtl_r16_texture != NULL) {
        CFRelease(tile->mtl_r16_texture);
    }
    if (tile->mtl_rgba_texture != NULL) {
        CFRelease(tile->mtl_rgba_texture);
    }
    if (tile->is_mlocked != 0 && tile->base_addr != NULL && tile->alloc_bytes > 0) {
        munlock(tile->base_addr, (size_t)tile->alloc_bytes);
    }
    if (tile->pixel_buffer != NULL) {
        CVPixelBufferRelease(tile->pixel_buffer);
    }
    free(tile);
}

ami_status ami_kv_tile_query(const ami_kv_tile *tile, ami_kv_tile_info *out_info) {
    if (tile == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_info, 0, sizeof *out_info);
    out_info->row_f16 = tile->row_f16;
    out_info->rows = tile->rows;
    out_info->bytes_per_row = tile->bytes_per_row;
    out_info->has_alias_buffer = (tile->alias_buffer != NULL) ? 1u : 0u;
    out_info->is_mlocked = tile->is_mlocked;
    out_info->data_bytes = tile->data_bytes;
    out_info->alloc_bytes = tile->alloc_bytes;
    out_info->surface_alloc_size = tile->surface_alloc_size;
    out_info->rgba_resource_id = ami_kv_tile_rgba_resource_id(tile);
    out_info->r16_resource_id = ami_kv_tile_r16_resource_id(tile);
    out_info->alias_gpu_address =
        (tile->alias_buffer != NULL) ? ami_buffer_gpu_address(tile->alias_buffer) : 0u;
    return AMI_OK;
}

uint64_t ami_kv_tile_rgba_resource_id(const ami_kv_tile *tile) {
    if (tile == NULL || tile->mtl_rgba_texture == NULL) {
        return 0;
    }
    return tile_texture_of(tile, AMI_TEX_VIEW_RGBA16F).gpuResourceID._impl;
}

uint64_t ami_kv_tile_r16_resource_id(const ami_kv_tile *tile) {
    if (tile == NULL || tile->mtl_r16_texture == NULL) {
        return 0;
    }
    return tile_texture_of(tile, AMI_TEX_VIEW_R16F).gpuResourceID._impl;
}

ami_buffer *ami_kv_tile_alias_buffer(const ami_kv_tile *tile) {
    if (tile == NULL) {
        return NULL;
    }
    return tile->alias_buffer;
}

ami_status ami_kv_tile_lock(ami_kv_tile *tile, uint32_t read_only,
                            void **out_base_ptr, size_t *out_bytes_per_row,
                            size_t *out_data_bytes) {
    if (out_base_ptr != NULL) *out_base_ptr = NULL;
    if (out_bytes_per_row != NULL) *out_bytes_per_row = 0;
    if (out_data_bytes != NULL) *out_data_bytes = 0;
    if (tile == NULL || out_base_ptr == NULL || out_bytes_per_row == NULL ||
        out_data_bytes == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    CVPixelBufferLockFlags lock_flags = (read_only != 0) ? kCVPixelBufferLock_ReadOnly : 0;
    CVReturn ret = CVPixelBufferLockBaseAddress(tile->pixel_buffer, lock_flags);
    if (ret != kCVReturnSuccess) {
        return AMI_ERR_TILE;
    }
    void *base = CVPixelBufferGetBaseAddress(tile->pixel_buffer);
    if (base == NULL || ((uintptr_t)base % AMI_PAGE_SIZE) != 0) {
        CVPixelBufferUnlockBaseAddress(tile->pixel_buffer, lock_flags);
        return AMI_ERR_TILE;
    }
    *out_base_ptr = base;
    *out_bytes_per_row = CVPixelBufferGetBytesPerRow(tile->pixel_buffer);
    *out_data_bytes = (size_t)tile->data_bytes;
    return AMI_OK;
}

ami_status ami_kv_tile_unlock(ami_kv_tile *tile, uint32_t read_only) {
    if (tile == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    CVPixelBufferLockFlags lock_flags = (read_only != 0) ? kCVPixelBufferLock_ReadOnly : 0;
    CVReturn ret = CVPixelBufferUnlockBaseAddress(tile->pixel_buffer, lock_flags);
    return (ret == kCVReturnSuccess) ? AMI_OK : AMI_ERR_TILE;
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
        user_res_desc.initialCapacity = 128;
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
            table_desc.maxTextureBindCount = AMI_MAX_TEXTURE_BINDINGS;
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

ami_status ami_compute_make_tiles_resident(ami_compute *compute, ami_kv_tile *const *tiles,
                                           size_t count) {
    if (compute == NULL || (count > 0 && tiles == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return AMI_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (tiles[i] == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            [user_res addAllocation:tile_texture_of(tiles[i], AMI_TEX_VIEW_RGBA16F)];
            [user_res addAllocation:tile_texture_of(tiles[i], AMI_TEX_VIEW_R16F)];
            if (tiles[i]->alias_buffer != NULL) {
                [user_res addAllocation:buffer_of(tiles[i]->alias_buffer)];
            }
        }
        [user_res commit];
        [user_res requestResidency];
    }
    return AMI_OK;
}

ami_status ami_compute_remove_tiles_resident(ami_compute *compute, ami_kv_tile *const *tiles,
                                             size_t count) {
    if (compute == NULL || (count > 0 && tiles == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return AMI_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (tiles[i] == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            [user_res removeAllocation:tile_texture_of(tiles[i], AMI_TEX_VIEW_RGBA16F)];
            [user_res removeAllocation:tile_texture_of(tiles[i], AMI_TEX_VIEW_R16F)];
            if (tiles[i]->alias_buffer != NULL) {
                [user_res removeAllocation:buffer_of(tiles[i]->alias_buffer)];
            }
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
    return ami_compute_encode_dispatch_ex(compute, pipeline, bindings, binding_count, NULL, 0,
                                          uniform_bytes, uniform_len, grid_x, grid_y, grid_z,
                                          tg_x, tg_y, tg_z, dispatch_mode, insert_barrier_before);
}

ami_status ami_compute_encode_dispatch_ex(ami_compute *compute, const ami_pipeline *pipeline,
                                          const ami_buffer_binding *bindings, size_t binding_count,
                                          const ami_texture_binding *textures, size_t texture_count,
                                          const void *uniform_bytes, size_t uniform_len,
                                          uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                                          uint64_t tg_x, uint64_t tg_y, uint64_t tg_z,
                                          uint32_t dispatch_mode, uint32_t insert_barrier_before) {
    if (compute == NULL || compute->active_slot < 0 || compute->active_encoder == NULL ||
        pipeline == NULL || grid_x == 0 || grid_y == 0 || grid_z == 0 ||
        tg_x == 0 || tg_y == 0 || tg_z == 0 ||
        (binding_count > 0 && bindings == NULL) ||
        (texture_count > 0 && textures == NULL) ||
        texture_count > AMI_MAX_TEXTURE_BINDINGS ||
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
    for (size_t i = 0; i < texture_count; i++) {
        if (textures[i].tile == NULL ||
            (textures[i].view != AMI_TEX_VIEW_RGBA16F && textures[i].view != AMI_TEX_VIEW_R16F)) {
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

        for (size_t i = 0; i < texture_count; i++) {
            id<MTLTexture> tex = tile_texture_of(textures[i].tile, textures[i].view);
            [arg_table setTexture:tex.gpuResourceID atIndex:(NSUInteger)i];
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

/* -------------------------------------------------------------------------
 * CoreML / Apple Neural Engine zero-copy IOSurface bridge (ADR-0004, D5, S4)
 * ------------------------------------------------------------------------- */

struct ami_ane_model {
    void *ml_model;          // MLModel *, retained
    void *temp_compiled_url; // NSURL *, retained (deleted on destroy if non-NULL)
    ami_ane_model_info info;
};

struct ami_ane_tensor {
    void *ml_array; // MLMultiArray *, retained (backed zero-copy by CVPixelBuffer)
    CVPixelBufferRef pixel_buffer; // borrowed for zero-copy identity checks
    uint32_t rank;
    uint32_t shape[AMI_ANE_MAX_DIMS];
};

struct ami_ane_plan {
    void *input_provider; // MLDictionaryFeatureProvider *, retained
    void *options;        // MLPredictionOptions *, retained
    uint32_t output_count;
    void *output_names[AMI_ANE_MAX_BINDINGS];  // NSString *, retained
    void *output_arrays[AMI_ANE_MAX_BINDINGS]; // MLMultiArray *, retained
    CVPixelBufferRef output_pbs[AMI_ANE_MAX_BINDINGS];
};

static bool ane_compute_units_from_raw(uint32_t raw, MLComputeUnits *out_units) {
    switch (raw) {
    case AMI_ANE_COMPUTE_CPU_AND_NE:
        *out_units = MLComputeUnitsCPUAndNeuralEngine;
        return true;
    case AMI_ANE_COMPUTE_ALL:
        *out_units = MLComputeUnitsAll;
        return true;
    case AMI_ANE_COMPUTE_CPU_ONLY:
        *out_units = MLComputeUnitsCPUOnly;
        return true;
    case AMI_ANE_COMPUTE_CPU_AND_GPU:
        *out_units = MLComputeUnitsCPUAndGPU;
        return true;
    default:
        return false;
    }
}

static void record_device_usage(MLComputePlanDeviceUsage *usage, ami_ane_model_info *info) {
    if (usage == nil || info == NULL) {
        return;
    }
    info->total_layers++;
    id<MLComputeDeviceProtocol> pref = usage.preferredComputeDevice;
    if ([pref isKindOfClass:[MLNeuralEngineComputeDevice class]]) {
        info->ane_preferred_layers++;
    } else if ([pref isKindOfClass:[MLGPUComputeDevice class]]) {
        info->gpu_preferred_layers++;
    } else if ([pref isKindOfClass:[MLCPUComputeDevice class]]) {
        info->cpu_preferred_layers++;
    }
}

static void populate_compute_plan_info(MLComputePlan *plan, ami_ane_model_info *info) {
    if (plan == nil || info == NULL) {
        return;
    }
    MLModelStructure *structure = plan.modelStructure;
    if (structure.neuralNetwork != nil) {
        for (MLModelStructureNeuralNetworkLayer *layer in structure.neuralNetwork.layers) {
            record_device_usage([plan computeDeviceUsageForNeuralNetworkLayer:layer], info);
        }
    } else if (structure.program != nil) {
        for (NSString *fn_name in structure.program.functions) {
            MLModelStructureProgramFunction *fn = structure.program.functions[fn_name];
            if (fn != nil && fn.block != nil) {
                for (MLModelStructureProgramOperation *op in fn.block.operations) {
                    record_device_usage([plan computeDeviceUsageForMLProgramOperation:op], info);
                }
            }
        }
    }
}

ami_status ami_ane_model_load_path(const char *path, uint32_t compute_units,
                                   ami_ane_model **out_model) {
    return ami_ane_model_load_path_ex(path, NULL, compute_units, out_model);
}

ami_status ami_ane_model_load_path_ex(const char *path, const char *function_name,
                                      uint32_t compute_units,
                                      ami_ane_model **out_model) {
    if (out_model == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_model = NULL;
    MLComputeUnits units = MLComputeUnitsCPUAndNeuralEngine;
    if (path == NULL || path[0] == '\0' || !ane_compute_units_from_raw(compute_units, &units)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        if (ns_path == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        NSURL *src_url = [[NSURL fileURLWithPath:ns_path] URLByStandardizingPath];
        if (src_url == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }

        NSString *ns_fn = nil;
        if (function_name != NULL && function_name[0] != '\0') {
            ns_fn = [NSString stringWithUTF8String:function_name];
            if (ns_fn == nil) {
                return AMI_ERR_INVALID_ARGUMENT;
            }
        }

        NSURL *compiled_url = src_url;
        bool owns_temp = false;
        NSString *ext = src_url.pathExtension.lowercaseString;
        if (![ext isEqualToString:@"mlmodelc"]) {
            dispatch_semaphore_t compile_sem = dispatch_semaphore_create(0);
            if (compile_sem == NULL) {
                return AMI_ERR_OUT_OF_MEMORY;
            }
            __block NSURL *out_compiled = nil;
            [MLModel compileModelAtURL:src_url
                     completionHandler:^(NSURL *c_url, NSError *c_err) {
                         (void)c_err;
                         out_compiled = c_url;
                         dispatch_semaphore_signal(compile_sem);
                     }];
            dispatch_semaphore_wait(compile_sem, DISPATCH_TIME_FOREVER);
            if (out_compiled == nil) {
                return AMI_ERR_ANE;
            }
            compiled_url = out_compiled;
            owns_temp = true;
        }

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = units;
        if (ns_fn != nil) {
            cfg.functionName = ns_fn;
        }

        NSError *error = nil;
        MLModel *ml_model = [MLModel modelWithContentsOfURL:compiled_url
                                              configuration:cfg
                                                      error:&error];
        if (ml_model == nil) {
            if (owns_temp) {
                [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:nil];
            }
            return AMI_ERR_ANE;
        }

        ami_ane_model *model = calloc(1, sizeof *model);
        if (model == NULL) {
            ml_model = nil;
            if (owns_temp) {
                [[NSFileManager defaultManager] removeItemAtURL:compiled_url error:nil];
            }
            return AMI_ERR_OUT_OF_MEMORY;
        }
        model->info.compute_units = compute_units;

        // Inspect MLComputePlan to record ANE / GPU / CPU layer placement.
        dispatch_semaphore_t plan_sem = dispatch_semaphore_create(0);
        if (plan_sem != NULL) {
            __block MLComputePlan *loaded_plan = nil;
            [MLComputePlan loadContentsOfURL:compiled_url
                               configuration:cfg
                           completionHandler:^(MLComputePlan *p, NSError *p_err) {
                               (void)p_err;
                               loaded_plan = p;
                               dispatch_semaphore_signal(plan_sem);
                           }];
            dispatch_semaphore_wait(plan_sem, DISPATCH_TIME_FOREVER);
            populate_compute_plan_info(loaded_plan, &model->info);
        }

        model->ml_model = (__bridge_retained void *)ml_model;
        if (owns_temp) {
            model->temp_compiled_url = (__bridge_retained void *)compiled_url;
        }
        *out_model = model;
    }
    return AMI_OK;
}

ami_status ami_ane_model_load_spec_data(const void *spec_bytes, size_t spec_len,
                                        uint32_t compute_units,
                                        ami_ane_model **out_model) {
    if (out_model == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_model = NULL;
    MLComputeUnits units = MLComputeUnitsCPUAndNeuralEngine;
    if (spec_bytes == NULL || spec_len == 0 || !ane_compute_units_from_raw(compute_units, &units)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSData *data = [NSData dataWithBytes:spec_bytes length:spec_len];
        if (data == nil) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        NSError *error = nil;
        MLModelAsset *asset = [MLModelAsset modelAssetWithSpecificationData:data error:&error];
        if (asset == nil) {
            return AMI_ERR_ANE;
        }

        MLModelConfiguration *cfg = [[MLModelConfiguration alloc] init];
        cfg.computeUnits = units;

        dispatch_semaphore_t load_sem = dispatch_semaphore_create(0);
        if (load_sem == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        __block MLModel *loaded_model = nil;
        [MLModel loadModelAsset:asset
                  configuration:cfg
              completionHandler:^(MLModel *m, NSError *m_err) {
                  (void)m_err;
                  loaded_model = m;
                  dispatch_semaphore_signal(load_sem);
              }];
        dispatch_semaphore_wait(load_sem, DISPATCH_TIME_FOREVER);
        if (loaded_model == nil) {
            return AMI_ERR_ANE;
        }

        ami_ane_model *model = calloc(1, sizeof *model);
        if (model == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        model->info.compute_units = compute_units;

        dispatch_semaphore_t plan_sem = dispatch_semaphore_create(0);
        if (plan_sem != NULL) {
            __block MLComputePlan *loaded_plan = nil;
            [MLComputePlan loadModelAsset:asset
                            configuration:cfg
                        completionHandler:^(MLComputePlan *p, NSError *p_err) {
                            (void)p_err;
                            loaded_plan = p;
                            dispatch_semaphore_signal(plan_sem);
                        }];
            dispatch_semaphore_wait(plan_sem, DISPATCH_TIME_FOREVER);
            populate_compute_plan_info(loaded_plan, &model->info);
        }

        model->ml_model = (__bridge_retained void *)loaded_model;
        *out_model = model;
    }
    return AMI_OK;
}

void ami_ane_model_destroy(ami_ane_model *model) {
    if (model == NULL) {
        return;
    }
    if (model->ml_model != NULL) {
        CFRelease(model->ml_model);
        model->ml_model = NULL;
    }
    if (model->temp_compiled_url != NULL) {
        @autoreleasepool {
            NSURL *url = (__bridge_transfer NSURL *)model->temp_compiled_url;
            model->temp_compiled_url = NULL;
            [[NSFileManager defaultManager] removeItemAtURL:url error:nil];
        }
    }
    free(model);
}

ami_status ami_ane_model_query(const ami_ane_model *model, ami_ane_model_info *out_info) {
    if (model == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_info = model->info;
    return AMI_OK;
}

ami_status ami_ane_model_io_counts(const ami_ane_model *model,
                                   uint32_t *out_input_count,
                                   uint32_t *out_output_count) {
    if (out_input_count != NULL) *out_input_count = 0;
    if (out_output_count != NULL) *out_output_count = 0;
    if (model == NULL || model->ml_model == NULL ||
        out_input_count == NULL || out_output_count == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        MLModel *ml_model = (__bridge MLModel *)model->ml_model;
        MLModelDescription *desc = ml_model.modelDescription;
        if (desc == nil) {
            return AMI_ERR_ANE;
        }
        *out_input_count = (uint32_t)desc.inputDescriptionsByName.count;
        *out_output_count = (uint32_t)desc.outputDescriptionsByName.count;
    }
    return AMI_OK;
}

ami_status ami_ane_model_io_name(const ami_ane_model *model, uint32_t is_output,
                                 uint32_t index, char *out_buf, size_t out_cap) {
    if (out_buf != NULL && out_cap > 0) {
        out_buf[0] = '\0';
    }
    if (model == NULL || model->ml_model == NULL || out_buf == NULL || out_cap == 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        MLModel *ml_model = (__bridge MLModel *)model->ml_model;
        MLModelDescription *desc = ml_model.modelDescription;
        if (desc == nil) {
            return AMI_ERR_ANE;
        }
        NSDictionary<NSString *, MLFeatureDescription *> *dict =
            (is_output != 0) ? desc.outputDescriptionsByName : desc.inputDescriptionsByName;
        if (dict == nil || index >= (uint32_t)dict.count) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        NSArray<NSString *> *keys =
            [dict.allKeys sortedArrayUsingSelector:@selector(compare:)];
        const char *utf8 = keys[index].UTF8String;
        if (utf8 == NULL) {
            return AMI_ERR_ANE;
        }
        if (strlcpy(out_buf, utf8, out_cap) >= out_cap) {
            out_buf[0] = '\0';
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    return AMI_OK;
}

ami_status ami_ane_tensor_from_kv_tile(const ami_kv_tile *tile, const uint32_t *shape,
                                       size_t rank, ami_ane_tensor **out_tensor) {
    if (out_tensor == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_tensor = NULL;
    if (tile == NULL || tile->pixel_buffer == NULL || shape == NULL ||
        rank < 2 || rank > AMI_ANE_MAX_DIMS) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    uint64_t leading_rows = 1u;
    for (size_t i = 0; i < rank; i++) {
        if (shape[i] == 0u) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        if (i + 1u < rank) {
            if (leading_rows > UINT64_MAX / (uint64_t)shape[i]) {
                return AMI_ERR_INVALID_ARGUMENT;
            }
            leading_rows *= (uint64_t)shape[i];
        }
    }
    if (shape[rank - 1u] != tile->row_f16 || leading_rows != (uint64_t)tile->rows) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray arrayWithCapacity:rank];
        for (size_t i = 0; i < rank; i++) {
            [ns_shape addObject:@(shape[i])];
        }

        // Balance CoreVideo ownership whether CoreML retains or adopts pixel_buffer.
        CFIndex rc_before = CFGetRetainCount(tile->pixel_buffer);
        MLMultiArray *arr = [[MLMultiArray alloc] initWithPixelBuffer:tile->pixel_buffer
                                                                shape:ns_shape];
        if (arr == nil) {
            return AMI_ERR_ANE;
        }
        CFIndex rc_after = CFGetRetainCount(tile->pixel_buffer);
        if (rc_after <= rc_before) {
            CVPixelBufferRetain(tile->pixel_buffer);
        }

        ami_ane_tensor *tensor = calloc(1, sizeof *tensor);
        if (tensor == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        tensor->ml_array = (__bridge_retained void *)arr;
        tensor->pixel_buffer = tile->pixel_buffer;
        tensor->rank = (uint32_t)rank;
        for (size_t i = 0; i < rank; i++) {
            tensor->shape[i] = shape[i];
        }
        *out_tensor = tensor;
    }
    return AMI_OK;
}

void ami_ane_tensor_destroy(ami_ane_tensor *tensor) {
    if (tensor == NULL) {
        return;
    }
    if (tensor->ml_array != NULL) {
        CFRelease(tensor->ml_array);
        tensor->ml_array = NULL;
    }
    free(tensor);
}

ami_status ami_ane_plan_create(const ami_ane_binding *inputs, size_t input_count,
                               const ami_ane_binding *outputs, size_t output_count,
                               ami_ane_plan **out_plan) {
    if (out_plan == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_plan = NULL;
    if (input_count == 0 || input_count > AMI_ANE_MAX_BINDINGS || inputs == NULL ||
        output_count == 0 || output_count > AMI_ANE_MAX_BINDINGS || outputs == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < input_count; i++) {
        if (inputs[i].name == NULL || inputs[i].name[0] == '\0' || inputs[i].tensor == NULL ||
            inputs[i].tensor->ml_array == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    for (size_t i = 0; i < output_count; i++) {
        if (outputs[i].name == NULL || outputs[i].name[0] == '\0' || outputs[i].tensor == NULL ||
            outputs[i].tensor->ml_array == NULL) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }

    @autoreleasepool {
        NSMutableDictionary<NSString *, MLFeatureValue *> *in_dict =
            [NSMutableDictionary dictionaryWithCapacity:input_count];
        for (size_t i = 0; i < input_count; i++) {
            NSString *key = [NSString stringWithUTF8String:inputs[i].name];
            if (key == nil) {
                return AMI_ERR_INVALID_ARGUMENT;
            }
            MLMultiArray *arr = (__bridge MLMultiArray *)inputs[i].tensor->ml_array;
            MLFeatureValue *fv = [MLFeatureValue featureValueWithMultiArray:arr];
            if (fv == nil) {
                return AMI_ERR_ANE;
            }
            in_dict[key] = fv;
        }

        NSError *error = nil;
        MLDictionaryFeatureProvider *provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:in_dict error:&error];
        if (provider == nil) {
            return AMI_ERR_ANE;
        }

        NSMutableDictionary<NSString *, id> *out_backings =
            [NSMutableDictionary dictionaryWithCapacity:output_count];
        ami_ane_plan *plan = calloc(1, sizeof *plan);
        if (plan == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }

        for (size_t i = 0; i < output_count; i++) {
            NSString *key = [NSString stringWithUTF8String:outputs[i].name];
            if (key == nil) {
                ami_ane_plan_destroy(plan);
                return AMI_ERR_INVALID_ARGUMENT;
            }
            MLMultiArray *arr = (__bridge MLMultiArray *)outputs[i].tensor->ml_array;
            out_backings[key] = arr;
            plan->output_names[i] = (__bridge_retained void *)key;
            plan->output_arrays[i] = (__bridge_retained void *)arr;
            plan->output_pbs[i] = outputs[i].tensor->pixel_buffer;
            plan->output_count = (uint32_t)(i + 1u);
        }

        MLPredictionOptions *opts = [[MLPredictionOptions alloc] init];
        opts.outputBackings = out_backings;

        plan->input_provider = (__bridge_retained void *)provider;
        plan->options = (__bridge_retained void *)opts;
        *out_plan = plan;
    }
    return AMI_OK;
}

void ami_ane_plan_destroy(ami_ane_plan *plan) {
    if (plan == NULL) {
        return;
    }
    for (uint32_t i = 0; i < plan->output_count; i++) {
        if (plan->output_names[i] != NULL) CFRelease(plan->output_names[i]);
        if (plan->output_arrays[i] != NULL) CFRelease(plan->output_arrays[i]);
    }
    if (plan->options != NULL) CFRelease(plan->options);
    if (plan->input_provider != NULL) CFRelease(plan->input_provider);
    free(plan);
}

ami_status ami_ane_model_predict(const ami_ane_model *model, const ami_ane_plan *plan,
                                 ami_ane_predict_info *out_info) {
    if (out_info != NULL) {
        memset(out_info, 0, sizeof *out_info);
    }
    if (model == NULL || model->ml_model == NULL || plan == NULL ||
        plan->input_provider == NULL || plan->options == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        MLModel *ml_model = (__bridge MLModel *)model->ml_model;
        MLDictionaryFeatureProvider *provider =
            (__bridge MLDictionaryFeatureProvider *)plan->input_provider;
        MLPredictionOptions *opts = (__bridge MLPredictionOptions *)plan->options;

        NSError *error = nil;
        id<MLFeatureProvider> result = [ml_model predictionFromFeatures:provider
                                                                options:opts
                                                                  error:&error];
        if (result == nil) {
            return AMI_ERR_ANE;
        }

        uint32_t matched = 0;
        for (uint32_t i = 0; i < plan->output_count; i++) {
            NSString *name = (__bridge NSString *)plan->output_names[i];
            MLMultiArray *expected_arr = (__bridge MLMultiArray *)plan->output_arrays[i];
            MLFeatureValue *fv = [result featureValueForName:name];
            MLMultiArray *got_arr = (fv != nil) ? fv.multiArrayValue : nil;
            if (got_arr == nil) {
                return AMI_ERR_ANE;
            }
            IOSurfaceRef got_surface =
                (got_arr.pixelBuffer != NULL) ? CVPixelBufferGetIOSurface(got_arr.pixelBuffer) : NULL;
            IOSurfaceRef exp_surface =
                (plan->output_pbs[i] != NULL) ? CVPixelBufferGetIOSurface(plan->output_pbs[i]) : NULL;
            if (got_arr == expected_arr ||
                (got_arr.pixelBuffer != NULL && got_arr.pixelBuffer == plan->output_pbs[i]) ||
                (got_surface != NULL && got_surface == exp_surface)) {
                matched++;
            } else {
                // CoreML did not honor outputBacking (e.g. CPUOnly fallback); copy into destination
                // while accurately reporting zero-copy miss via `outputs_zero_copy_matched`.
                [got_arr transferToMultiArray:expected_arr];
            }
        }
        if (out_info != NULL) {
            out_info->outputs_zero_copy_matched = matched;
            out_info->outputs_total = plan->output_count;
        }
    }
    return AMI_OK;
}

ami_status ami_ane_model_predict_oneshot(const ami_ane_model *model,
                                         const ami_ane_binding *inputs, size_t input_count,
                                         const ami_ane_binding *outputs, size_t output_count,
                                         ami_ane_predict_info *out_info) {
    ami_ane_plan *plan = NULL;
    ami_status st = ami_ane_plan_create(inputs, input_count, outputs, output_count, &plan);
    if (st != AMI_OK) {
        return st;
    }
    st = ami_ane_model_predict(model, plan, out_info);
    ami_ane_plan_destroy(plan);
    return st;
}

ami_status ami_accelerate_sgemm(uint32_t m, uint32_t n, uint32_t k,
                                float alpha, const float *a, const float *b,
                                float beta, float *c, uint32_t transpose_b) {
    if (a == NULL || b == NULL || c == NULL ||
        m == 0 || n == 0 || k == 0 ||
        m > (uint32_t)INT_MAX || n > (uint32_t)INT_MAX || k > (uint32_t)INT_MAX) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans,
                transpose_b != 0 ? CblasTrans : CblasNoTrans,
                (int)m,
                (int)n,
                (int)k,
                alpha,
                a,
                (int)k,
                b,
                transpose_b != 0 ? (int)k : (int)n,
                beta,
                c,
                (int)n);
    return AMI_OK;
}
