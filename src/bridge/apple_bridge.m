// apple_bridge.m — Objective-C (ARC) implementation of apple_bridge.h.
//
// Handles own their Objective-C objects through __bridge_retained pointers and
// release them in the matching *_destroy function.

#define __STDC_WANT_LIB_EXT1__ 1
#define ACCELERATE_NEW_LAPACK 1
#import <Accelerate/Accelerate.h>
#import <CoreGraphics/CoreGraphics.h>
#import <CoreML/CoreML.h>
#import <CoreMedia/CoreMedia.h>
#import <CoreVideo/CoreVideo.h>
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#import <Network/Network.h>
#import <Security/Security.h>
#import <VideoToolbox/VideoToolbox.h>

#include <CommonCrypto/CommonCryptor.h>
#include <CommonCrypto/CommonDigest.h>
#include <CommonCrypto/CommonHMAC.h>
#include <CommonCrypto/CommonKeyDerivation.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/processor_info.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <termios.h>
#include <unistd.h>

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

static uint8_t highest_apple_family(id<MTLDevice> device) {
    for (NSInteger family = 1011; family >= 1007; family--) {
        if ([device supportsFamily:(MTLGPUFamily)family]) {
            return (uint8_t)(family - 1000);
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
    @autoreleasepool {
        CFRelease(device->mtl_device);
    }
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
        const uint8_t family = highest_apple_family(mtl_device);
        out_info->apple_gpu_family = (uint32_t)family;
        out_info->gpu_apple_family = family;
        out_info->ane_core_count = neural_engine_cores();
        out_info->supports_apple9 =
            [mtl_device supportsFamily:(MTLGPUFamily)1009] ? 1 : 0;
        out_info->supports_apple10 =
            [mtl_device supportsFamily:(MTLGPUFamily)1010] ? 1 : 0;
        out_info->supports_apple11 =
            [mtl_device supportsFamily:(MTLGPUFamily)1011] ? 1 : 0;
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
    @autoreleasepool {
        CFRelease(library->mtl_library);
    }
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
    @autoreleasepool {
        CFRelease(buffer->mtl_buffer);
    }
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
    @autoreleasepool {
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
    @autoreleasepool {
        CFRelease(pipeline->mtl_pipeline);
    }
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
        for (uint32_t i = 0; i < AMI_MAX_FRAMES_IN_FLIGHT; i++) {
            release_frame_slot(&compute->slots[i]);
        }
        if (compute->mtl_event != NULL) CFRelease(compute->mtl_event);
        if (compute->mtl_internal_residency_set != NULL) CFRelease(compute->mtl_internal_residency_set);
        if (compute->mtl_user_residency_set != NULL) CFRelease(compute->mtl_user_residency_set);
        if (compute->mtl_queue != NULL) CFRelease(compute->mtl_queue);
    }
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

_Static_assert(sizeof(ami_indirect_dispatch_args) == sizeof(MTLDispatchThreadgroupsIndirectArguments),
               "ami_indirect_dispatch_args must match MTLDispatchThreadgroupsIndirectArguments");

ami_status ami_mtl4_compute_dispatch_indirect(ami_compute *compute, const ami_pipeline *pipeline,
                                              const ami_buffer_binding *bindings,
                                              size_t binding_count,
                                              const void *uniform_bytes, size_t uniform_len,
                                              const ami_buffer *indirect_buffer,
                                              uint64_t indirect_offset,
                                              uint64_t tg_x, uint64_t tg_y, uint64_t tg_z,
                                              uint32_t insert_barrier_before) {
    if (compute == NULL || compute->active_slot < 0 || compute->active_encoder == NULL ||
        pipeline == NULL || indirect_buffer == NULL ||
        tg_x == 0 || tg_y == 0 || tg_z == 0 ||
        (binding_count > 0 && bindings == NULL) ||
        (uniform_len > 0 && uniform_bytes == NULL) ||
        uniform_len > AMI_UNIFORM_STRIDE ||
        (indirect_offset & 3ull) != 0) {
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
        id<MTLBuffer> ind_buf = buffer_of(indirect_buffer);
        if (ind_buf == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        uint64_t ind_len = (uint64_t)ind_buf.length;
        if (indirect_offset > ind_len ||
            (ind_len - indirect_offset) < (uint64_t)sizeof(MTLDispatchThreadgroupsIndirectArguments)) {
            return AMI_ERR_INVALID_ARGUMENT;
        }

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

        MTLGPUAddress indirect_addr = ind_buf.gpuAddress + indirect_offset;
        MTLSize tg = MTLSizeMake((NSUInteger)tg_x, (NSUInteger)tg_y, (NSUInteger)tg_z);
        [encoder dispatchThreadgroupsWithIndirectBuffer:indirect_addr
                                  threadsPerThreadgroup:tg];
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

/* -------------------------------------------------------------------------
 * Zero-copy multimodal vision/video ingress bridge (M6.5 G14)
 * ------------------------------------------------------------------------- */

struct ami_vision_frame {
    CVPixelBufferRef pixel_buffer; // retained via CoreVideo
    IOSurfaceRef surface;          // borrowed from pixel_buffer
    void *mtl_textures[2];         // id<MTLTexture>, retained
    uint32_t format;
    uint32_t cv_pixel_format;
    uint32_t width;
    uint32_t height;
    uint32_t plane_count;
    uint32_t is_full_range;
    uint32_t zero_copy_verified;
    uint32_t is_page_aligned;
    uint32_t iosurface_id;
    uint32_t vt_decoded;
    uint64_t base_address;
    uint64_t alloc_size;
    uint64_t surface_alloc_size;
    uint32_t plane_width[2];
    uint32_t plane_height[2];
    uint32_t plane_bytes_per_row[2];
    int64_t pts_value;
    int32_t pts_timescale;
};

struct ami_vt_clip {
    CMVideoFormatDescriptionRef format_desc; // retained
    CMSampleBufferRef *samples;              // owned array of retained CMSampleBuffers
    uint32_t codec;
    uint32_t width;
    uint32_t height;
    uint32_t frame_count;
    uint32_t fps;
    uint32_t using_hardware_encoder;
    uint64_t total_compressed_bytes;
};

struct ami_vt_decode_session {
    void *mtl_device;                        // id<MTLDevice>, retained
    VTDecompressionSessionRef vt_session;    // retained
    uint32_t codec;
    uint32_t output_format;
    uint32_t width;
    uint32_t height;
    uint32_t using_hardware_decoder;
};

static uint64_t align_up_u64(uint64_t val, uint64_t align) {
    return ((val + align - 1u) / align) * align;
}

static id<MTLTexture> vision_plane_texture_of(const ami_vision_frame *frame, uint32_t plane) {
    if (frame == NULL || plane >= frame->plane_count || frame->mtl_textures[plane] == NULL) {
        return nil;
    }
    return (__bridge id<MTLTexture>)frame->mtl_textures[plane];
}

static ami_status vision_frame_init_from_mtl_device(id<MTLDevice> mtl_device,
                                                    CVPixelBufferRef pixel_buffer,
                                                    uint64_t requested_alloc_bytes,
                                                    uint32_t vt_decoded,
                                                    int64_t pts_value,
                                                    int32_t pts_timescale,
                                                    ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (mtl_device == nil || pixel_buffer == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const OSType cv_fmt = CVPixelBufferGetPixelFormatType(pixel_buffer);
    uint32_t format = 0;
    uint32_t plane_count = 0;
    uint32_t is_full_range = 1u;
    MTLPixelFormat mtl_fmts[2] = { MTLPixelFormatInvalid, MTLPixelFormatInvalid };

    if (cv_fmt == kCVPixelFormatType_32BGRA) {
        format = AMI_VISION_FMT_BGRA8;
        plane_count = 1u;
        is_full_range = 1u;
        mtl_fmts[0] = MTLPixelFormatBGRA8Unorm;
    } else if (cv_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange) {
        format = AMI_VISION_FMT_NV12;
        plane_count = 2u;
        is_full_range = 1u;
        mtl_fmts[0] = MTLPixelFormatR8Unorm;
        mtl_fmts[1] = MTLPixelFormatRG8Unorm;
    } else if (cv_fmt == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange) {
        format = AMI_VISION_FMT_NV12;
        plane_count = 2u;
        is_full_range = 0u;
        mtl_fmts[0] = MTLPixelFormatR8Unorm;
        mtl_fmts[1] = MTLPixelFormatRG8Unorm;
    } else if (cv_fmt == kCVPixelFormatType_OneComponent16Half) {
        format = AMI_VISION_FMT_F16;
        plane_count = 1u;
        is_full_range = 1u;
        mtl_fmts[0] = MTLPixelFormatR16Float;
    } else {
        return AMI_ERR_UNSUPPORTED;
    }

    IOSurfaceRef surface = CVPixelBufferGetIOSurface(pixel_buffer);
    if (surface == NULL) {
        return AMI_ERR_VISION;
    }

    const size_t width = CVPixelBufferGetWidth(pixel_buffer);
    const size_t height = CVPixelBufferGetHeight(pixel_buffer);
    if (width == 0 || height == 0 || width > AMI_MAX_TEXTURE_DIM || height > AMI_MAX_TEXTURE_DIM) {
        return AMI_ERR_VISION;
    }

    // Lock read-only briefly so freshly VideoToolbox-decoded IOSurfaces establish
    // their CPU virtual address mapping before querying base address and plane strides.
    CVReturn lock_ret = CVPixelBufferLockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);
    if (lock_ret != kCVReturnSuccess) {
        return AMI_ERR_VISION;
    }

    void *base_addr = IOSurfaceGetBaseAddress(surface);
    const size_t surface_alloc = IOSurfaceGetAllocSize(surface);
    const IOSurfaceID sid = IOSurfaceGetID(surface);

    uint32_t plane_w[2] = { 0, 0 };
    uint32_t plane_h[2] = { 0, 0 };
    uint32_t plane_bpr[2] = { 0, 0 };
    bool planes_ok = true;

    if (plane_count == 1u) {
        plane_w[0] = (uint32_t)width;
        plane_h[0] = (uint32_t)height;
        plane_bpr[0] = (uint32_t)CVPixelBufferGetBytesPerRow(pixel_buffer);
        if (plane_bpr[0] == 0) {
            planes_ok = false;
        }
    } else {
        if (CVPixelBufferGetPlaneCount(pixel_buffer) < 2u) {
            planes_ok = false;
        } else {
            for (uint32_t p = 0; p < 2u; p++) {
                size_t pw = CVPixelBufferGetWidthOfPlane(pixel_buffer, p);
                size_t ph = CVPixelBufferGetHeightOfPlane(pixel_buffer, p);
                size_t pbpr = CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer, p);
                if (pw == 0 || ph == 0 || pbpr == 0 ||
                    pw > AMI_MAX_TEXTURE_DIM || ph > AMI_MAX_TEXTURE_DIM) {
                    planes_ok = false;
                    break;
                }
                plane_w[p] = (uint32_t)pw;
                plane_h[p] = (uint32_t)ph;
                plane_bpr[p] = (uint32_t)pbpr;
            }
        }
    }

    CVPixelBufferUnlockBaseAddress(pixel_buffer, kCVPixelBufferLock_ReadOnly);

    if (!planes_ok || base_addr == NULL || surface_alloc == 0) {
        return AMI_ERR_VISION;
    }
    const bool base_aligned = (((uintptr_t)base_addr % AMI_PAGE_SIZE) == 0);
    const bool size_aligned = ((surface_alloc % AMI_PAGE_SIZE) == 0);
    if (!base_aligned) {
        return AMI_ERR_VISION;
    }
    if (requested_alloc_bytes > 0 &&
        (surface_alloc < requested_alloc_bytes || !size_aligned)) {
        return AMI_ERR_VISION;
    }

    @autoreleasepool {
        id<MTLTexture> textures[2] = { nil, nil };
        MTLTextureUsage usage =
            (vt_decoded != 0 || format == AMI_VISION_FMT_NV12)
                ? MTLTextureUsageShaderRead
                : (MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

        for (uint32_t p = 0; p < plane_count; p++) {
            MTLTextureDescriptor *desc =
                [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:mtl_fmts[p]
                                                                   width:(NSUInteger)plane_w[p]
                                                                  height:(NSUInteger)plane_h[p]
                                                               mipmapped:NO];
            desc.storageMode = MTLStorageModeShared;
            desc.usage = usage;
            id<MTLTexture> tex = [mtl_device newTextureWithDescriptor:desc
                                                            iosurface:surface
                                                                plane:(NSUInteger)p];
            if (tex == nil) {
                return AMI_ERR_VISION;
            }
            // Prove zero-copy IOSurface physical page identity at the Metal handle level.
            IOSurfaceRef tex_surface = tex.iosurface;
            if (tex_surface == NULL || tex_surface != surface ||
                IOSurfaceGetID(tex_surface) != sid ||
                tex.iosurfacePlane != (NSUInteger)p) {
                return AMI_ERR_VISION;
            }
            textures[p] = tex;
        }

        ami_vision_frame *frame = calloc(1, sizeof *frame);
        if (frame == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        CVPixelBufferRetain(pixel_buffer);
        frame->pixel_buffer = pixel_buffer;
        frame->surface = surface;
        for (uint32_t p = 0; p < plane_count; p++) {
            frame->mtl_textures[p] = (__bridge_retained void *)textures[p];
        }
        frame->format = format;
        frame->cv_pixel_format = (uint32_t)cv_fmt;
        frame->width = (uint32_t)width;
        frame->height = (uint32_t)height;
        frame->plane_count = plane_count;
        frame->is_full_range = is_full_range;
        frame->zero_copy_verified = 1u;
        frame->is_page_aligned = (base_aligned && size_aligned) ? 1u : 0u;
        frame->iosurface_id = (uint32_t)sid;
        frame->vt_decoded = vt_decoded;
        frame->base_address = (uint64_t)(uintptr_t)base_addr;
        frame->alloc_size = (requested_alloc_bytes > 0) ? requested_alloc_bytes : (uint64_t)surface_alloc;
        frame->surface_alloc_size = (uint64_t)surface_alloc;
        for (uint32_t p = 0; p < 2u; p++) {
            frame->plane_width[p] = plane_w[p];
            frame->plane_height[p] = plane_h[p];
            frame->plane_bytes_per_row[p] = plane_bpr[p];
        }
        frame->pts_value = pts_value;
        frame->pts_timescale = pts_timescale;
        *out_frame = frame;
    }
    return AMI_OK;
}

static ami_status create_iosurface_pixel_buffer(uint32_t format, uint32_t width, uint32_t height,
                                                CVPixelBufferRef *out_pb,
                                                uint64_t *out_alloc_bytes) {
    if (out_pb == NULL || out_alloc_bytes == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_pb = NULL;
    *out_alloc_bytes = 0;
    if (width == 0 || height == 0 || width > AMI_MAX_TEXTURE_DIM || height > AMI_MAX_TEXTURE_DIM) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSDictionary *surface_props = nil;
        uint64_t alloc_bytes = 0;

        if (format == AMI_VISION_FMT_BGRA8) {
            const uint64_t bpr = align_up_u64((uint64_t)width * 4u, 64u);
            const uint64_t data_bytes = bpr * (uint64_t)height;
            alloc_bytes = align_up_u64(data_bytes, AMI_PAGE_SIZE);
            surface_props = @{
                (__bridge id)kIOSurfaceWidth: @(width),
                (__bridge id)kIOSurfaceHeight: @(height),
                (__bridge id)kIOSurfaceBytesPerElement: @4,
                (__bridge id)kIOSurfaceBytesPerRow: @(bpr),
                (__bridge id)kIOSurfaceAllocSize: @(alloc_bytes),
                (__bridge id)kIOSurfacePixelFormat: @(kCVPixelFormatType_32BGRA),
            };
        } else if (format == AMI_VISION_FMT_NV12) {
            if ((width & 1u) != 0 || (height & 1u) != 0 || width < 2u || height < 2u) {
                return AMI_ERR_INVALID_ARGUMENT;
            }
            const uint64_t w0 = width;
            const uint64_t h0 = height;
            const uint64_t bpr0 = align_up_u64(w0, 64u);
            const uint64_t size0 = align_up_u64(bpr0 * h0, 64u);

            const uint64_t w1 = width / 2u;
            const uint64_t h1 = height / 2u;
            const uint64_t bpr1 = align_up_u64(w1 * 2u, 64u);
            const uint64_t size1 = align_up_u64(bpr1 * h1, 64u);

            const uint64_t offset0 = 0u;
            const uint64_t offset1 = size0;
            alloc_bytes = align_up_u64(offset1 + size1, AMI_PAGE_SIZE);

            NSDictionary *plane0 = @{
                (__bridge id)kIOSurfacePlaneWidth: @(w0),
                (__bridge id)kIOSurfacePlaneHeight: @(h0),
                (__bridge id)kIOSurfacePlaneBytesPerElement: @1,
                (__bridge id)kIOSurfacePlaneBytesPerRow: @(bpr0),
                (__bridge id)kIOSurfacePlaneOffset: @(offset0),
                (__bridge id)kIOSurfacePlaneSize: @(size0),
            };
            NSDictionary *plane1 = @{
                (__bridge id)kIOSurfacePlaneWidth: @(w1),
                (__bridge id)kIOSurfacePlaneHeight: @(h1),
                (__bridge id)kIOSurfacePlaneBytesPerElement: @2,
                (__bridge id)kIOSurfacePlaneBytesPerRow: @(bpr1),
                (__bridge id)kIOSurfacePlaneOffset: @(offset1),
                (__bridge id)kIOSurfacePlaneSize: @(size1),
            };
            surface_props = @{
                (__bridge id)kIOSurfaceWidth: @(width),
                (__bridge id)kIOSurfaceHeight: @(height),
                (__bridge id)kIOSurfacePixelFormat: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
                (__bridge id)kIOSurfaceAllocSize: @(alloc_bytes),
                (__bridge id)kIOSurfacePlaneInfo: @[plane0, plane1],
            };
        } else if (format == AMI_VISION_FMT_F16) {
            const uint64_t bpr = align_up_u64((uint64_t)width * 2u, 64u);
            const uint64_t data_bytes = bpr * (uint64_t)height;
            alloc_bytes = align_up_u64(data_bytes, AMI_PAGE_SIZE);
            surface_props = @{
                (__bridge id)kIOSurfaceWidth: @(width),
                (__bridge id)kIOSurfaceHeight: @(height),
                (__bridge id)kIOSurfaceBytesPerElement: @2,
                (__bridge id)kIOSurfaceBytesPerRow: @(bpr),
                (__bridge id)kIOSurfaceAllocSize: @(alloc_bytes),
                (__bridge id)kIOSurfacePixelFormat: @(kCVPixelFormatType_OneComponent16Half),
            };
        } else {
            return AMI_ERR_INVALID_ARGUMENT;
        }

        IOSurfaceRef surface = IOSurfaceCreate((__bridge CFDictionaryRef)surface_props);
        if (surface == NULL) {
            return AMI_ERR_VISION;
        }
        NSDictionary *pb_attrs = @{
            (__bridge id)kCVPixelBufferMetalCompatibilityKey: @YES,
        };
        CVPixelBufferRef pb = NULL;
        CVReturn cv_ret = CVPixelBufferCreateWithIOSurface(
            kCFAllocatorDefault, surface, (__bridge CFDictionaryRef)pb_attrs, &pb);
        CFRelease(surface);
        if (cv_ret != kCVReturnSuccess || pb == NULL) {
            return AMI_ERR_VISION;
        }

        if (format == AMI_VISION_FMT_NV12) {
            CVBufferSetAttachment(pb, kCVImageBufferYCbCrMatrixKey,
                                  kCVImageBufferYCbCrMatrix_ITU_R_709_2,
                                  kCVAttachmentMode_ShouldPropagate);
            CVBufferSetAttachment(pb, kCVImageBufferColorPrimariesKey,
                                  kCVImageBufferColorPrimaries_ITU_R_709_2,
                                  kCVAttachmentMode_ShouldPropagate);
            CVBufferSetAttachment(pb, kCVImageBufferTransferFunctionKey,
                                  kCVImageBufferTransferFunction_ITU_R_709_2,
                                  kCVAttachmentMode_ShouldPropagate);
        }

        *out_pb = pb;
        *out_alloc_bytes = alloc_bytes;
    }
    return AMI_OK;
}

ami_status ami_vision_frame_create(ami_device *device, uint32_t format,
                                   uint32_t width, uint32_t height,
                                   ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (device == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    CVPixelBufferRef pb = NULL;
    uint64_t alloc_bytes = 0;
    ami_status st = create_iosurface_pixel_buffer(format, width, height, &pb, &alloc_bytes);
    if (st != AMI_OK) {
        return st;
    }

    st = vision_frame_init_from_mtl_device(device_of(device), pb, alloc_bytes, 0u, 0, 0, out_frame);
    CVPixelBufferRelease(pb);
    return st;
}

void ami_vision_frame_destroy(ami_vision_frame *frame) {
    if (frame == NULL) {
        return;
    }
    for (uint32_t p = 0; p < 2u; p++) {
        if (frame->mtl_textures[p] != NULL) {
            CFRelease(frame->mtl_textures[p]);
            frame->mtl_textures[p] = NULL;
        }
    }
    if (frame->pixel_buffer != NULL) {
        CVPixelBufferRelease(frame->pixel_buffer);
        frame->pixel_buffer = NULL;
    }
    free(frame);
}

ami_status ami_vision_frame_query(const ami_vision_frame *frame,
                                  ami_vision_frame_info *out_info) {
    if (frame == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_info, 0, sizeof *out_info);
    out_info->format = frame->format;
    out_info->cv_pixel_format = frame->cv_pixel_format;
    out_info->width = frame->width;
    out_info->height = frame->height;
    out_info->plane_count = frame->plane_count;
    out_info->is_full_range = frame->is_full_range;
    out_info->zero_copy_verified = frame->zero_copy_verified;
    out_info->is_page_aligned = frame->is_page_aligned;
    out_info->iosurface_id = frame->iosurface_id;
    out_info->vt_decoded = frame->vt_decoded;
    out_info->base_address = frame->base_address;
    out_info->alloc_size = frame->alloc_size;
    out_info->surface_alloc_size = frame->surface_alloc_size;
    for (uint32_t p = 0; p < 2u; p++) {
        out_info->plane_width[p] = frame->plane_width[p];
        out_info->plane_height[p] = frame->plane_height[p];
        out_info->plane_bytes_per_row[p] = frame->plane_bytes_per_row[p];
        out_info->plane_resource_id[p] = ami_vision_frame_plane_resource_id(frame, p);
    }
    out_info->pts_value = frame->pts_value;
    out_info->pts_timescale = frame->pts_timescale;
    return AMI_OK;
}

uint64_t ami_vision_frame_plane_resource_id(const ami_vision_frame *frame, uint32_t plane) {
    id<MTLTexture> tex = vision_plane_texture_of(frame, plane);
    if (tex == nil) {
        return 0;
    }
    return tex.gpuResourceID._impl;
}

ami_status ami_vision_frame_lock_plane(ami_vision_frame *frame, uint32_t plane,
                                       uint32_t read_only, void **out_base_ptr,
                                       size_t *out_bytes_per_row, uint32_t *out_height) {
    if (out_base_ptr != NULL) *out_base_ptr = NULL;
    if (out_bytes_per_row != NULL) *out_bytes_per_row = 0;
    if (out_height != NULL) *out_height = 0;
    if (frame == NULL || plane >= frame->plane_count ||
        out_base_ptr == NULL || out_bytes_per_row == NULL || out_height == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    CVPixelBufferLockFlags lock_flags = (read_only != 0) ? kCVPixelBufferLock_ReadOnly : 0;
    CVReturn ret = CVPixelBufferLockBaseAddress(frame->pixel_buffer, lock_flags);
    if (ret != kCVReturnSuccess) {
        return AMI_ERR_VISION;
    }
    void *base = NULL;
    size_t bpr = 0;
    if (frame->plane_count == 1u) {
        base = CVPixelBufferGetBaseAddress(frame->pixel_buffer);
        bpr = CVPixelBufferGetBytesPerRow(frame->pixel_buffer);
    } else {
        base = CVPixelBufferGetBaseAddressOfPlane(frame->pixel_buffer, plane);
        bpr = CVPixelBufferGetBytesPerRowOfPlane(frame->pixel_buffer, plane);
    }
    if (base == NULL || bpr == 0) {
        CVPixelBufferUnlockBaseAddress(frame->pixel_buffer, lock_flags);
        return AMI_ERR_VISION;
    }
    *out_base_ptr = base;
    *out_bytes_per_row = bpr;
    *out_height = frame->plane_height[plane];
    return AMI_OK;
}

ami_status ami_vision_frame_unlock(ami_vision_frame *frame, uint32_t read_only) {
    if (frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    CVPixelBufferLockFlags lock_flags = (read_only != 0) ? kCVPixelBufferLock_ReadOnly : 0;
    CVReturn ret = CVPixelBufferUnlockBaseAddress(frame->pixel_buffer, lock_flags);
    return (ret == kCVReturnSuccess) ? AMI_OK : AMI_ERR_VISION;
}

ami_status ami_vision_frame_probe_ane(const ami_vision_frame *frame,
                                      ami_vision_ane_probe *out_probe) {
    if (frame == NULL || frame->pixel_buffer == NULL || out_probe == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_probe, 0, sizeof *out_probe);

    @autoreleasepool {
        MLFeatureValue *fv = [MLFeatureValue featureValueWithPixelBuffer:frame->pixel_buffer];
        if (fv != nil && fv.imageBufferValue != NULL) {
            out_probe->pixel_buffer_feature_ok = 1u;
            IOSurfaceRef fv_surface = CVPixelBufferGetIOSurface(fv.imageBufferValue);
            if (fv.imageBufferValue == frame->pixel_buffer &&
                fv_surface != NULL &&
                IOSurfaceGetID(fv_surface) == frame->iosurface_id) {
                out_probe->iosurface_identity_matched = 1u;
            }
        }

        // CoreML MLMultiArray(pixelBuffer:shape:) accepts OneComponent16Half with tight rows (ADR-0004).
        if (frame->format == AMI_VISION_FMT_F16 &&
            frame->plane_bytes_per_row[0] == frame->width * 2u) {
            NSArray<NSNumber *> *shape = @[ @(frame->height), @(frame->width) ];
            CFIndex rc_before = CFGetRetainCount(frame->pixel_buffer);
            MLMultiArray *arr = [[MLMultiArray alloc] initWithPixelBuffer:frame->pixel_buffer
                                                                    shape:shape];
            if (arr != nil) {
                CFIndex rc_after = CFGetRetainCount(frame->pixel_buffer);
                if (rc_after <= rc_before) {
                    CVPixelBufferRetain(frame->pixel_buffer);
                }
                CVPixelBufferRef arr_pb = arr.pixelBuffer;
                IOSurfaceRef arr_surface = (arr_pb != NULL) ? CVPixelBufferGetIOSurface(arr_pb) : NULL;
                if (arr_pb == frame->pixel_buffer &&
                    arr_surface != NULL &&
                    IOSurfaceGetID(arr_surface) == frame->iosurface_id) {
                    out_probe->multi_array_zero_copy_ok = 1u;
                    out_probe->iosurface_identity_matched = 1u;
                }
            }
        }
    }
    return AMI_OK;
}

ami_status ami_ane_tensor_from_vision_frame(const ami_vision_frame *frame,
                                            const uint32_t *shape, size_t rank,
                                            ami_ane_tensor **out_tensor) {
    if (out_tensor == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_tensor = NULL;
    if (frame == NULL || frame->pixel_buffer == NULL || shape == NULL ||
        rank < 2 || rank > AMI_ANE_MAX_DIMS) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (frame->format != AMI_VISION_FMT_F16) {
        return AMI_ERR_UNSUPPORTED;
    }
    if (frame->plane_bytes_per_row[0] != frame->width * 2u) {
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
    if (shape[rank - 1u] != frame->width || leading_rows != (uint64_t)frame->height) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSMutableArray<NSNumber *> *ns_shape = [NSMutableArray arrayWithCapacity:rank];
        for (size_t i = 0; i < rank; i++) {
            [ns_shape addObject:@(shape[i])];
        }

        CFIndex rc_before = CFGetRetainCount(frame->pixel_buffer);
        MLMultiArray *arr = [[MLMultiArray alloc] initWithPixelBuffer:frame->pixel_buffer
                                                                shape:ns_shape];
        if (arr == nil) {
            return AMI_ERR_ANE;
        }
        CFIndex rc_after = CFGetRetainCount(frame->pixel_buffer);
        if (rc_after <= rc_before) {
            CVPixelBufferRetain(frame->pixel_buffer);
        }

        ami_ane_tensor *tensor = calloc(1, sizeof *tensor);
        if (tensor == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        tensor->ml_array = (__bridge_retained void *)arr;
        tensor->pixel_buffer = frame->pixel_buffer;
        tensor->rank = (uint32_t)rank;
        for (size_t i = 0; i < rank; i++) {
            tensor->shape[i] = shape[i];
        }
        *out_tensor = tensor;
    }
    return AMI_OK;
}

ami_status ami_compute_make_vision_frames_resident(ami_compute *compute,
                                                 ami_vision_frame *const *frames,
                                                 size_t count) {
    if (compute == NULL || (count > 0 && frames == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return AMI_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (frames[i] == NULL || frames[i]->plane_count == 0) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            for (uint32_t p = 0; p < frames[i]->plane_count; p++) {
                id<MTLTexture> tex = vision_plane_texture_of(frames[i], p);
                if (tex != nil) {
                    [user_res addAllocation:tex];
                }
            }
        }
        [user_res commit];
    }
    return AMI_OK;
}

ami_status ami_compute_remove_vision_frames_resident(ami_compute *compute,
                                                   ami_vision_frame *const *frames,
                                                   size_t count) {
    if (compute == NULL || (count > 0 && frames == NULL)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (count == 0) {
        return AMI_OK;
    }
    for (size_t i = 0; i < count; i++) {
        if (frames[i] == NULL || frames[i]->plane_count == 0) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }
    @autoreleasepool {
        id<MTLResidencySet> user_res =
            (__bridge id<MTLResidencySet>)compute->mtl_user_residency_set;
        for (size_t i = 0; i < count; i++) {
            for (uint32_t p = 0; p < frames[i]->plane_count; p++) {
                id<MTLTexture> tex = vision_plane_texture_of(frames[i], p);
                if (tex != nil) {
                    [user_res removeAllocation:tex];
                }
            }
        }
        [user_res commit];
    }
    return AMI_OK;
}

ami_status ami_compute_encode_dispatch_vision(ami_compute *compute,
                                              const ami_pipeline *pipeline,
                                              const ami_buffer_binding *bindings,
                                              size_t binding_count,
                                              const ami_vision_texture_binding *vision_textures,
                                              size_t vision_texture_count,
                                              const void *uniform_bytes, size_t uniform_len,
                                              uint64_t grid_x, uint64_t grid_y, uint64_t grid_z,
                                              uint64_t tg_x, uint64_t tg_y, uint64_t tg_z,
                                              uint32_t dispatch_mode,
                                              uint32_t insert_barrier_before) {
    if (compute == NULL || compute->active_slot < 0 || compute->active_encoder == NULL ||
        pipeline == NULL || grid_x == 0 || grid_y == 0 || grid_z == 0 ||
        tg_x == 0 || tg_y == 0 || tg_z == 0 ||
        (binding_count > 0 && bindings == NULL) ||
        (vision_texture_count > 0 && vision_textures == NULL) ||
        vision_texture_count > AMI_MAX_TEXTURE_BINDINGS ||
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
    for (size_t i = 0; i < vision_texture_count; i++) {
        if (vision_textures[i].frame == NULL ||
            vision_textures[i].plane >= vision_textures[i].frame->plane_count ||
            vision_textures[i].frame->mtl_textures[vision_textures[i].plane] == NULL) {
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

        for (size_t i = 0; i < vision_texture_count; i++) {
            id<MTLTexture> tex = vision_plane_texture_of(vision_textures[i].frame,
                                                         vision_textures[i].plane);
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

static ami_status vision_frame_from_cg_image(ami_device *device,
                                             CGImageRef cg_img,
                                             uint32_t max_dim,
                                             ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (device == NULL || cg_img == NULL || max_dim == 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const size_t width = CGImageGetWidth(cg_img);
    const size_t height = CGImageGetHeight(cg_img);
    if (width == 0 || height == 0 || width > max_dim || height > max_dim) {
        return AMI_ERR_VISION;
    }

    ami_vision_frame *frame = NULL;
    ami_status st = ami_vision_frame_create(device, AMI_VISION_FMT_BGRA8,
                                            (uint32_t)width, (uint32_t)height, &frame);
    if (st != AMI_OK) {
        return st;
    }

    void *base = NULL;
    size_t bpr = 0;
    uint32_t locked_h = 0;
    st = ami_vision_frame_lock_plane(frame, 0, 0, &base, &bpr, &locked_h);
    if (st != AMI_OK) {
        ami_vision_frame_destroy(frame);
        return st;
    }

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (cs == NULL) {
        ami_vision_frame_unlock(frame, 0);
        ami_vision_frame_destroy(frame);
        return AMI_ERR_VISION;
    }
    // Little-endian 32-bit PremultipliedFirst yields [B, G, R, A] bytes matching BGRA8Unorm.
    const CGBitmapInfo bmp_info =
        (CGBitmapInfo)kCGBitmapByteOrder32Little | (CGBitmapInfo)kCGImageAlphaPremultipliedFirst;
    CGContextRef ctx = CGBitmapContextCreate(base, width, height, 8, bpr, cs, bmp_info);
    CGColorSpaceRelease(cs);
    if (ctx == NULL) {
        ami_vision_frame_unlock(frame, 0);
        ami_vision_frame_destroy(frame);
        return AMI_ERR_VISION;
    }
    CGContextSetBlendMode(ctx, kCGBlendModeCopy);
    CGContextDrawImage(ctx, CGRectMake(0.0, 0.0, (CGFloat)width, (CGFloat)height), cg_img);
    CGContextRelease(ctx);

    st = ami_vision_frame_unlock(frame, 0);
    if (st != AMI_OK) {
        ami_vision_frame_destroy(frame);
        return st;
    }
    *out_frame = frame;
    return AMI_OK;
}

static ami_status cg_image_from_vision_frame(ami_vision_frame *frame,
                                             bool opaque_alpha,
                                             CGImageRef *out_img) {
    if (out_img == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_img = NULL;
    if (frame == NULL || frame->format != AMI_VISION_FMT_BGRA8) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    void *base = NULL;
    size_t bpr = 0;
    uint32_t locked_h = 0;
    ami_status st = ami_vision_frame_lock_plane(frame, 0, 1u, &base, &bpr, &locked_h);
    if (st != AMI_OK) {
        return st;
    }

    CGColorSpaceRef cs = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
    if (cs == NULL) {
        ami_vision_frame_unlock(frame, 1u);
        return AMI_ERR_VISION;
    }
    const CGBitmapInfo alpha_info = opaque_alpha
        ? (CGBitmapInfo)kCGImageAlphaNoneSkipFirst
        : (CGBitmapInfo)kCGImageAlphaPremultipliedFirst;
    const CGBitmapInfo bmp_info = (CGBitmapInfo)kCGBitmapByteOrder32Little | alpha_info;
    CGContextRef ctx = CGBitmapContextCreate(base, frame->width, frame->height, 8, bpr, cs, bmp_info);
    CGColorSpaceRelease(cs);
    if (ctx == NULL) {
        ami_vision_frame_unlock(frame, 1u);
        return AMI_ERR_VISION;
    }
    CGImageRef cg_img = CGBitmapContextCreateImage(ctx);
    CGContextRelease(ctx);
    ami_vision_frame_unlock(frame, 1u);
    if (cg_img == NULL) {
        return AMI_ERR_VISION;
    }
    *out_img = cg_img;
    return AMI_OK;
}

ami_status ami_image_load_iosurface(ami_device *device, const char *path,
                                    ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (device == NULL || path == NULL || path[0] == '\0') {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        if (ns_path == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        NSURL *url = [NSURL fileURLWithPath:ns_path];
        CGImageSourceRef src = CGImageSourceCreateWithURL((__bridge CFURLRef)url, NULL);
        if (src == NULL) {
            return AMI_ERR_VISION;
        }
        CGImageRef cg_img = CGImageSourceCreateImageAtIndex(src, 0, NULL);
        CFRelease(src);
        if (cg_img == NULL) {
            return AMI_ERR_VISION;
        }

        ami_status st = vision_frame_from_cg_image(device, cg_img, AMI_MAX_TEXTURE_DIM, out_frame);
        CGImageRelease(cg_img);
        return st;
    }
}

ami_status ami_image_write_png(ami_vision_frame *frame, const char *path) {
    if (frame == NULL || frame->format != AMI_VISION_FMT_BGRA8 ||
        path == NULL || path[0] == '\0') {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSString *ns_path = [NSString stringWithUTF8String:path];
        if (ns_path == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        NSURL *url = [NSURL fileURLWithPath:ns_path];

        CGImageRef cg_img = NULL;
        ami_status st = cg_image_from_vision_frame(frame, false, &cg_img);
        if (st != AMI_OK) {
            return st;
        }

        CGImageDestinationRef dst = CGImageDestinationCreateWithURL(
            (__bridge CFURLRef)url, CFSTR("public.png"), 1, NULL);
        if (dst == NULL) {
            CGImageRelease(cg_img);
            return AMI_ERR_VISION;
        }
        CGImageDestinationAddImage(dst, cg_img, NULL);
        bool ok = CGImageDestinationFinalize(dst);
        CFRelease(dst);
        CGImageRelease(cg_img);
        return ok ? AMI_OK : AMI_ERR_VISION;
    }
}

ami_status ami_image_decode_bytes_iosurface(ami_device *device,
                                            const uint8_t *bytes, size_t len,
                                            ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (device == NULL || bytes == NULL || len == 0) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        NSData *data = [NSData dataWithBytesNoCopy:(void *)(uintptr_t)bytes
                                            length:(NSUInteger)len
                                      freeWhenDone:NO];
        if (data == nil) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        NSDictionary *src_opts = @{
            (__bridge id)kCGImageSourceShouldCache: @NO,
        };
        CGImageSourceRef src = CGImageSourceCreateWithData(
            (__bridge CFDataRef)data, (__bridge CFDictionaryRef)src_opts);
        if (src == NULL) {
            return AMI_ERR_VISION;
        }
        if (CGImageSourceGetCount(src) == 0) {
            CFRelease(src);
            return AMI_ERR_VISION;
        }
        NSDictionary *img_opts = @{
            (__bridge id)kCGImageSourceShouldCache: @NO,
            (__bridge id)kCGImageSourceShouldCacheImmediately: @YES,
        };
        CGImageRef cg_img = CGImageSourceCreateImageAtIndex(
            src, 0, (__bridge CFDictionaryRef)img_opts);
        CFRelease(src);
        if (cg_img == NULL) {
            return AMI_ERR_VISION;
        }

        ami_status st = vision_frame_from_cg_image(
            device, cg_img, AMI_VISION_MAX_BYTES_IMAGE_DIM, out_frame);
        CGImageRelease(cg_img);
        return st;
    }
}

void ami_vision_free_bytes(void *bytes) {
    free(bytes);
}

ami_status ami_image_encode_uti_alloc(ami_vision_frame *const *frames,
                                      uint32_t frame_count, const char *uti,
                                      uint8_t **out_bytes, size_t *out_len) {
    if (out_bytes != NULL) {
        *out_bytes = NULL;
    }
    if (out_len != NULL) {
        *out_len = 0;
    }
    if (frames == NULL || frame_count == 0 || frame_count > 128u ||
        uti == NULL || uti[0] == '\0' || out_bytes == NULL || out_len == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    for (uint32_t i = 0; i < frame_count; i++) {
        if (frames[i] == NULL || frames[i]->format != AMI_VISION_FMT_BGRA8) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }

    const bool is_jpeg = (strcmp(uti, "public.jpeg") == 0);
    const bool is_bmp = (strcmp(uti, "com.microsoft.bmp") == 0);
    const bool is_gif = (strcmp(uti, "com.compuserve.gif") == 0);
    const bool opaque_alpha = is_jpeg || is_bmp;

    @autoreleasepool {
        NSString *ns_uti = [NSString stringWithUTF8String:uti];
        if (ns_uti == nil) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        CFMutableDataRef cf_out = CFDataCreateMutable(kCFAllocatorDefault, 0);
        if (cf_out == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        CGImageDestinationRef dst = CGImageDestinationCreateWithData(
            cf_out, (__bridge CFStringRef)ns_uti, (size_t)frame_count, NULL);
        if (dst == NULL) {
            CFRelease(cf_out);
            return AMI_ERR_VISION;
        }

        if (is_gif && frame_count > 1u) {
            NSDictionary *gif_props = @{
                (__bridge id)kCGImagePropertyGIFDictionary: @{
                    (__bridge id)kCGImagePropertyGIFLoopCount: @0,
                },
            };
            CGImageDestinationSetProperties(dst, (__bridge CFDictionaryRef)gif_props);
        }

        NSDictionary *frame_props = (is_gif && frame_count > 1u)
            ? @{
                (__bridge id)kCGImagePropertyGIFDictionary: @{
                    (__bridge id)kCGImagePropertyGIFDelayTime: @(0.05),
                },
              }
            : nil;

        for (uint32_t i = 0; i < frame_count; i++) {
            CGImageRef cg_img = NULL;
            ami_status st = cg_image_from_vision_frame(frames[i], opaque_alpha, &cg_img);
            if (st != AMI_OK) {
                CFRelease(dst);
                CFRelease(cf_out);
                return st;
            }
            CGImageDestinationAddImage(dst, cg_img, (__bridge CFDictionaryRef)frame_props);
            CGImageRelease(cg_img);
        }

        bool ok = CGImageDestinationFinalize(dst);
        CFRelease(dst);
        if (!ok) {
            CFRelease(cf_out);
            return AMI_ERR_VISION;
        }

        const CFIndex cf_len = CFDataGetLength(cf_out);
        if (cf_len <= 0) {
            CFRelease(cf_out);
            return AMI_ERR_VISION;
        }
        uint8_t *buf = malloc((size_t)cf_len);
        if (buf == NULL) {
            CFRelease(cf_out);
            return AMI_ERR_OUT_OF_MEMORY;
        }
        memcpy(buf, CFDataGetBytePtr(cf_out), (size_t)cf_len);
        CFRelease(cf_out);
        *out_bytes = buf;
        *out_len = (size_t)cf_len;
        return AMI_OK;
    }
}

ami_status ami_image_encode_png_alloc(ami_vision_frame *frame,
                                      uint8_t **out_bytes, size_t *out_len) {
    if (frame == NULL) {
        if (out_bytes != NULL) {
            *out_bytes = NULL;
        }
        if (out_len != NULL) {
            *out_len = 0;
        }
        return AMI_ERR_INVALID_ARGUMENT;
    }
    return ami_image_encode_uti_alloc(&frame, 1u, "public.png", out_bytes, out_len);
}

typedef struct ami_vt_encode_collector {
    CMSampleBufferRef *samples;
    uint32_t capacity;
    uint32_t count;
    OSStatus first_error;
} ami_vt_encode_collector;

static void vt_compress_output_callback(void *outputCallbackRefCon,
                                        void *sourceFrameRefCon,
                                        OSStatus status,
                                        VTEncodeInfoFlags infoFlags,
                                        CMSampleBufferRef sampleBuffer) {
    (void)sourceFrameRefCon;
    (void)infoFlags;
    ami_vt_encode_collector *col = (ami_vt_encode_collector *)outputCallbackRefCon;
    if (col == NULL) {
        return;
    }
    if (status != noErr) {
        if (col->first_error == noErr) {
            col->first_error = status;
        }
        return;
    }
    if (sampleBuffer == NULL) {
        return;
    }
    if (col->count >= col->capacity) {
        uint32_t new_cap = (col->capacity == 0) ? 8u : col->capacity * 2u;
        CMSampleBufferRef *grown =
            realloc(col->samples, (size_t)new_cap * sizeof(CMSampleBufferRef));
        if (grown == NULL) {
            if (col->first_error == noErr) {
                col->first_error = kVTAllocationFailedErr;
            }
            return;
        }
        col->samples = grown;
        col->capacity = new_cap;
    }
    CFRetain(sampleBuffer);
    col->samples[col->count++] = sampleBuffer;
}

static ami_status vt_encode_pixel_buffers(uint32_t codec,
                                          CVPixelBufferRef const *pbs,
                                          uint32_t width, uint32_t height,
                                          uint32_t frame_count, uint32_t fps,
                                          ami_vt_clip **out_clip) {
    if (out_clip == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_clip = NULL;
    if ((codec != AMI_VT_CODEC_H264 && codec != AMI_VT_CODEC_HEVC) ||
        pbs == NULL || frame_count == 0 || fps == 0 || fps > 240u ||
        width < 16u || height < 16u || (width & 1u) != 0 || (height & 1u) != 0 ||
        width > AMI_MAX_TEXTURE_DIM || height > AMI_MAX_TEXTURE_DIM) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const CMVideoCodecType cm_codec =
        (codec == AMI_VT_CODEC_HEVC) ? kCMVideoCodecType_HEVC : kCMVideoCodecType_H264;

    ami_vt_encode_collector col = { 0 };
    col.samples = calloc(frame_count, sizeof(CMSampleBufferRef));
    if (col.samples == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }
    col.capacity = frame_count;

    uint32_t using_hw = 0;

    @autoreleasepool {
        NSDictionary *enc_spec = @{
            (__bridge id)kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder: @YES,
        };
        VTCompressionSessionRef session = NULL;
        OSStatus os_st = VTCompressionSessionCreate(
            kCFAllocatorDefault,
            (int32_t)width,
            (int32_t)height,
            cm_codec,
            (__bridge CFDictionaryRef)enc_spec,
            NULL,
            kCFAllocatorDefault,
            vt_compress_output_callback,
            &col,
            &session);
        if (os_st != noErr || session == NULL) {
            free(col.samples);
            return AMI_ERR_VISION;
        }

        VTSessionSetProperty(session, kVTCompressionPropertyKey_RealTime, kCFBooleanFalse);
        VTSessionSetProperty(session, kVTCompressionPropertyKey_AllowFrameReordering, kCFBooleanFalse);
        NSNumber *fps_num = @(fps);
        VTSessionSetProperty(session, kVTCompressionPropertyKey_ExpectedFrameRate,
                             (__bridge CFTypeRef)fps_num);
        NSNumber *bitrate_num = @(4000000);
        VTSessionSetProperty(session, kVTCompressionPropertyKey_AverageBitRate,
                             (__bridge CFTypeRef)bitrate_num);

        os_st = VTCompressionSessionPrepareToEncodeFrames(session);
        if (os_st != noErr) {
            VTCompressionSessionInvalidate(session);
            CFRelease(session);
            free(col.samples);
            return AMI_ERR_VISION;
        }

        CFBooleanRef hw_val = NULL;
        if (VTSessionCopyProperty(session,
                                  kVTCompressionPropertyKey_UsingHardwareAcceleratedVideoEncoder,
                                  kCFAllocatorDefault, &hw_val) == noErr &&
            hw_val != NULL) {
            using_hw = CFBooleanGetValue(hw_val) ? 1u : 0u;
            CFRelease(hw_val);
        }

        for (uint32_t i = 0; i < frame_count; i++) {
            CMTime pts = CMTimeMake((int64_t)i, (int32_t)fps);
            CMTime dur = CMTimeMake(1, (int32_t)fps);
            os_st = VTCompressionSessionEncodeFrame(session, pbs[i], pts, dur, NULL, NULL, NULL);
            if (os_st != noErr) {
                break;
            }
        }
        if (os_st == noErr) {
            os_st = VTCompressionSessionCompleteFrames(session, kCMTimeInvalid);
        }
        VTCompressionSessionInvalidate(session);
        CFRelease(session);

        if (os_st != noErr || col.first_error != noErr || col.count != frame_count) {
            for (uint32_t i = 0; i < col.count; i++) {
                CFRelease(col.samples[i]);
            }
            free(col.samples);
            return AMI_ERR_VISION;
        }
    }

    CMFormatDescriptionRef fmt = CMSampleBufferGetFormatDescription(col.samples[0]);
    if (fmt == NULL) {
        for (uint32_t i = 0; i < col.count; i++) {
            CFRelease(col.samples[i]);
        }
        free(col.samples);
        return AMI_ERR_VISION;
    }

    uint64_t total_bytes = 0;
    for (uint32_t i = 0; i < col.count; i++) {
        CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(col.samples[i]);
        if (bb != NULL) {
            total_bytes += (uint64_t)CMBlockBufferGetDataLength(bb);
        }
    }
    if (total_bytes == 0) {
        for (uint32_t i = 0; i < col.count; i++) {
            CFRelease(col.samples[i]);
        }
        free(col.samples);
        return AMI_ERR_VISION;
    }

    ami_vt_clip *clip = calloc(1, sizeof *clip);
    if (clip == NULL) {
        for (uint32_t i = 0; i < col.count; i++) {
            CFRelease(col.samples[i]);
        }
        free(col.samples);
        return AMI_ERR_OUT_OF_MEMORY;
    }
    CFRetain(fmt);
    clip->format_desc = fmt;
    clip->samples = col.samples;
    clip->codec = codec;
    clip->width = width;
    clip->height = height;
    clip->frame_count = col.count;
    clip->fps = fps;
    clip->using_hardware_encoder = using_hw;
    clip->total_compressed_bytes = total_bytes;
    *out_clip = clip;
    return AMI_OK;
}

ami_status ami_vt_encode_test_clip(uint32_t codec, uint32_t width, uint32_t height,
                                   uint32_t frame_count, uint32_t fps,
                                   ami_vt_clip **out_clip) {
    if (out_clip == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_clip = NULL;
    if ((codec != AMI_VT_CODEC_H264 && codec != AMI_VT_CODEC_HEVC) ||
        frame_count == 0 || frame_count > 128u || fps == 0 || fps > 240u ||
        width < 16u || height < 16u || (width & 1u) != 0 || (height & 1u) != 0 ||
        width > AMI_MAX_TEXTURE_DIM || height > AMI_MAX_TEXTURE_DIM) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    CVPixelBufferRef *pbs = calloc(frame_count, sizeof(CVPixelBufferRef));
    if (pbs == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }

    ami_status st = AMI_OK;
    for (uint32_t f = 0; f < frame_count; f++) {
        uint64_t alloc_bytes = 0;
        st = create_iosurface_pixel_buffer(AMI_VISION_FMT_NV12, width, height,
                                           &pbs[f], &alloc_bytes);
        if (st != AMI_OK) {
            break;
        }
        if (CVPixelBufferLockBaseAddress(pbs[f], 0) != kCVReturnSuccess) {
            st = AMI_ERR_VISION;
            break;
        }
        uint8_t *y_plane = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pbs[f], 0);
        size_t y_bpr = CVPixelBufferGetBytesPerRowOfPlane(pbs[f], 0);
        uint8_t *uv_plane = (uint8_t *)CVPixelBufferGetBaseAddressOfPlane(pbs[f], 1);
        size_t uv_bpr = CVPixelBufferGetBytesPerRowOfPlane(pbs[f], 1);
        if (y_plane == NULL || uv_plane == NULL) {
            CVPixelBufferUnlockBaseAddress(pbs[f], 0);
            st = AMI_ERR_VISION;
            break;
        }

        // Deterministic spatiotemporal pattern (moving blocks + smooth gradients).
        for (uint32_t y = 0; y < height; y++) {
            uint8_t *row = y_plane + (size_t)y * y_bpr;
            for (uint32_t x = 0; x < width; x++) {
                uint32_t block = (((x + f * 8u) / 16u) ^ ((y + f * 4u) / 16u)) & 1u;
                uint32_t grad = ((x * 160u) / width) + ((y * 64u) / height);
                uint32_t luma = 24u + (grad + block * 28u + f * 11u) % 208u;
                row[x] = (uint8_t)luma;
            }
        }
        const uint32_t cw = width / 2u;
        const uint32_t ch = height / 2u;
        for (uint32_t y = 0; y < ch; y++) {
            uint8_t *row = uv_plane + (size_t)y * uv_bpr;
            for (uint32_t x = 0; x < cw; x++) {
                uint8_t cb = (uint8_t)(64u + ((x * 128u) / cw + f * 13u) % 128u);
                uint8_t cr = (uint8_t)(64u + ((y * 128u) / ch + f * 17u) % 128u);
                row[x * 2u + 0u] = cb;
                row[x * 2u + 1u] = cr;
            }
        }
        CVPixelBufferUnlockBaseAddress(pbs[f], 0);
    }

    if (st == AMI_OK) {
        st = vt_encode_pixel_buffers(codec, pbs, width, height, frame_count, fps, out_clip);
    }

    for (uint32_t f = 0; f < frame_count; f++) {
        if (pbs[f] != NULL) {
            CVPixelBufferRelease(pbs[f]);
        }
    }
    free(pbs);
    return st;
}

ami_status ami_vt_encode_frames(uint32_t codec, ami_vision_frame *const *frames,
                                uint32_t frame_count, uint32_t fps,
                                ami_vt_clip **out_clip) {
    if (out_clip == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_clip = NULL;
    if (frames == NULL || frame_count == 0 || frame_count > 128u) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (frames[0] == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    const uint32_t width = frames[0]->width;
    const uint32_t height = frames[0]->height;
    for (uint32_t i = 0; i < frame_count; i++) {
        if (frames[i] == NULL || frames[i]->pixel_buffer == NULL ||
            frames[i]->width != width || frames[i]->height != height ||
            (frames[i]->format != AMI_VISION_FMT_BGRA8 &&
             frames[i]->format != AMI_VISION_FMT_NV12)) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
    }

    CVPixelBufferRef *pbs = calloc(frame_count, sizeof(CVPixelBufferRef));
    if (pbs == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < frame_count; i++) {
        pbs[i] = frames[i]->pixel_buffer;
    }
    ami_status st = vt_encode_pixel_buffers(codec, pbs, width, height, frame_count, fps, out_clip);
    free(pbs);
    return st;
}

void ami_vt_clip_destroy(ami_vt_clip *clip) {
    if (clip == NULL) {
        return;
    }
    if (clip->samples != NULL) {
        for (uint32_t i = 0; i < clip->frame_count; i++) {
            if (clip->samples[i] != NULL) {
                CFRelease(clip->samples[i]);
            }
        }
        free(clip->samples);
    }
    if (clip->format_desc != NULL) {
        CFRelease(clip->format_desc);
    }
    free(clip);
}

ami_status ami_vt_clip_query(const ami_vt_clip *clip, ami_vt_clip_info *out_info) {
    if (clip == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_info, 0, sizeof *out_info);
    out_info->codec = clip->codec;
    out_info->width = clip->width;
    out_info->height = clip->height;
    out_info->frame_count = clip->frame_count;
    out_info->fps = clip->fps;
    out_info->using_hardware_encoder = clip->using_hardware_encoder;
    out_info->total_compressed_bytes = clip->total_compressed_bytes;
    return AMI_OK;
}

typedef struct ami_vt_decode_output {
    CVPixelBufferRef pixel_buffer;
    CMTime pts;
    OSStatus status;
    bool received;
} ami_vt_decode_output;

static void vt_decompress_output_callback(void *decompressionOutputRefCon,
                                          void *sourceFrameRefCon,
                                          OSStatus status,
                                          VTDecodeInfoFlags infoFlags,
                                          CVImageBufferRef imageBuffer,
                                          CMTime presentationTimeStamp,
                                          CMTime presentationDuration) {
    (void)decompressionOutputRefCon;
    (void)infoFlags;
    (void)presentationDuration;
    ami_vt_decode_output *out = (ami_vt_decode_output *)sourceFrameRefCon;
    if (out == NULL) {
        return;
    }
    out->status = status;
    out->pts = presentationTimeStamp;
    out->received = true;
    if (status == noErr && imageBuffer != NULL) {
        if (out->pixel_buffer != NULL) {
            CVPixelBufferRelease(out->pixel_buffer);
        }
        CVPixelBufferRetain(imageBuffer);
        out->pixel_buffer = imageBuffer;
    }
}

ami_status ami_vt_decode_session_create(ami_device *device, const ami_vt_clip *clip,
                                        uint32_t output_format,
                                        ami_vt_decode_session **out_session) {
    if (out_session == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_session = NULL;
    if (device == NULL || clip == NULL || clip->format_desc == NULL ||
        (output_format != AMI_VISION_FMT_NV12 && output_format != AMI_VISION_FMT_BGRA8)) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const OSType cv_pix_fmt =
        (output_format == AMI_VISION_FMT_BGRA8)
            ? kCVPixelFormatType_32BGRA
            : kCVPixelFormatType_420YpCbCr8BiPlanarFullRange;

    @autoreleasepool {
        NSDictionary *dec_spec = @{
            (__bridge id)kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: @YES,
        };
        NSDictionary *dst_attrs = @{
            (__bridge id)kCVPixelBufferPixelFormatTypeKey: @(cv_pix_fmt),
            (__bridge id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (__bridge id)kCVPixelBufferMetalCompatibilityKey: @YES,
            (__bridge id)kCVPixelBufferBytesPerRowAlignmentKey: @64,
        };
        VTDecompressionOutputCallbackRecord cb = {
            .decompressionOutputCallback = vt_decompress_output_callback,
            .decompressionOutputRefCon = NULL,
        };

        VTDecompressionSessionRef vt_session = NULL;
        OSStatus os_st = VTDecompressionSessionCreate(
            kCFAllocatorDefault,
            clip->format_desc,
            (__bridge CFDictionaryRef)dec_spec,
            (__bridge CFDictionaryRef)dst_attrs,
            &cb,
            &vt_session);
        if (os_st != noErr || vt_session == NULL) {
            return AMI_ERR_VISION;
        }

        uint32_t using_hw = 0;
        CFBooleanRef hw_val = NULL;
        if (VTSessionCopyProperty(vt_session,
                                  kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                                  kCFAllocatorDefault, &hw_val) == noErr &&
            hw_val != NULL) {
            using_hw = CFBooleanGetValue(hw_val) ? 1u : 0u;
            CFRelease(hw_val);
        }

        ami_vt_decode_session *session = calloc(1, sizeof *session);
        if (session == NULL) {
            VTDecompressionSessionInvalidate(vt_session);
            CFRelease(vt_session);
            return AMI_ERR_OUT_OF_MEMORY;
        }
        id<MTLDevice> mtl_dev = device_of(device);
        session->mtl_device = (__bridge_retained void *)mtl_dev;
        session->vt_session = vt_session;
        session->codec = clip->codec;
        session->output_format = output_format;
        session->width = clip->width;
        session->height = clip->height;
        session->using_hardware_decoder = using_hw;
        *out_session = session;
    }
    return AMI_OK;
}

void ami_vt_decode_session_destroy(ami_vt_decode_session *session) {
    if (session == NULL) {
        return;
    }
    if (session->vt_session != NULL) {
        VTDecompressionSessionWaitForAsynchronousFrames(session->vt_session);
        VTDecompressionSessionInvalidate(session->vt_session);
        CFRelease(session->vt_session);
        session->vt_session = NULL;
    }
    if (session->mtl_device != NULL) {
        CFRelease(session->mtl_device);
        session->mtl_device = NULL;
    }
    free(session);
}

ami_status ami_vt_decode_session_query(const ami_vt_decode_session *session,
                                       ami_vt_decode_session_info *out_info) {
    if (session == NULL || out_info == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_info, 0, sizeof *out_info);
    out_info->codec = session->codec;
    out_info->output_format = session->output_format;
    out_info->width = session->width;
    out_info->height = session->height;
    out_info->using_hardware_decoder = session->using_hardware_decoder;
    return AMI_OK;
}

ami_status ami_vt_decode_clip_frame(ami_vt_decode_session *session,
                                    const ami_vt_clip *clip, uint32_t frame_index,
                                    ami_vision_frame **out_frame) {
    if (out_frame == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame = NULL;
    if (session == NULL || session->vt_session == NULL || clip == NULL ||
        frame_index >= clip->frame_count || clip->samples[frame_index] == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    ami_vt_decode_output out = { 0 };
    out.pts = kCMTimeZero;

    OSStatus os_st = VTDecompressionSessionDecodeFrame(
        session->vt_session,
        clip->samples[frame_index],
        0,
        &out,
        NULL);
    if (os_st != noErr) {
        if (out.pixel_buffer != NULL) {
            CVPixelBufferRelease(out.pixel_buffer);
        }
        return AMI_ERR_VISION;
    }

    os_st = VTDecompressionSessionWaitForAsynchronousFrames(session->vt_session);
    if (os_st != noErr || !out.received || out.status != noErr || out.pixel_buffer == NULL) {
        if (out.pixel_buffer != NULL) {
            CVPixelBufferRelease(out.pixel_buffer);
        }
        return AMI_ERR_VISION;
    }

    id<MTLDevice> mtl_dev = (__bridge id<MTLDevice>)session->mtl_device;
    ami_status st = vision_frame_init_from_mtl_device(
        mtl_dev,
        out.pixel_buffer,
        0u,
        1u,
        out.pts.value,
        out.pts.timescale,
        out_frame);
    CVPixelBufferRelease(out.pixel_buffer);
    return st;
}

// ----------------------------------------------------------------------------
// M9.6 C29 Item 7: Pure In-Memory ISO-BMFF Muxer/Demuxer, Annex-B, & ImageIO
// ----------------------------------------------------------------------------

#define AMI_ISO_FOURCC(a, b, c, d) \
    (((uint32_t)(uint8_t)(a) << 24) | ((uint32_t)(uint8_t)(b) << 16) | \
     ((uint32_t)(uint8_t)(c) << 8)  | (uint32_t)(uint8_t)(d))

static inline uint16_t ami_read_be16(const uint8_t *p) {
    return (uint16_t)(((uint16_t)p[0] << 8) | (uint16_t)p[1]);
}

static inline uint32_t ami_read_be32(const uint8_t *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

static inline uint64_t ami_read_be64(const uint8_t *p) {
    return ((uint64_t)ami_read_be32(p) << 32) | (uint64_t)ami_read_be32(p + 4);
}

static inline void ami_write_be32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >> 8);
    p[3] = (uint8_t)(v & 0xFFu);
}

typedef struct ami_byte_vec {
    uint8_t *data;
    size_t len;
    size_t cap;
} ami_byte_vec;

static bool bv_reserve(ami_byte_vec *v, size_t extra) {
    if (extra > SIZE_MAX - v->len) {
        return false;
    }
    const size_t needed = v->len + extra;
    if (needed <= v->cap) {
        return true;
    }
    size_t new_cap = (v->cap == 0) ? 4096u : v->cap;
    while (new_cap < needed) {
        if (new_cap > SIZE_MAX / 2u) {
            new_cap = needed;
            break;
        }
        new_cap *= 2u;
    }
    uint8_t *grown = realloc(v->data, new_cap);
    if (grown == NULL) {
        return false;
    }
    v->data = grown;
    v->cap = new_cap;
    return true;
}

static bool bv_append_bytes(ami_byte_vec *v, const void *src, size_t n) {
    if (n == 0) {
        return true;
    }
    if (!bv_reserve(v, n)) {
        return false;
    }
    memcpy(v->data + v->len, src, n);
    v->len += n;
    return true;
}

static bool bv_append_zeros(ami_byte_vec *v, size_t n) {
    if (n == 0) {
        return true;
    }
    if (!bv_reserve(v, n)) {
        return false;
    }
    memset(v->data + v->len, 0, n);
    v->len += n;
    return true;
}

static bool bv_append_u8(ami_byte_vec *v, uint8_t b) {
    return bv_append_bytes(v, &b, 1u);
}

static bool bv_append_be16(ami_byte_vec *v, uint16_t val) {
    const uint8_t buf[2] = {
        (uint8_t)(val >> 8),
        (uint8_t)(val & 0xFFu),
    };
    return bv_append_bytes(v, buf, sizeof(buf));
}

static bool bv_append_be32(ami_byte_vec *v, uint32_t val) {
    uint8_t buf[4];
    ami_write_be32(buf, val);
    return bv_append_bytes(v, buf, sizeof(buf));
}

static bool bv_append_fourcc(ami_byte_vec *v, const char tag[4]) {
    return bv_append_bytes(v, tag, 4u);
}

static bool bv_begin_box(ami_byte_vec *v, const char tag[4], size_t *out_pos) {
    *out_pos = v->len;
    return bv_append_be32(v, 0u) && bv_append_fourcc(v, tag);
}

static bool bv_begin_fullbox(ami_byte_vec *v, const char tag[4],
                             uint8_t version, uint32_t flags, size_t *out_pos) {
    const uint32_t vf = ((uint32_t)version << 24) | (flags & 0x00FFFFFFu);
    return bv_begin_box(v, tag, out_pos) && bv_append_be32(v, vf);
}

static bool bv_end_box(ami_byte_vec *v, size_t pos) {
    const size_t sz = v->len - pos;
    if (sz > UINT32_MAX) {
        return false;
    }
    ami_write_be32(v->data + pos, (uint32_t)sz);
    return true;
}

static bool bv_append_unity_matrix(ami_byte_vec *v) {
    const uint32_t m[9] = {
        0x00010000u, 0u, 0u,
        0u, 0x00010000u, 0u,
        0u, 0u, 0x40000000u,
    };
    for (size_t i = 0; i < 9u; i++) {
        if (!bv_append_be32(v, m[i])) {
            return false;
        }
    }
    return true;
}

static ami_status extract_clip_codec_config(const ami_vt_clip *clip, ami_byte_vec *out_cfg) {
    if (clip == NULL || clip->format_desc == NULL || out_cfg == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    @autoreleasepool {
        CFDictionaryRef exts = CMFormatDescriptionGetExtensions(clip->format_desc);
        if (exts != NULL) {
            CFDictionaryRef atoms = (CFDictionaryRef)CFDictionaryGetValue(
                exts, kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms);
            if (atoms != NULL) {
                CFStringRef key = (clip->codec == AMI_VT_CODEC_HEVC) ? CFSTR("hvcC") : CFSTR("avcC");
                CFDataRef cfg = (CFDataRef)CFDictionaryGetValue(atoms, key);
                if (cfg != NULL && CFDataGetLength(cfg) > 0) {
                    if (!bv_append_bytes(out_cfg, CFDataGetBytePtr(cfg),
                                         (size_t)CFDataGetLength(cfg))) {
                        return AMI_ERR_OUT_OF_MEMORY;
                    }
                    return AMI_OK;
                }
            }
        }
    }

    // Fallback for H.264 if SampleDescriptionExtensionAtoms did not populate avcC.
    if (clip->codec == AMI_VT_CODEC_H264) {
        size_t param_count = 0;
        int nal_len_hdr = 0;
        if (CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                clip->format_desc, 0, NULL, NULL, &param_count, &nal_len_hdr) == noErr &&
            param_count >= 2u) {
            const uint8_t *sps = NULL;
            size_t sps_sz = 0;
            const uint8_t *pps = NULL;
            size_t pps_sz = 0;
            if (CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                    clip->format_desc, 0, &sps, &sps_sz, NULL, NULL) == noErr &&
                CMVideoFormatDescriptionGetH264ParameterSetAtIndex(
                    clip->format_desc, 1, &pps, &pps_sz, NULL, NULL) == noErr &&
                sps != NULL && sps_sz >= 4u && sps_sz <= UINT16_MAX &&
                pps != NULL && pps_sz >= 1u && pps_sz <= UINT16_MAX) {
                const uint8_t len_m1 = (nal_len_hdr == 1 || nal_len_hdr == 2 || nal_len_hdr == 4)
                    ? (uint8_t)(nal_len_hdr - 1) : 3u;
                bool ok = bv_append_u8(out_cfg, 1u) &&
                          bv_append_u8(out_cfg, sps[1]) &&
                          bv_append_u8(out_cfg, sps[2]) &&
                          bv_append_u8(out_cfg, sps[3]) &&
                          bv_append_u8(out_cfg, (uint8_t)(0xFCu | len_m1)) &&
                          bv_append_u8(out_cfg, 0xE1u) &&
                          bv_append_be16(out_cfg, (uint16_t)sps_sz) &&
                          bv_append_bytes(out_cfg, sps, sps_sz) &&
                          bv_append_u8(out_cfg, 1u) &&
                          bv_append_be16(out_cfg, (uint16_t)pps_sz) &&
                          bv_append_bytes(out_cfg, pps, pps_sz);
                return ok ? AMI_OK : AMI_ERR_OUT_OF_MEMORY;
            }
        }
    }
    return AMI_ERR_VISION;
}

ami_status ami_vt_clip_export_mp4_bytes(const ami_vt_clip *clip,
                                        uint8_t **out_bytes, size_t *out_len) {
    if (out_bytes != NULL) {
        *out_bytes = NULL;
    }
    if (out_len != NULL) {
        *out_len = 0;
    }
    if (clip == NULL || clip->format_desc == NULL || clip->samples == NULL ||
        clip->frame_count == 0 || clip->fps == 0 ||
        (clip->codec != AMI_VT_CODEC_H264 && clip->codec != AMI_VT_CODEC_HEVC) ||
        out_bytes == NULL || out_len == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    const uint32_t n = clip->frame_count;
    const uint32_t timescale = clip->fps * 1000u;
    const uint32_t sample_delta = 1000u;
    if ((uint64_t)n * (uint64_t)sample_delta > UINT32_MAX) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    const uint32_t total_duration = n * sample_delta;

    ami_byte_vec codec_cfg = { 0 };
    ami_status st = extract_clip_codec_config(clip, &codec_cfg);
    if (st != AMI_OK) {
        free(codec_cfg.data);
        return st;
    }

    uint32_t *sample_sizes = calloc(n, sizeof(uint32_t));
    uint32_t *sync_indices = calloc(n, sizeof(uint32_t));
    if (sample_sizes == NULL || sync_indices == NULL) {
        free(sample_sizes);
        free(sync_indices);
        free(codec_cfg.data);
        return AMI_ERR_OUT_OF_MEMORY;
    }

    uint32_t sync_count = 0;
    uint64_t mdat_payload_bytes = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (clip->samples[i] == NULL) {
            free(sample_sizes);
            free(sync_indices);
            free(codec_cfg.data);
            return AMI_ERR_VISION;
        }
        CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(clip->samples[i]);
        if (bb == NULL) {
            free(sample_sizes);
            free(sync_indices);
            free(codec_cfg.data);
            return AMI_ERR_VISION;
        }
        const size_t sz = CMBlockBufferGetDataLength(bb);
        if (sz == 0 || sz > UINT32_MAX || mdat_payload_bytes > UINT32_MAX - sz) {
            free(sample_sizes);
            free(sync_indices);
            free(codec_cfg.data);
            return AMI_ERR_VISION;
        }
        sample_sizes[i] = (uint32_t)sz;
        mdat_payload_bytes += sz;

        bool is_sync = true;
        CFArrayRef atts = CMSampleBufferGetSampleAttachmentsArray(clip->samples[i], false);
        if (atts != NULL && CFArrayGetCount(atts) > 0) {
            CFDictionaryRef dict = (CFDictionaryRef)CFArrayGetValueAtIndex(atts, 0);
            if (dict != NULL) {
                CFBooleanRef not_sync = NULL;
                if (CFDictionaryGetValueIfPresent(dict, kCMSampleAttachmentKey_NotSync,
                                                  (const void **)&not_sync) &&
                    not_sync != NULL && CFBooleanGetValue(not_sync)) {
                    is_sync = false;
                }
            }
        }
        if (is_sync) {
            sync_indices[sync_count++] = i + 1u;
        }
    }

    ami_byte_vec out = { 0 };
    bool ok = true;
    size_t ftyp_pos = 0, moov_pos = 0, mvhd_pos = 0, trak_pos = 0, tkhd_pos = 0;
    size_t mdia_pos = 0, mdhd_pos = 0, hdlr_pos = 0, minf_pos = 0, vmhd_pos = 0;
    size_t dinf_pos = 0, dref_pos = 0, url_pos = 0, stbl_pos = 0, stsd_pos = 0;
    size_t entry_pos = 0, cfg_box_pos = 0, stts_pos = 0, stss_pos = 0;
    size_t stsc_pos = 0, stsz_pos = 0, stco_pos = 0, mdat_pos = 0;
    size_t stco_offset_pos = 0;

    // 1. ftyp
    ok = ok && bv_begin_box(&out, "ftyp", &ftyp_pos);
    ok = ok && bv_append_fourcc(&out, "isom");
    ok = ok && bv_append_be32(&out, 512u);
    ok = ok && bv_append_fourcc(&out, "isom");
    ok = ok && bv_append_fourcc(&out, "iso2");
    ok = ok && bv_append_fourcc(&out, (clip->codec == AMI_VT_CODEC_HEVC) ? "hvc1" : "avc1");
    ok = ok && bv_append_fourcc(&out, "mp41");
    ok = ok && bv_end_box(&out, ftyp_pos);

    // 2. moov
    ok = ok && bv_begin_box(&out, "moov", &moov_pos);

    // mvhd (v0)
    ok = ok && bv_begin_fullbox(&out, "mvhd", 0u, 0u, &mvhd_pos);
    ok = ok && bv_append_be32(&out, 0u) && bv_append_be32(&out, 0u); // creation, mod
    ok = ok && bv_append_be32(&out, timescale);
    ok = ok && bv_append_be32(&out, total_duration);
    ok = ok && bv_append_be32(&out, 0x00010000u); // rate 1.0
    ok = ok && bv_append_be16(&out, 0x0100u);     // volume 1.0
    ok = ok && bv_append_zeros(&out, 10u);
    ok = ok && bv_append_unity_matrix(&out);
    ok = ok && bv_append_zeros(&out, 24u);        // pre_defined[6]
    ok = ok && bv_append_be32(&out, 2u);          // next_track_ID
    ok = ok && bv_end_box(&out, mvhd_pos);

    // trak
    ok = ok && bv_begin_box(&out, "trak", &trak_pos);

    // tkhd (v0, enabled | in_movie)
    ok = ok && bv_begin_fullbox(&out, "tkhd", 0u, 0x000003u, &tkhd_pos);
    ok = ok && bv_append_be32(&out, 0u) && bv_append_be32(&out, 0u);
    ok = ok && bv_append_be32(&out, 1u); // track_ID
    ok = ok && bv_append_be32(&out, 0u);
    ok = ok && bv_append_be32(&out, total_duration);
    ok = ok && bv_append_zeros(&out, 8u);
    ok = ok && bv_append_be16(&out, 0u) && bv_append_be16(&out, 0u);
    ok = ok && bv_append_be16(&out, 0u) && bv_append_be16(&out, 0u);
    ok = ok && bv_append_unity_matrix(&out);
    ok = ok && bv_append_be32(&out, clip->width << 16);
    ok = ok && bv_append_be32(&out, clip->height << 16);
    ok = ok && bv_end_box(&out, tkhd_pos);

    // mdia
    ok = ok && bv_begin_box(&out, "mdia", &mdia_pos);

    // mdhd (v0)
    ok = ok && bv_begin_fullbox(&out, "mdhd", 0u, 0u, &mdhd_pos);
    ok = ok && bv_append_be32(&out, 0u) && bv_append_be32(&out, 0u);
    ok = ok && bv_append_be32(&out, timescale);
    ok = ok && bv_append_be32(&out, total_duration);
    ok = ok && bv_append_be16(&out, 0x55C4u); // "und"
    ok = ok && bv_append_be16(&out, 0u);
    ok = ok && bv_end_box(&out, mdhd_pos);

    // hdlr (vide)
    ok = ok && bv_begin_fullbox(&out, "hdlr", 0u, 0u, &hdlr_pos);
    ok = ok && bv_append_be32(&out, 0u);
    ok = ok && bv_append_fourcc(&out, "vide");
    ok = ok && bv_append_zeros(&out, 12u);
    ok = ok && bv_append_bytes(&out, "VideoHandler", 13u);
    ok = ok && bv_end_box(&out, hdlr_pos);

    // minf
    ok = ok && bv_begin_box(&out, "minf", &minf_pos);

    // vmhd
    ok = ok && bv_begin_fullbox(&out, "vmhd", 0u, 1u, &vmhd_pos);
    ok = ok && bv_append_zeros(&out, 8u);
    ok = ok && bv_end_box(&out, vmhd_pos);

    // dinf -> dref -> url
    ok = ok && bv_begin_box(&out, "dinf", &dinf_pos);
    ok = ok && bv_begin_fullbox(&out, "dref", 0u, 0u, &dref_pos);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_begin_fullbox(&out, "url ", 0u, 1u, &url_pos);
    ok = ok && bv_end_box(&out, url_pos);
    ok = ok && bv_end_box(&out, dref_pos);
    ok = ok && bv_end_box(&out, dinf_pos);

    // stbl
    ok = ok && bv_begin_box(&out, "stbl", &stbl_pos);

    // stsd
    ok = ok && bv_begin_fullbox(&out, "stsd", 0u, 0u, &stsd_pos);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_begin_box(&out, (clip->codec == AMI_VT_CODEC_HEVC) ? "hvc1" : "avc1", &entry_pos);
    ok = ok && bv_append_zeros(&out, 6u);
    ok = ok && bv_append_be16(&out, 1u); // data_reference_index
    ok = ok && bv_append_zeros(&out, 16u);
    ok = ok && bv_append_be16(&out, (uint16_t)clip->width);
    ok = ok && bv_append_be16(&out, (uint16_t)clip->height);
    ok = ok && bv_append_be32(&out, 0x00480000u); // 72 dpi
    ok = ok && bv_append_be32(&out, 0x00480000u);
    ok = ok && bv_append_be32(&out, 0u);
    ok = ok && bv_append_be16(&out, 1u); // frame_count
    ok = ok && bv_append_zeros(&out, 32u); // compressorname
    ok = ok && bv_append_be16(&out, 24u); // depth
    ok = ok && bv_append_be16(&out, 0xFFFFu); // pre_defined = -1
    ok = ok && bv_begin_box(&out, (clip->codec == AMI_VT_CODEC_HEVC) ? "hvcC" : "avcC", &cfg_box_pos);
    ok = ok && bv_append_bytes(&out, codec_cfg.data, codec_cfg.len);
    ok = ok && bv_end_box(&out, cfg_box_pos);
    ok = ok && bv_end_box(&out, entry_pos);
    ok = ok && bv_end_box(&out, stsd_pos);

    // stts
    ok = ok && bv_begin_fullbox(&out, "stts", 0u, 0u, &stts_pos);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_append_be32(&out, n);
    ok = ok && bv_append_be32(&out, sample_delta);
    ok = ok && bv_end_box(&out, stts_pos);

    // stss (if any sync samples)
    if (sync_count > 0) {
        ok = ok && bv_begin_fullbox(&out, "stss", 0u, 0u, &stss_pos);
        ok = ok && bv_append_be32(&out, sync_count);
        for (uint32_t i = 0; i < sync_count; i++) {
            ok = ok && bv_append_be32(&out, sync_indices[i]);
        }
        ok = ok && bv_end_box(&out, stss_pos);
    }

    // stsc (all samples in 1 contiguous chunk)
    ok = ok && bv_begin_fullbox(&out, "stsc", 0u, 0u, &stsc_pos);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_append_be32(&out, n);
    ok = ok && bv_append_be32(&out, 1u);
    ok = ok && bv_end_box(&out, stsc_pos);

    // stsz
    ok = ok && bv_begin_fullbox(&out, "stsz", 0u, 0u, &stsz_pos);
    ok = ok && bv_append_be32(&out, 0u);
    ok = ok && bv_append_be32(&out, n);
    for (uint32_t i = 0; i < n; i++) {
        ok = ok && bv_append_be32(&out, sample_sizes[i]);
    }
    ok = ok && bv_end_box(&out, stsz_pos);

    // stco (chunk 1 offset patched right before mdat)
    ok = ok && bv_begin_fullbox(&out, "stco", 0u, 0u, &stco_pos);
    ok = ok && bv_append_be32(&out, 1u);
    stco_offset_pos = out.len;
    ok = ok && bv_append_be32(&out, 0u);
    ok = ok && bv_end_box(&out, stco_pos);

    ok = ok && bv_end_box(&out, stbl_pos);
    ok = ok && bv_end_box(&out, minf_pos);
    ok = ok && bv_end_box(&out, mdia_pos);
    ok = ok && bv_end_box(&out, trak_pos);
    ok = ok && bv_end_box(&out, moov_pos);

    // 3. mdat
    if (ok) {
        const size_t chunk_off = out.len + 8u;
        if (chunk_off > UINT32_MAX || mdat_payload_bytes > UINT32_MAX - 8u) {
            ok = false;
        } else {
            ami_write_be32(out.data + stco_offset_pos, (uint32_t)chunk_off);
        }
    }
    ok = ok && bv_begin_box(&out, "mdat", &mdat_pos);
    if (ok && mdat_payload_bytes > 0) {
        if (!bv_reserve(&out, (size_t)mdat_payload_bytes)) {
            ok = false;
        } else {
            for (uint32_t i = 0; i < n; i++) {
                CMBlockBufferRef bb = CMSampleBufferGetDataBuffer(clip->samples[i]);
                const uint32_t sz = sample_sizes[i];
                if (CMBlockBufferCopyDataBytes(bb, 0, sz, out.data + out.len) != kCMBlockBufferNoErr) {
                    ok = false;
                    break;
                }
                out.len += sz;
            }
        }
    }
    ok = ok && bv_end_box(&out, mdat_pos);

    free(sample_sizes);
    free(sync_indices);
    free(codec_cfg.data);

    if (!ok) {
        free(out.data);
        return AMI_ERR_OUT_OF_MEMORY;
    }

    *out_bytes = out.data;
    *out_len = out.len;
    return AMI_OK;
}

typedef struct ami_vt_compressed_sample {
    const uint8_t *data;
    size_t size;
    int64_t dts;
    int64_t pts;
    int64_t duration;
    bool is_sync;
} ami_vt_compressed_sample;

typedef struct ami_vt_batch_collector {
    pthread_mutex_t mu;
    CVPixelBufferRef *slots;
    int64_t *pts_vals;
    int32_t *pts_scales;
    uint32_t capacity;
    OSStatus first_error;
} ami_vt_batch_collector;

static void vt_batch_decompress_callback(void *decompressionOutputRefCon,
                                         void *sourceFrameRefCon,
                                         OSStatus status,
                                         VTDecodeInfoFlags infoFlags,
                                         CVImageBufferRef imageBuffer,
                                         CMTime presentationTimeStamp,
                                         CMTime presentationDuration) {
    (void)infoFlags;
    (void)presentationDuration;
    ami_vt_batch_collector *col = (ami_vt_batch_collector *)decompressionOutputRefCon;
    if (col == NULL) {
        return;
    }
    pthread_mutex_lock(&col->mu);
    if (status != noErr) {
        if (col->first_error == noErr) {
            col->first_error = status;
        }
        pthread_mutex_unlock(&col->mu);
        return;
    }
    const uintptr_t slot_1based = (uintptr_t)sourceFrameRefCon;
    if (slot_1based >= 1u && slot_1based <= (uintptr_t)col->capacity && imageBuffer != NULL) {
        const uint32_t slot = (uint32_t)(slot_1based - 1u);
        if (col->slots[slot] != NULL) {
            CVPixelBufferRelease(col->slots[slot]);
        }
        CVPixelBufferRetain(imageBuffer);
        col->slots[slot] = imageBuffer;
        col->pts_vals[slot] = presentationTimeStamp.value;
        col->pts_scales[slot] = presentationTimeStamp.timescale;
    }
    pthread_mutex_unlock(&col->mu);
}

typedef struct ami_pres_order_entry {
    int64_t pts;
    uint32_t decode_idx;
} ami_pres_order_entry;

static int cmp_pres_order(const void *a, const void *b) {
    const ami_pres_order_entry *pa = (const ami_pres_order_entry *)a;
    const ami_pres_order_entry *pb = (const ami_pres_order_entry *)b;
    if (pa->pts < pb->pts) {
        return -1;
    }
    if (pa->pts > pb->pts) {
        return 1;
    }
    if (pa->decode_idx < pb->decode_idx) {
        return -1;
    }
    if (pa->decode_idx > pb->decode_idx) {
        return 1;
    }
    return 0;
}

static ami_status decode_compressed_samples_iosurface(
    ami_device *device,
    CMVideoFormatDescriptionRef fmt_desc,
    const ami_vt_compressed_sample *samples,
    uint32_t sample_count,
    uint32_t timescale,
    bool has_reordering,
    uint32_t max_frames,
    ami_vision_frame **out_frames,
    uint32_t *out_frame_count) {
    if (device == NULL || fmt_desc == NULL || samples == NULL || sample_count == 0 ||
        timescale == 0 || max_frames == 0 || max_frames > AMI_VISION_MAX_VIDEO_FRAMES ||
        out_frames == NULL || out_frame_count == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_frame_count = 0;
    for (uint32_t i = 0; i < max_frames; i++) {
        out_frames[i] = NULL;
    }

    const CMVideoDimensions dims = CMVideoFormatDescriptionGetDimensions(fmt_desc);
    if (dims.width <= 0 || dims.height <= 0 ||
        dims.width > (int32_t)AMI_VISION_MAX_BYTES_IMAGE_DIM ||
        dims.height > (int32_t)AMI_VISION_MAX_BYTES_IMAGE_DIM) {
        return AMI_ERR_VISION;
    }

    const uint32_t k_count = (sample_count < max_frames) ? sample_count : max_frames;

    uint32_t *want_slot = calloc(sample_count, sizeof(uint32_t));
    bool *need_decode = calloc(sample_count, sizeof(bool));
    ami_pres_order_entry *pres = calloc(sample_count, sizeof(ami_pres_order_entry));
    if (want_slot == NULL || need_decode == NULL || pres == NULL) {
        free(want_slot);
        free(need_decode);
        free(pres);
        return AMI_ERR_OUT_OF_MEMORY;
    }

    for (uint32_t i = 0; i < sample_count; i++) {
        pres[i].pts = samples[i].pts;
        pres[i].decode_idx = i;
    }
    if (has_reordering) {
        qsort(pres, sample_count, sizeof(ami_pres_order_entry), cmp_pres_order);
    }

    for (uint32_t k = 0; k < k_count; k++) {
        const uint32_t pres_idx = (k_count == 1u)
            ? 0u
            : (uint32_t)(((uint64_t)k * (uint64_t)(sample_count - 1u)) / (uint64_t)(k_count - 1u));
        const uint32_t d_idx = pres[pres_idx].decode_idx;
        want_slot[d_idx] = k + 1u;

        uint32_t sync_start = d_idx;
        while (sync_start > 0 && !samples[sync_start].is_sync) {
            sync_start--;
        }
        uint32_t gop_end = d_idx;
        if (has_reordering) {
            while (gop_end + 1u < sample_count && !samples[gop_end + 1u].is_sync) {
                gop_end++;
            }
        }
        for (uint32_t j = sync_start; j <= gop_end; j++) {
            need_decode[j] = true;
        }
    }
    free(pres);

    ami_vt_batch_collector col = { 0 };
    if (pthread_mutex_init(&col.mu, NULL) != 0) {
        free(want_slot);
        free(need_decode);
        return AMI_ERR_VISION;
    }
    col.slots = calloc(k_count, sizeof(CVPixelBufferRef));
    col.pts_vals = calloc(k_count, sizeof(int64_t));
    col.pts_scales = calloc(k_count, sizeof(int32_t));
    col.capacity = k_count;
    if (col.slots == NULL || col.pts_vals == NULL || col.pts_scales == NULL) {
        free(col.slots);
        free(col.pts_vals);
        free(col.pts_scales);
        pthread_mutex_destroy(&col.mu);
        free(want_slot);
        free(need_decode);
        return AMI_ERR_OUT_OF_MEMORY;
    }

    ami_status st = AMI_OK;
    VTDecompressionSessionRef vt_session = NULL;

    @autoreleasepool {
        NSDictionary *dec_spec = @{
            (__bridge id)kVTVideoDecoderSpecification_EnableHardwareAcceleratedVideoDecoder: @YES,
        };
        NSDictionary *dst_attrs = @{
            (__bridge id)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_420YpCbCr8BiPlanarFullRange),
            (__bridge id)kCVPixelBufferIOSurfacePropertiesKey: @{},
            (__bridge id)kCVPixelBufferMetalCompatibilityKey: @YES,
            (__bridge id)kCVPixelBufferBytesPerRowAlignmentKey: @64,
        };
        VTDecompressionOutputCallbackRecord cb = {
            .decompressionOutputCallback = vt_batch_decompress_callback,
            .decompressionOutputRefCon = &col,
        };
        OSStatus os_st = VTDecompressionSessionCreate(
            kCFAllocatorDefault,
            fmt_desc,
            (__bridge CFDictionaryRef)dec_spec,
            (__bridge CFDictionaryRef)dst_attrs,
            &cb,
            &vt_session);
        if (os_st != noErr || vt_session == NULL) {
            st = AMI_ERR_VISION;
        }
    }

    if (st == AMI_OK) {
        for (uint32_t i = 0; i < sample_count; i++) {
            if (!need_decode[i]) {
                continue;
            }
            if (i > 0 && !need_decode[i - 1u]) {
                VTDecompressionSessionFinishDelayedFrames(vt_session);
                VTDecompressionSessionWaitForAsynchronousFrames(vt_session);
            }

            const ami_vt_compressed_sample *s = &samples[i];
            CMBlockBufferRef block_buf = NULL;
            OSStatus os_st = CMBlockBufferCreateWithMemoryBlock(
                kCFAllocatorDefault,
                (void *)(uintptr_t)s->data,
                s->size,
                kCFAllocatorNull,
                NULL,
                0,
                s->size,
                0,
                &block_buf);
            if (os_st != kCMBlockBufferNoErr || block_buf == NULL) {
                st = AMI_ERR_VISION;
                break;
            }

            CMSampleTimingInfo timing = {
                .duration = CMTimeMake(s->duration > 0 ? s->duration : 1, (int32_t)timescale),
                .presentationTimeStamp = CMTimeMake(s->pts, (int32_t)timescale),
                .decodeTimeStamp = CMTimeMake(s->dts, (int32_t)timescale),
            };
            size_t sample_sz = s->size;
            CMSampleBufferRef sample_buf = NULL;
            os_st = CMSampleBufferCreateReady(
                kCFAllocatorDefault,
                block_buf,
                fmt_desc,
                1,
                1,
                &timing,
                1,
                &sample_sz,
                &sample_buf);
            CFRelease(block_buf);
            if (os_st != noErr || sample_buf == NULL) {
                st = AMI_ERR_VISION;
                break;
            }

            os_st = VTDecompressionSessionDecodeFrame(
                vt_session,
                sample_buf,
                0,
                (void *)(uintptr_t)want_slot[i],
                NULL);
            CFRelease(sample_buf);
            if (os_st != noErr) {
                st = AMI_ERR_VISION;
                break;
            }
        }

        if (st == AMI_OK) {
            OSStatus finish_st = VTDecompressionSessionFinishDelayedFrames(vt_session);
            OSStatus wait_st = VTDecompressionSessionWaitForAsynchronousFrames(vt_session);
            if (finish_st != noErr || wait_st != noErr) {
                st = AMI_ERR_VISION;
            }
        } else {
            VTDecompressionSessionWaitForAsynchronousFrames(vt_session);
        }
    }

    if (vt_session != NULL) {
        VTDecompressionSessionInvalidate(vt_session);
        CFRelease(vt_session);
    }

    free(want_slot);
    free(need_decode);

    id<MTLDevice> mtl_dev = device_of(device);
    if (st == AMI_OK) {
        pthread_mutex_lock(&col.mu);
        if (col.first_error != noErr) {
            st = AMI_ERR_VISION;
        } else {
            for (uint32_t k = 0; k < k_count; k++) {
                if (col.slots[k] == NULL) {
                    st = AMI_ERR_VISION;
                    break;
                }
            }
        }
        pthread_mutex_unlock(&col.mu);
    }

    uint32_t built = 0;
    if (st == AMI_OK) {
        for (uint32_t k = 0; k < k_count; k++) {
            st = vision_frame_init_from_mtl_device(
                mtl_dev,
                col.slots[k],
                0u,
                1u,
                col.pts_vals[k],
                col.pts_scales[k],
                &out_frames[k]);
            if (st != AMI_OK) {
                break;
            }
            built++;
        }
    }

    if (st != AMI_OK) {
        for (uint32_t k = 0; k < built; k++) {
            ami_vision_frame_destroy(out_frames[k]);
            out_frames[k] = NULL;
        }
    } else {
        *out_frame_count = built;
    }

    for (uint32_t k = 0; k < k_count; k++) {
        if (col.slots[k] != NULL) {
            CVPixelBufferRelease(col.slots[k]);
        }
    }
    free(col.slots);
    free(col.pts_vals);
    free(col.pts_scales);
    pthread_mutex_destroy(&col.mu);
    return st;
}

// ----------------------------------------------------------------------------
// Pure-C Bounds-Checked ISO-BMFF (MP4 / MOV) Demuxer
// ----------------------------------------------------------------------------

static int iso_next_box(const uint8_t *buf, size_t *cursor, size_t parent_end,
                        uint32_t *out_type, size_t *out_payload_start,
                        size_t *out_box_end) {
    if (*cursor > parent_end) {
        return -1;
    }
    if (*cursor == parent_end) {
        return 0;
    }
    const size_t avail = parent_end - *cursor;
    if (avail < 8u) {
        return -1;
    }
    const uint32_t size32 = ami_read_be32(buf + *cursor);
    const uint32_t box_type = ami_read_be32(buf + *cursor + 4u);
    size_t hdr_len = 8u;
    uint64_t box_size = (uint64_t)size32;
    if (size32 == 1u) {
        if (avail < 16u) {
            return -1;
        }
        box_size = ami_read_be64(buf + *cursor + 8u);
        hdr_len = 16u;
    } else if (size32 == 0u) {
        box_size = (uint64_t)avail;
    }
    if (box_size < (uint64_t)hdr_len || box_size > (uint64_t)avail) {
        return -1;
    }
    *out_type = box_type;
    *out_payload_start = *cursor + hdr_len;
    *out_box_end = *cursor + (size_t)box_size;
    *cursor = *out_box_end;
    return 1;
}

typedef struct ami_iso_video_track {
    bool seen_hdlr;
    uint32_t handler_type;
    uint32_t timescale;
    uint32_t codec;
    bool has_codec_config;
    const uint8_t *codec_config_ptr;
    size_t codec_config_len;
    uint32_t width;
    uint32_t height;
    const uint8_t *stsz_ptr;
    size_t stsz_len;
    const uint8_t *stsc_ptr;
    size_t stsc_len;
    const uint8_t *stco_ptr;
    size_t stco_len;
    bool stco_is_64;
    const uint8_t *stss_ptr;
    size_t stss_len;
    const uint8_t *stts_ptr;
    size_t stts_len;
    const uint8_t *ctts_ptr;
    size_t ctts_len;
} ami_iso_video_track;

static bool iso_parse_stsd(const uint8_t *buf, size_t start, size_t end,
                           ami_iso_video_track *trk) {
    if (end - start < 8u) {
        return false;
    }
    const uint32_t entry_count = ami_read_be32(buf + start + 4u);
    if (entry_count == 0 || entry_count > 64u) {
        return false;
    }
    size_t cur = start + 8u;
    for (uint32_t e = 0; e < entry_count; e++) {
        uint32_t etype = 0;
        size_t estart = 0;
        size_t eend = 0;
        const int rc = iso_next_box(buf, &cur, end, &etype, &estart, &eend);
        if (rc <= 0) {
            return false;
        }
        const bool is_h264 = (etype == AMI_ISO_FOURCC('a', 'v', 'c', '1') ||
                              etype == AMI_ISO_FOURCC('a', 'v', 'c', '3'));
        const bool is_hevc = (etype == AMI_ISO_FOURCC('h', 'v', 'c', '1') ||
                              etype == AMI_ISO_FOURCC('h', 'e', 'v', '1'));
        if (!is_h264 && !is_hevc) {
            continue;
        }
        if (eend - estart < 78u) {
            return false;
        }
        const uint16_t w = ami_read_be16(buf + estart + 24u);
        const uint16_t h = ami_read_be16(buf + estart + 26u);
        size_t sub_cur = estart + 78u;
        const uint32_t want_cfg = is_hevc
            ? AMI_ISO_FOURCC('h', 'v', 'c', 'C')
            : AMI_ISO_FOURCC('a', 'v', 'c', 'C');
        while (sub_cur < eend) {
            uint32_t stype = 0;
            size_t sstart = 0;
            size_t send = 0;
            const int src = iso_next_box(buf, &sub_cur, eend, &stype, &sstart, &send);
            if (src < 0) {
                return false;
            }
            if (src == 0) {
                break;
            }
            if (stype == want_cfg && send > sstart) {
                trk->codec = is_hevc ? AMI_VT_CODEC_HEVC : AMI_VT_CODEC_H264;
                trk->has_codec_config = true;
                trk->codec_config_ptr = buf + sstart;
                trk->codec_config_len = send - sstart;
                trk->width = w;
                trk->height = h;
                return true;
            }
        }
    }
    return true;
}

static bool iso_parse_stbl(const uint8_t *buf, size_t start, size_t end,
                           ami_iso_video_track *trk) {
    size_t cur = start;
    while (cur < end) {
        uint32_t btype = 0;
        size_t bstart = 0;
        size_t bend = 0;
        const int rc = iso_next_box(buf, &cur, end, &btype, &bstart, &bend);
        if (rc < 0) {
            return false;
        }
        if (rc == 0) {
            break;
        }
        const size_t blen = bend - bstart;
        if (btype == AMI_ISO_FOURCC('s', 't', 's', 'd')) {
            if (!iso_parse_stsd(buf, bstart, bend, trk)) {
                return false;
            }
        } else if (btype == AMI_ISO_FOURCC('s', 't', 's', 'z')) {
            trk->stsz_ptr = buf + bstart;
            trk->stsz_len = blen;
        } else if (btype == AMI_ISO_FOURCC('s', 't', 's', 'c')) {
            trk->stsc_ptr = buf + bstart;
            trk->stsc_len = blen;
        } else if (btype == AMI_ISO_FOURCC('s', 't', 'c', 'o')) {
            trk->stco_ptr = buf + bstart;
            trk->stco_len = blen;
            trk->stco_is_64 = false;
        } else if (btype == AMI_ISO_FOURCC('c', 'o', '6', '4')) {
            trk->stco_ptr = buf + bstart;
            trk->stco_len = blen;
            trk->stco_is_64 = true;
        } else if (btype == AMI_ISO_FOURCC('s', 't', 's', 's')) {
            trk->stss_ptr = buf + bstart;
            trk->stss_len = blen;
        } else if (btype == AMI_ISO_FOURCC('s', 't', 't', 's')) {
            trk->stts_ptr = buf + bstart;
            trk->stts_len = blen;
        } else if (btype == AMI_ISO_FOURCC('c', 't', 't', 's')) {
            trk->ctts_ptr = buf + bstart;
            trk->ctts_len = blen;
        }
    }
    return true;
}

static bool iso_parse_minf(const uint8_t *buf, size_t start, size_t end,
                           ami_iso_video_track *trk) {
    size_t cur = start;
    while (cur < end) {
        uint32_t btype = 0;
        size_t bstart = 0;
        size_t bend = 0;
        const int rc = iso_next_box(buf, &cur, end, &btype, &bstart, &bend);
        if (rc < 0) {
            return false;
        }
        if (rc == 0) {
            break;
        }
        if (btype == AMI_ISO_FOURCC('s', 't', 'b', 'l')) {
            if (!iso_parse_stbl(buf, bstart, bend, trk)) {
                return false;
            }
        }
    }
    return true;
}

static bool iso_parse_mdia(const uint8_t *buf, size_t start, size_t end,
                           ami_iso_video_track *trk) {
    size_t cur = start;
    while (cur < end) {
        uint32_t btype = 0;
        size_t bstart = 0;
        size_t bend = 0;
        const int rc = iso_next_box(buf, &cur, end, &btype, &bstart, &bend);
        if (rc < 0) {
            return false;
        }
        if (rc == 0) {
            break;
        }
        const size_t blen = bend - bstart;
        if (btype == AMI_ISO_FOURCC('m', 'd', 'h', 'd')) {
            if (blen < 4u) {
                return false;
            }
            const uint8_t ver = buf[bstart];
            if (ver == 0u) {
                if (blen < 20u) {
                    return false;
                }
                trk->timescale = ami_read_be32(buf + bstart + 12u);
            } else if (ver == 1u) {
                if (blen < 28u) {
                    return false;
                }
                trk->timescale = ami_read_be32(buf + bstart + 20u);
            }
        } else if (btype == AMI_ISO_FOURCC('h', 'd', 'l', 'r')) {
            if (blen < 12u) {
                return false;
            }
            trk->seen_hdlr = true;
            trk->handler_type = ami_read_be32(buf + bstart + 8u);
        } else if (btype == AMI_ISO_FOURCC('m', 'i', 'n', 'f')) {
            if (!iso_parse_minf(buf, bstart, bend, trk)) {
                return false;
            }
        }
    }
    return true;
}

static bool iso_parse_trak(const uint8_t *buf, size_t start, size_t end,
                           ami_iso_video_track *trk) {
    memset(trk, 0, sizeof(*trk));
    trk->timescale = 30000u;
    size_t cur = start;
    while (cur < end) {
        uint32_t btype = 0;
        size_t bstart = 0;
        size_t bend = 0;
        const int rc = iso_next_box(buf, &cur, end, &btype, &bstart, &bend);
        if (rc < 0) {
            return false;
        }
        if (rc == 0) {
            break;
        }
        if (btype == AMI_ISO_FOURCC('m', 'd', 'i', 'a')) {
            if (!iso_parse_mdia(buf, bstart, bend, trk)) {
                return false;
            }
        }
    }
    return true;
}

static ami_status parse_avcc_format_desc(const uint8_t *cfg, size_t cfg_len,
                                          CMVideoFormatDescriptionRef *out_fmt) {
    *out_fmt = NULL;
    if (cfg == NULL || cfg_len < 7u || cfg[0] != 1u) {
        return AMI_ERR_VISION;
    }
    const int nal_len_size = (int)((cfg[4] & 0x03u) + 1u);
    if (nal_len_size != 1 && nal_len_size != 2 && nal_len_size != 4) {
        return AMI_ERR_VISION;
    }
    const uint8_t num_sps = cfg[5] & 0x1Fu;
    size_t p = 6u;

    const uint8_t *ptrs[64];
    size_t sizes[64];
    size_t total = 0;

    for (uint8_t i = 0; i < num_sps; i++) {
        if (p + 2u > cfg_len) {
            return AMI_ERR_VISION;
        }
        const uint16_t sz = ami_read_be16(cfg + p);
        p += 2u;
        if (sz == 0 || sz > cfg_len - p || total >= 64u) {
            return AMI_ERR_VISION;
        }
        ptrs[total] = cfg + p;
        sizes[total] = sz;
        total++;
        p += sz;
    }
    if (p >= cfg_len) {
        return AMI_ERR_VISION;
    }
    const uint8_t num_pps = cfg[p++];
    if (num_sps == 0 || num_pps == 0) {
        return AMI_ERR_VISION;
    }
    for (uint8_t i = 0; i < num_pps; i++) {
        if (p + 2u > cfg_len) {
            return AMI_ERR_VISION;
        }
        const uint16_t sz = ami_read_be16(cfg + p);
        p += 2u;
        if (sz == 0 || sz > cfg_len - p || total >= 64u) {
            return AMI_ERR_VISION;
        }
        ptrs[total] = cfg + p;
        sizes[total] = sz;
        total++;
        p += sz;
    }

    CMFormatDescriptionRef fmt = NULL;
    OSStatus os_st = CMVideoFormatDescriptionCreateFromH264ParameterSets(
        kCFAllocatorDefault, total, ptrs, sizes, nal_len_size, &fmt);
    if (os_st != noErr || fmt == NULL) {
        return AMI_ERR_VISION;
    }
    *out_fmt = (CMVideoFormatDescriptionRef)fmt;
    return AMI_OK;
}

static ami_status parse_hvcc_format_desc(const uint8_t *cfg, size_t cfg_len,
                                          CMVideoFormatDescriptionRef *out_fmt) {
    *out_fmt = NULL;
    if (cfg == NULL || cfg_len < 23u || cfg[0] != 1u) {
        return AMI_ERR_VISION;
    }
    const int nal_len_size = (int)((cfg[21] & 0x03u) + 1u);
    if (nal_len_size != 1 && nal_len_size != 2 && nal_len_size != 4) {
        return AMI_ERR_VISION;
    }
    const uint8_t num_arrays = cfg[22];
    size_t p = 23u;

    const uint8_t *ptrs[64];
    size_t sizes[64];
    size_t total = 0;

    for (uint8_t a = 0; a < num_arrays; a++) {
        if (p + 3u > cfg_len) {
            return AMI_ERR_VISION;
        }
        const uint16_t num_nalus = ami_read_be16(cfg + p + 1u);
        p += 3u;
        for (uint16_t j = 0; j < num_nalus; j++) {
            if (p + 2u > cfg_len) {
                return AMI_ERR_VISION;
            }
            const uint16_t sz = ami_read_be16(cfg + p);
            p += 2u;
            if (sz == 0 || sz > cfg_len - p || total >= 64u) {
                return AMI_ERR_VISION;
            }
            ptrs[total] = cfg + p;
            sizes[total] = sz;
            total++;
            p += sz;
        }
    }
    if (total == 0) {
        return AMI_ERR_VISION;
    }

    CMFormatDescriptionRef fmt = NULL;
    OSStatus os_st = CMVideoFormatDescriptionCreateFromHEVCParameterSets(
        kCFAllocatorDefault, total, ptrs, sizes, nal_len_size, NULL, &fmt);
    if (os_st != noErr || fmt == NULL) {
        return AMI_ERR_VISION;
    }
    *out_fmt = (CMVideoFormatDescriptionRef)fmt;
    return AMI_OK;
}

static bool is_isobmff_container(const uint8_t *bytes, size_t len) {
    if (bytes == NULL || len < 8u) {
        return false;
    }
    const uint32_t tag = ami_read_be32(bytes + 4u);
    if (tag == AMI_ISO_FOURCC('f', 't', 'y', 'p') ||
        tag == AMI_ISO_FOURCC('m', 'o', 'o', 'v')) {
        return true;
    }
    if (tag == AMI_ISO_FOURCC('w', 'i', 'd', 'e') ||
        tag == AMI_ISO_FOURCC('f', 'r', 'e', 'e') ||
        tag == AMI_ISO_FOURCC('s', 'k', 'i', 'p') ||
        tag == AMI_ISO_FOURCC('m', 'd', 'a', 't')) {
        size_t cur = 0;
        uint32_t boxes_seen = 0;
        while (cur < len && boxes_seen < 32u) {
            uint32_t btype = 0;
            size_t bstart = 0;
            size_t bend = 0;
            const int rc = iso_next_box(bytes, &cur, len, &btype, &bstart, &bend);
            if (rc <= 0) {
                break;
            }
            boxes_seen++;
            if (btype == AMI_ISO_FOURCC('m', 'o', 'o', 'v')) {
                return true;
            }
        }
    }
    return false;
}

static ami_status decode_isobmff_bytes(ami_device *device,
                                       const uint8_t *bytes, size_t len,
                                       uint32_t max_frames,
                                       ami_vision_frame **out_frames,
                                       uint32_t *out_frame_count) {
    bool found_moov = false;
    bool found_video = false;
    ami_iso_video_track video_trk = { 0 };

    size_t top_cur = 0;
    while (top_cur < len) {
        uint32_t btype = 0;
        size_t bstart = 0;
        size_t bend = 0;
        const int rc = iso_next_box(bytes, &top_cur, len, &btype, &bstart, &bend);
        if (rc < 0) {
            return AMI_ERR_VISION;
        }
        if (rc == 0) {
            break;
        }
        if (btype == AMI_ISO_FOURCC('m', 'o', 'o', 'v')) {
            found_moov = true;
            size_t mcur = bstart;
            while (mcur < bend) {
                uint32_t mtype = 0;
                size_t mstart = 0;
                size_t mend = 0;
                const int mrc = iso_next_box(bytes, &mcur, bend, &mtype, &mstart, &mend);
                if (mrc < 0) {
                    return AMI_ERR_VISION;
                }
                if (mrc == 0) {
                    break;
                }
                if (mtype == AMI_ISO_FOURCC('t', 'r', 'a', 'k') && !found_video) {
                    ami_iso_video_track cand;
                    if (!iso_parse_trak(bytes, mstart, mend, &cand)) {
                        return AMI_ERR_VISION;
                    }
                    if (cand.seen_hdlr && cand.handler_type != AMI_ISO_FOURCC('v', 'i', 'd', 'e')) {
                        continue;
                    }
                    if (cand.has_codec_config) {
                        video_trk = cand;
                        found_video = true;
                    }
                }
            }
        }
    }

    if (!found_moov || !found_video ||
        video_trk.stsz_ptr == NULL || video_trk.stsc_ptr == NULL || video_trk.stco_ptr == NULL ||
        video_trk.timescale == 0) {
        return AMI_ERR_VISION;
    }

    // Parse stsz
    if (video_trk.stsz_len < 12u) {
        return AMI_ERR_VISION;
    }
    const uint32_t uniform_size = ami_read_be32(video_trk.stsz_ptr + 4u);
    const uint32_t sample_count = ami_read_be32(video_trk.stsz_ptr + 8u);
    if (sample_count == 0 || sample_count > 250000u) {
        return AMI_ERR_VISION;
    }
    if (uniform_size == 0) {
        if (video_trk.stsz_len < 12u + (size_t)sample_count * 4u) {
            return AMI_ERR_VISION;
        }
    } else if ((uint64_t)uniform_size * (uint64_t)sample_count > (uint64_t)len) {
        return AMI_ERR_VISION;
    }

    // Parse stco / co64
    if (video_trk.stco_len < 8u) {
        return AMI_ERR_VISION;
    }
    const uint32_t chunk_count = ami_read_be32(video_trk.stco_ptr + 4u);
    const size_t chunk_entry_sz = video_trk.stco_is_64 ? 8u : 4u;
    if (chunk_count == 0 || chunk_count > 250000u ||
        video_trk.stco_len - 8u < (size_t)chunk_count * chunk_entry_sz) {
        return AMI_ERR_VISION;
    }

    // Parse stsc
    if (video_trk.stsc_len < 8u) {
        return AMI_ERR_VISION;
    }
    const uint32_t stsc_count = ami_read_be32(video_trk.stsc_ptr + 4u);
    if (stsc_count == 0 || stsc_count > 250000u ||
        video_trk.stsc_len - 8u < (size_t)stsc_count * 12u) {
        return AMI_ERR_VISION;
    }
    uint32_t prev_fc = 0;
    for (uint32_t r = 0; r < stsc_count; r++) {
        const uint8_t *rp = video_trk.stsc_ptr + 8u + (size_t)r * 12u;
        const uint32_t fc = ami_read_be32(rp);
        const uint32_t spc = ami_read_be32(rp + 4u);
        if (fc == 0 || spc == 0 || (r == 0 && fc != 1u) || (r > 0 && fc <= prev_fc)) {
            return AMI_ERR_VISION;
        }
        prev_fc = fc;
    }

    ami_vt_compressed_sample *samples = calloc(sample_count, sizeof(ami_vt_compressed_sample));
    if (samples == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }

    // Resolve chunk offsets + stsc + stsz -> exact verified sample slices
    uint32_t stsc_idx = 0;
    uint32_t s_idx = 0;
    for (uint32_t c = 0; c < chunk_count && s_idx < sample_count; c++) {
        const uint32_t chunk_1 = c + 1u;
        while (stsc_idx + 1u < stsc_count) {
            const uint32_t next_fc = ami_read_be32(
                video_trk.stsc_ptr + 8u + (size_t)(stsc_idx + 1u) * 12u);
            if (chunk_1 >= next_fc) {
                stsc_idx++;
            } else {
                break;
            }
        }
        const uint32_t spc = ami_read_be32(
            video_trk.stsc_ptr + 8u + (size_t)stsc_idx * 12u + 4u);
        uint64_t off = video_trk.stco_is_64
            ? ami_read_be64(video_trk.stco_ptr + 8u + (size_t)c * 8u)
            : (uint64_t)ami_read_be32(video_trk.stco_ptr + 8u + (size_t)c * 4u);
        for (uint32_t j = 0; j < spc && s_idx < sample_count; j++) {
            const uint32_t sz = (uniform_size != 0)
                ? uniform_size
                : ami_read_be32(video_trk.stsz_ptr + 12u + (size_t)s_idx * 4u);
            if (sz == 0 || off > (uint64_t)len || (uint64_t)sz > (uint64_t)len - off) {
                free(samples);
                return AMI_ERR_VISION;
            }
            samples[s_idx].data = bytes + (size_t)off;
            samples[s_idx].size = (size_t)sz;
            samples[s_idx].is_sync = true;
            off += (uint64_t)sz;
            s_idx++;
        }
    }
    if (s_idx != sample_count) {
        free(samples);
        return AMI_ERR_VISION;
    }

    // Parse stts (DTS + duration)
    if (video_trk.stts_ptr != NULL) {
        if (video_trk.stts_len < 8u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        const uint32_t stts_count = ami_read_be32(video_trk.stts_ptr + 4u);
        if (video_trk.stts_len - 8u < (size_t)stts_count * 8u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        uint32_t ti = 0;
        int64_t cur_dts = 0;
        for (uint32_t e = 0; e < stts_count && ti < sample_count; e++) {
            const uint8_t *ep = video_trk.stts_ptr + 8u + (size_t)e * 8u;
            const uint32_t sc = ami_read_be32(ep);
            const uint32_t delta = ami_read_be32(ep + 4u);
            for (uint32_t k = 0; k < sc && ti < sample_count; k++) {
                samples[ti].dts = cur_dts;
                samples[ti].pts = cur_dts;
                samples[ti].duration = (delta > 0) ? (int64_t)delta : 1;
                cur_dts += (int64_t)delta;
                ti++;
            }
        }
        if (ti != sample_count) {
            free(samples);
            return AMI_ERR_VISION;
        }
    } else {
        const int64_t def_delta = (int64_t)(video_trk.timescale / 30u > 0 ? video_trk.timescale / 30u : 1u);
        for (uint32_t i = 0; i < sample_count; i++) {
            samples[i].dts = (int64_t)i * def_delta;
            samples[i].pts = samples[i].dts;
            samples[i].duration = def_delta;
        }
    }

    // Parse optional ctts (composition time offset -> B-frame reordering)
    bool has_reordering = false;
    if (video_trk.ctts_ptr != NULL) {
        if (video_trk.ctts_len < 8u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        const uint8_t ver = video_trk.ctts_ptr[0];
        const uint32_t ctts_count = ami_read_be32(video_trk.ctts_ptr + 4u);
        if (video_trk.ctts_len - 8u < (size_t)ctts_count * 8u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        uint32_t ci = 0;
        for (uint32_t e = 0; e < ctts_count && ci < sample_count; e++) {
            const uint8_t *ep = video_trk.ctts_ptr + 8u + (size_t)e * 8u;
            const uint32_t sc = ami_read_be32(ep);
            const int64_t off = (ver == 1u)
                ? (int64_t)(int32_t)ami_read_be32(ep + 4u)
                : (int64_t)ami_read_be32(ep + 4u);
            if (off != 0) {
                has_reordering = true;
            }
            for (uint32_t k = 0; k < sc && ci < sample_count; k++) {
                samples[ci].pts = samples[ci].dts + off;
                ci++;
            }
        }
    }

    // Parse optional stss (sync sample table)
    if (video_trk.stss_ptr != NULL) {
        if (video_trk.stss_len < 8u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        const uint32_t sync_cnt = ami_read_be32(video_trk.stss_ptr + 4u);
        if (video_trk.stss_len - 8u < (size_t)sync_cnt * 4u) {
            free(samples);
            return AMI_ERR_VISION;
        }
        for (uint32_t i = 0; i < sample_count; i++) {
            samples[i].is_sync = false;
        }
        for (uint32_t i = 0; i < sync_cnt; i++) {
            const uint32_t s1 = ami_read_be32(video_trk.stss_ptr + 8u + (size_t)i * 4u);
            if (s1 == 0 || s1 > sample_count) {
                free(samples);
                return AMI_ERR_VISION;
            }
            samples[s1 - 1u].is_sync = true;
        }
    }

    CMVideoFormatDescriptionRef fmt_desc = NULL;
    ami_status st = (video_trk.codec == AMI_VT_CODEC_HEVC)
        ? parse_hvcc_format_desc(video_trk.codec_config_ptr, video_trk.codec_config_len, &fmt_desc)
        : parse_avcc_format_desc(video_trk.codec_config_ptr, video_trk.codec_config_len, &fmt_desc);
    if (st != AMI_OK || fmt_desc == NULL) {
        free(samples);
        return AMI_ERR_VISION;
    }

    st = decode_compressed_samples_iosurface(
        device, fmt_desc, samples, sample_count, video_trk.timescale,
        has_reordering, max_frames, out_frames, out_frame_count);
    CFRelease(fmt_desc);
    free(samples);
    return st;
}

// ----------------------------------------------------------------------------
// Raw Annex-B H.264 / HEVC Bitstream Demuxer
// ----------------------------------------------------------------------------

static bool looks_like_annexb(const uint8_t *bytes, size_t len) {
    if (bytes == NULL || len < 4u) {
        return false;
    }
    if (bytes[0] == 0u && bytes[1] == 0u) {
        if (bytes[2] == 1u) {
            return true;
        }
        if (len >= 5u && bytes[2] == 0u && bytes[3] == 1u) {
            return true;
        }
    }
    return false;
}

static size_t find_annexb_start_code(const uint8_t *buf, size_t pos, size_t len,
                                     size_t *out_sc_len) {
    for (size_t i = pos; i + 3u <= len; i++) {
        if (buf[i] == 0u && buf[i + 1u] == 0u) {
            if (buf[i + 2u] == 1u) {
                *out_sc_len = 3u;
                return i;
            }
            if (i + 4u <= len && buf[i + 2u] == 0u && buf[i + 3u] == 1u) {
                *out_sc_len = 4u;
                return i;
            }
        }
    }
    *out_sc_len = 0;
    return len;
}

static ami_status decode_annexb_bytes(ami_device *device,
                                      const uint8_t *bytes, size_t len,
                                      uint32_t max_frames,
                                      ami_vision_frame **out_frames,
                                      uint32_t *out_frame_count) {
    typedef struct nal_slice {
        const uint8_t *ptr;
        size_t len;
    } nal_slice;

    nal_slice *nals = NULL;
    size_t nal_cap = 0;
    size_t nal_count = 0;

    size_t sc_len = 0;
    size_t pos = find_annexb_start_code(bytes, 0u, len, &sc_len);
    while (pos < len) {
        const size_t nal_start = pos + sc_len;
        size_t next_sc_len = 0;
        const size_t next_pos = find_annexb_start_code(bytes, nal_start, len, &next_sc_len);
        size_t nal_end = next_pos;
        while (nal_end > nal_start && bytes[nal_end - 1u] == 0u) {
            nal_end--;
        }
        if (nal_end > nal_start) {
            if (nal_count >= nal_cap) {
                size_t new_cap = (nal_cap == 0) ? 32u : nal_cap * 2u;
                nal_slice *grown = realloc(nals, new_cap * sizeof(nal_slice));
                if (grown == NULL) {
                    free(nals);
                    return AMI_ERR_OUT_OF_MEMORY;
                }
                nals = grown;
                nal_cap = new_cap;
            }
            nals[nal_count++] = (nal_slice){ .ptr = bytes + nal_start, .len = nal_end - nal_start };
        }
        pos = next_pos;
        sc_len = next_sc_len;
    }

    if (nal_count == 0) {
        free(nals);
        return AMI_ERR_VISION;
    }

    bool saw_h264_sps = false;
    bool saw_h264_pps = false;
    bool saw_hevc_vps = false;
    bool saw_hevc_sps = false;
    bool saw_hevc_pps = false;

    for (size_t i = 0; i < nal_count; i++) {
        const uint8_t b0 = nals[i].ptr[0];
        const uint8_t h264_type = b0 & 0x1Fu;
        const uint8_t hevc_type = (b0 >> 1u) & 0x3Fu;
        if (h264_type == 7u && (b0 & 0x80u) == 0) {
            saw_h264_sps = true;
        }
        if (h264_type == 8u && (b0 & 0x80u) == 0) {
            saw_h264_pps = true;
        }
        if (hevc_type == 32u && (b0 & 0x80u) == 0) {
            saw_hevc_vps = true;
        }
        if (hevc_type == 33u && (b0 & 0x80u) == 0) {
            saw_hevc_sps = true;
        }
        if (hevc_type == 34u && (b0 & 0x80u) == 0) {
            saw_hevc_pps = true;
        }
    }

    const bool is_hevc = (saw_hevc_vps && saw_hevc_sps && saw_hevc_pps);
    const bool is_h264 = (!is_hevc && saw_h264_sps && saw_h264_pps);
    if (!is_hevc && !is_h264) {
        free(nals);
        return AMI_ERR_VISION;
    }

    const uint8_t *param_ptrs[64];
    size_t param_sizes[64];
    size_t param_count = 0;

    ami_byte_vec avcc_buf = { 0 };
    typedef struct au_range {
        size_t byte_off;
        size_t byte_len;
        bool is_sync;
    } au_range;
    au_range *aus = calloc(nal_count, sizeof(au_range));
    if (aus == NULL) {
        free(nals);
        return AMI_ERR_OUT_OF_MEMORY;
    }
    uint32_t au_count = 0;
    bool cur_au_open = false;
    size_t cur_au_start = 0;
    bool cur_au_sync = false;

    for (size_t i = 0; i < nal_count; i++) {
        const uint8_t *np = nals[i].ptr;
        const size_t nsz = nals[i].len;
        if (nsz > UINT32_MAX - 4u) {
            free(aus);
            free(avcc_buf.data);
            free(nals);
            return AMI_ERR_VISION;
        }
        bool is_param = false;
        bool is_aud = false;
        bool is_vcl = false;
        bool is_first_slice = false;
        bool is_idr = false;

        if (is_hevc) {
            const uint8_t t = (np[0] >> 1u) & 0x3Fu;
            is_param = (t == 32u || t == 33u || t == 34u);
            is_aud = (t == 35u);
            is_vcl = (t <= 21u);
            is_idr = (t == 19u || t == 20u || t == 21u);
            is_first_slice = is_vcl && nsz >= 3u && ((np[2] & 0x80u) != 0);
        } else {
            const uint8_t t = np[0] & 0x1Fu;
            is_param = (t == 7u || t == 8u);
            is_aud = (t == 9u);
            is_vcl = (t >= 1u && t <= 5u);
            is_idr = (t == 5u);
            is_first_slice = is_vcl && nsz >= 2u && ((np[1] & 0x80u) != 0);
        }

        if (is_param) {
            if (param_count < 64u) {
                param_ptrs[param_count] = np;
                param_sizes[param_count] = nsz;
                param_count++;
            }
            continue;
        }
        if (is_aud || (is_first_slice && cur_au_open)) {
            if (cur_au_open && avcc_buf.len > cur_au_start) {
                aus[au_count++] = (au_range){
                    .byte_off = cur_au_start,
                    .byte_len = avcc_buf.len - cur_au_start,
                    .is_sync = cur_au_sync,
                };
                cur_au_open = false;
                cur_au_sync = false;
            }
            if (is_aud) {
                continue;
            }
        }
        if (is_vcl) {
            if (!cur_au_open) {
                cur_au_open = true;
                cur_au_start = avcc_buf.len;
                cur_au_sync = is_idr;
            } else if (is_idr) {
                cur_au_sync = true;
            }
            if (!bv_append_be32(&avcc_buf, (uint32_t)nsz) ||
                !bv_append_bytes(&avcc_buf, np, nsz)) {
                free(aus);
                free(avcc_buf.data);
                free(nals);
                return AMI_ERR_OUT_OF_MEMORY;
            }
        }
    }
    if (cur_au_open && avcc_buf.len > cur_au_start) {
        aus[au_count++] = (au_range){
            .byte_off = cur_au_start,
            .byte_len = avcc_buf.len - cur_au_start,
            .is_sync = cur_au_sync,
        };
    }
    free(nals);

    if (au_count == 0 || param_count == 0) {
        free(aus);
        free(avcc_buf.data);
        return AMI_ERR_VISION;
    }

    CMFormatDescriptionRef fmt_desc = NULL;
    OSStatus os_st = is_hevc
        ? CMVideoFormatDescriptionCreateFromHEVCParameterSets(
              kCFAllocatorDefault, param_count, param_ptrs, param_sizes, 4, NULL, &fmt_desc)
        : CMVideoFormatDescriptionCreateFromH264ParameterSets(
              kCFAllocatorDefault, param_count, param_ptrs, param_sizes, 4, &fmt_desc);
    if (os_st != noErr || fmt_desc == NULL) {
        free(aus);
        free(avcc_buf.data);
        return AMI_ERR_VISION;
    }

    ami_vt_compressed_sample *samples = calloc(au_count, sizeof(ami_vt_compressed_sample));
    if (samples == NULL) {
        CFRelease(fmt_desc);
        free(aus);
        free(avcc_buf.data);
        return AMI_ERR_OUT_OF_MEMORY;
    }
    for (uint32_t i = 0; i < au_count; i++) {
        samples[i].data = avcc_buf.data + aus[i].byte_off;
        samples[i].size = aus[i].byte_len;
        samples[i].dts = (int64_t)i * 1000;
        samples[i].pts = (int64_t)i * 1000;
        samples[i].duration = 1000;
        samples[i].is_sync = aus[i].is_sync;
    }
    free(aus);

    ami_status st = decode_compressed_samples_iosurface(
        device, (CMVideoFormatDescriptionRef)fmt_desc, samples, au_count, 30000u,
        false, max_frames, out_frames, out_frame_count);
    CFRelease(fmt_desc);
    free(samples);
    free(avcc_buf.data);
    return st;
}

ami_status ami_video_decode_bytes_iosurface(ami_device *device,
                                            const uint8_t *bytes, size_t len,
                                            uint32_t max_frames,
                                            ami_vision_frame **out_frames,
                                            uint32_t *out_frame_count) {
    if (out_frame_count != NULL) {
        *out_frame_count = 0;
    }
    if (out_frames != NULL && max_frames > 0 && max_frames <= AMI_VISION_MAX_VIDEO_FRAMES) {
        for (uint32_t i = 0; i < max_frames; i++) {
            out_frames[i] = NULL;
        }
    }
    if (device == NULL || bytes == NULL || len == 0 ||
        max_frames == 0 || max_frames > AMI_VISION_MAX_VIDEO_FRAMES ||
        out_frames == NULL || out_frame_count == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    // 1. ISO-BMFF MP4 / MOV container path (H.264 / HEVC -> hardware VideoToolbox NV12 IOSurface)
    if (is_isobmff_container(bytes, len)) {
        return decode_isobmff_bytes(device, bytes, len, max_frames, out_frames, out_frame_count);
    }

    // 2. Raw Annex-B H.264 / HEVC bitstream path
    if (looks_like_annexb(bytes, len)) {
        return decode_annexb_bytes(device, bytes, len, max_frames, out_frames, out_frame_count);
    }

    // 3. Multi-frame animated ImageIO (GIF / APNG / WebP / multi-page TIFF) or single-image fallback
    @autoreleasepool {
        NSData *data = [NSData dataWithBytesNoCopy:(void *)(uintptr_t)bytes
                                            length:(NSUInteger)len
                                      freeWhenDone:NO];
        if (data == nil) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        NSDictionary *src_opts = @{
            (__bridge id)kCGImageSourceShouldCache: @NO,
        };
        CGImageSourceRef src = CGImageSourceCreateWithData(
            (__bridge CFDataRef)data, (__bridge CFDictionaryRef)src_opts);
        if (src == NULL) {
            return AMI_ERR_VISION;
        }
        const size_t total = CGImageSourceGetCount(src);
        if (total == 0) {
            CFRelease(src);
            return AMI_ERR_VISION;
        }

        const uint32_t k_count = (total < (size_t)max_frames) ? (uint32_t)total : max_frames;
        NSDictionary *img_opts = @{
            (__bridge id)kCGImageSourceShouldCache: @NO,
            (__bridge id)kCGImageSourceShouldCacheImmediately: @YES,
        };

        uint32_t decoded = 0;
        ami_status st = AMI_OK;
        for (uint32_t k = 0; k < k_count; k++) {
            const size_t idx = (k_count == 1u)
                ? 0u
                : (size_t)(((uint64_t)k * (uint64_t)(total - 1u)) / (uint64_t)(k_count - 1u));
            CGImageRef cg_img = CGImageSourceCreateImageAtIndex(
                src, idx, (__bridge CFDictionaryRef)img_opts);
            if (cg_img == NULL) {
                st = AMI_ERR_VISION;
                break;
            }
            st = vision_frame_from_cg_image(
                device, cg_img, AMI_VISION_MAX_BYTES_IMAGE_DIM, &out_frames[k]);
            CGImageRelease(cg_img);
            if (st != AMI_OK) {
                break;
            }
            if (total > 1u) {
                out_frames[k]->pts_value = (int64_t)idx;
                out_frames[k]->pts_timescale = 30;
            }
            decoded++;
        }
        CFRelease(src);

        if (st != AMI_OK) {
            for (uint32_t k = 0; k < decoded; k++) {
                ami_vision_frame_destroy(out_frames[k]);
                out_frames[k] = NULL;
            }
            return st;
        }
        *out_frame_count = decoded;
        return AMI_OK;
    }
}


// ============================================================================
// M7.5 C19: Rootless Apple Silicon Telemetry & Terminal Raw-Mode Helpers
// ============================================================================

_Static_assert(sizeof(struct termios) <= sizeof(((ami_tty_raw_guard *)0)->opaque),
               "ami_tty_raw_guard.opaque must fit struct termios");

typedef mach_port_t ami_io_object_t;
typedef ami_io_object_t ami_io_iterator_t;
typedef ami_io_object_t ami_io_registry_entry_t;

typedef CFMutableDictionaryRef (*ami_iokit_service_matching_fn)(const char *name);
typedef kern_return_t (*ami_iokit_get_matching_services_fn)(
    mach_port_t main_port,
    CFDictionaryRef matching,
    ami_io_iterator_t *existing);
typedef ami_io_object_t (*ami_iokit_iterator_next_fn)(ami_io_iterator_t iterator);
typedef kern_return_t (*ami_iokit_entry_create_props_fn)(
    ami_io_registry_entry_t entry,
    CFMutableDictionaryRef *properties,
    CFAllocatorRef allocator,
    uint32_t options);
typedef kern_return_t (*ami_iokit_object_release_fn)(ami_io_object_t object);
typedef ami_io_registry_entry_t (*ami_iokit_entry_from_path_fn)(
    mach_port_t main_port,
    const char *path);

typedef struct ami_iokit_api {
    bool ready;
    ami_iokit_service_matching_fn service_matching;
    ami_iokit_get_matching_services_fn get_matching_services;
    ami_iokit_iterator_next_fn iterator_next;
    ami_iokit_entry_create_props_fn entry_create_props;
    ami_iokit_object_release_fn object_release;
    ami_iokit_entry_from_path_fn entry_from_path;
} ami_iokit_api;

static ami_iokit_api g_iokit = { 0 };
static uint32_t g_cpu_cluster_kind[AMI_MAX_CPU_CORES];

static const ami_iokit_api *ami_get_iokit(void) {
    static dispatch_once_t once_tok;
    dispatch_once(&once_tok, ^{
        void *handle = dlopen("/System/Library/Frameworks/IOKit.framework/IOKit",
                              RTLD_LAZY | RTLD_LOCAL);
        if (handle == NULL) {
            return;
        }
        g_iokit.service_matching =
            (ami_iokit_service_matching_fn)dlsym(handle, "IOServiceMatching");
        g_iokit.get_matching_services =
            (ami_iokit_get_matching_services_fn)dlsym(handle, "IOServiceGetMatchingServices");
        g_iokit.iterator_next =
            (ami_iokit_iterator_next_fn)dlsym(handle, "IOIteratorNext");
        g_iokit.entry_create_props =
            (ami_iokit_entry_create_props_fn)dlsym(handle, "IORegistryEntryCreateCFProperties");
        g_iokit.object_release =
            (ami_iokit_object_release_fn)dlsym(handle, "IOObjectRelease");
        g_iokit.entry_from_path =
            (ami_iokit_entry_from_path_fn)dlsym(handle, "IORegistryEntryFromPath");

        if (g_iokit.service_matching != NULL &&
            g_iokit.get_matching_services != NULL &&
            g_iokit.iterator_next != NULL &&
            g_iokit.entry_create_props != NULL &&
            g_iokit.object_release != NULL) {
            g_iokit.ready = true;
        }
    });
    return &g_iokit;
}

static void ami_init_cpu_cluster_topology(void) {
    static dispatch_once_t topo_once;
    dispatch_once(&topo_once, ^{
        for (uint32_t i = 0; i < AMI_MAX_CPU_CORES; i++) {
            g_cpu_cluster_kind[i] = AMI_CPU_CLUSTER_UNKNOWN;
        }
        const ami_iokit_api *io = ami_get_iokit();
        if (!io->ready || io->entry_from_path == NULL) {
            return;
        }
        for (uint32_t i = 0; i < AMI_MAX_CPU_CORES; i++) {
            char path[64];
            int n = snprintf(path, sizeof(path), "IODeviceTree:/cpus/cpu%u", i);
            if (n <= 0 || (size_t)n >= sizeof(path)) {
                break;
            }
            ami_io_registry_entry_t entry = io->entry_from_path(MACH_PORT_NULL, path);
            if (entry == MACH_PORT_NULL) {
                break;
            }
            CFMutableDictionaryRef props = NULL;
            if (io->entry_create_props(entry, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS &&
                props != NULL) {
                CFTypeRef val = CFDictionaryGetValue(props, CFSTR("cluster-type"));
                if (val != NULL) {
                    CFTypeID tid = CFGetTypeID(val);
                    if (tid == CFDataGetTypeID()) {
                        CFDataRef data = (CFDataRef)val;
                        CFIndex len = CFDataGetLength(data);
                        const UInt8 *ptr = CFDataGetBytePtr(data);
                        if (len >= 1 && ptr != NULL) {
                            if (ptr[0] == 'E' || ptr[0] == 'e') {
                                g_cpu_cluster_kind[i] = AMI_CPU_CLUSTER_E;
                            } else if (ptr[0] == 'P' || ptr[0] == 'p') {
                                g_cpu_cluster_kind[i] = AMI_CPU_CLUSTER_P;
                            }
                        }
                    } else if (tid == CFStringGetTypeID()) {
                        char cbuf[8] = { 0 };
                        if (CFStringGetCString((CFStringRef)val, cbuf, sizeof(cbuf),
                                               kCFStringEncodingUTF8)) {
                            if (cbuf[0] == 'E' || cbuf[0] == 'e') {
                                g_cpu_cluster_kind[i] = AMI_CPU_CLUSTER_E;
                            } else if (cbuf[0] == 'P' || cbuf[0] == 'p') {
                                g_cpu_cluster_kind[i] = AMI_CPU_CLUSTER_P;
                            }
                        }
                    }
                }
                CFRelease(props);
            }
            io->object_release(entry);
        }
    });
}

static bool ami_cf_dict_read_u64(CFDictionaryRef dict, CFStringRef key, uint64_t *out_val) {
    if (dict == NULL || key == NULL || out_val == NULL) {
        return false;
    }
    CFTypeRef raw = CFDictionaryGetValue(dict, key);
    if (raw == NULL || CFGetTypeID(raw) != CFNumberGetTypeID()) {
        return false;
    }
    int64_t signed_val = 0;
    if (!CFNumberGetValue((CFNumberRef)raw, kCFNumberSInt64Type, &signed_val)) {
        double dbl_val = 0.0;
        if (!CFNumberGetValue((CFNumberRef)raw, kCFNumberDoubleType, &dbl_val)) {
            return false;
        }
        if (dbl_val < 0.0) {
            *out_val = 0;
        } else {
            *out_val = (uint64_t)dbl_val;
        }
        return true;
    }
    *out_val = (signed_val < 0) ? 0u : (uint64_t)signed_val;
    return true;
}

static void ami_cf_dict_read_cstring(CFDictionaryRef dict, CFStringRef key,
                                     char *out_buf, size_t cap) {
    if (dict == NULL || key == NULL || out_buf == NULL || cap == 0) {
        return;
    }
    CFTypeRef raw = CFDictionaryGetValue(dict, key);
    if (raw == NULL || CFGetTypeID(raw) != CFStringGetTypeID()) {
        return;
    }
    if (!CFStringGetCString((CFStringRef)raw, out_buf, (CFIndex)cap, kCFStringEncodingUTF8)) {
        out_buf[0] = '\0';
    }
}

ami_status ami_telemetry_sample_cpu(ami_telemetry_cpu_sample *out_sample) {
    if (out_sample == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_sample, 0, sizeof(*out_sample));
    ami_init_cpu_cluster_topology();

    mach_port_t host = mach_host_self();
    natural_t cpu_count = 0;
    processor_info_array_t info_array = NULL;
    mach_msg_type_number_t info_count = 0;

    kern_return_t kr = host_processor_info(
        host,
        PROCESSOR_CPU_LOAD_INFO,
        &cpu_count,
        &info_array,
        &info_count);
    mach_port_deallocate(mach_task_self(), host);

    if (kr != KERN_SUCCESS || info_array == NULL) {
        return AMI_ERR_TELEMETRY;
    }

    uint32_t max_from_words = (uint32_t)(info_count / CPU_STATE_MAX);
    uint32_t count = cpu_count;
    if (count > max_from_words) {
        count = max_from_words;
    }
    if (count > AMI_MAX_CPU_CORES) {
        count = AMI_MAX_CPU_CORES;
    }

    out_sample->core_count = count;
    for (uint32_t i = 0; i < count; i++) {
        const integer_t *base = &info_array[(size_t)i * (size_t)CPU_STATE_MAX];
        out_sample->cores[i].user = (uint32_t)base[CPU_STATE_USER];
        out_sample->cores[i].system = (uint32_t)base[CPU_STATE_SYSTEM];
        out_sample->cores[i].idle = (uint32_t)base[CPU_STATE_IDLE];
        out_sample->cores[i].nice = (uint32_t)base[CPU_STATE_NICE];
        out_sample->cores[i].cluster_kind = g_cpu_cluster_kind[i];
    }

    vm_deallocate(
        mach_task_self(),
        (vm_address_t)info_array,
        (vm_size_t)info_count * sizeof(integer_t));

    double avg[3] = { 0.0, 0.0, 0.0 };
    if (getloadavg(avg, 3) == 3) {
        out_sample->loadavg_1m = avg[0];
        out_sample->loadavg_5m = avg[1];
        out_sample->loadavg_15m = avg[2];
    }

    return AMI_OK;
}

ami_status ami_telemetry_sample_gpu(ami_telemetry_gpu_sample *out_sample) {
    if (out_sample == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_sample, 0, sizeof(*out_sample));

    @autoreleasepool {
        NSProcessInfoThermalState ts = [[NSProcessInfo processInfo] thermalState];
        switch (ts) {
            case NSProcessInfoThermalStateNominal:
                out_sample->thermal_state = AMI_THERMAL_NOMINAL;
                break;
            case NSProcessInfoThermalStateFair:
                out_sample->thermal_state = AMI_THERMAL_FAIR;
                break;
            case NSProcessInfoThermalStateSerious:
                out_sample->thermal_state = AMI_THERMAL_SERIOUS;
                break;
            case NSProcessInfoThermalStateCritical:
                out_sample->thermal_state = AMI_THERMAL_CRITICAL;
                break;
            default:
                out_sample->thermal_state = AMI_THERMAL_NOMINAL;
                break;
        }

        const ami_iokit_api *io = ami_get_iokit();
        if (!io->ready) {
            return AMI_OK;
        }

        CFMutableDictionaryRef matching = io->service_matching("IOAccelerator");
        if (matching == NULL) {
            return AMI_OK;
        }

        ami_io_iterator_t iter = MACH_PORT_NULL;
        kern_return_t kr = io->get_matching_services(MACH_PORT_NULL, matching, &iter);
        if (kr != KERN_SUCCESS || iter == MACH_PORT_NULL) {
            return AMI_OK;
        }

        ami_io_object_t service = MACH_PORT_NULL;
        while ((service = io->iterator_next(iter)) != MACH_PORT_NULL) {
            CFMutableDictionaryRef props = NULL;
            if (io->entry_create_props(service, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS &&
                props != NULL) {
                CFTypeRef perf_raw = CFDictionaryGetValue(props, CFSTR("PerformanceStatistics"));
                if (perf_raw != NULL && CFGetTypeID(perf_raw) == CFDictionaryGetTypeID()) {
                    CFDictionaryRef perf = (CFDictionaryRef)perf_raw;
                    uint64_t v = 0;

                    out_sample->available = 1u;
                    if (out_sample->accelerator_class[0] == '\0') {
                        ami_cf_dict_read_cstring(props, CFSTR("IOClass"),
                                                 out_sample->accelerator_class,
                                                 sizeof(out_sample->accelerator_class));
                        if (out_sample->accelerator_class[0] == '\0') {
                            ami_cf_dict_read_cstring(props, CFSTR("MetalPluginClassName"),
                                                     out_sample->accelerator_class,
                                                     sizeof(out_sample->accelerator_class));
                        }
                    }

                    if (ami_cf_dict_read_u64(perf, CFSTR("Device Utilization %"), &v)) {
                        uint32_t pct = (v > 100u) ? 100u : (uint32_t)v;
                        if (pct > out_sample->device_utilization_pct) {
                            out_sample->device_utilization_pct = pct;
                        }
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("Renderer Utilization %"), &v)) {
                        uint32_t pct = (v > 100u) ? 100u : (uint32_t)v;
                        if (pct > out_sample->renderer_utilization_pct) {
                            out_sample->renderer_utilization_pct = pct;
                        }
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("Tiler Utilization %"), &v)) {
                        uint32_t pct = (v > 100u) ? 100u : (uint32_t)v;
                        if (pct > out_sample->tiler_utilization_pct) {
                            out_sample->tiler_utilization_pct = pct;
                        }
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("In use system memory"), &v)) {
                        out_sample->in_use_system_memory += v;
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("Alloc system memory"), &v)) {
                        out_sample->alloc_system_memory += v;
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("In use system memory (driver)"), &v)) {
                        out_sample->driver_in_use_memory += v;
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("TiledSceneBytes"), &v)) {
                        out_sample->tiled_scene_bytes += v;
                    }
                    if (ami_cf_dict_read_u64(perf, CFSTR("recoveryCount"), &v)) {
                        out_sample->recovery_count += (uint32_t)v;
                    }
                }
                CFRelease(props);
            }
            io->object_release(service);
        }
        io->object_release(iter);
    }

    return AMI_OK;
}

ami_status ami_telemetry_sample_nvme(ami_telemetry_nvme_sample *out_sample) {
    if (out_sample == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_sample, 0, sizeof(*out_sample));

    @autoreleasepool {
        const ami_iokit_api *io = ami_get_iokit();
        if (!io->ready) {
            return AMI_OK;
        }

        CFMutableDictionaryRef matching = io->service_matching("IOBlockStorageDriver");
        if (matching == NULL) {
            return AMI_OK;
        }

        ami_io_iterator_t iter = MACH_PORT_NULL;
        kern_return_t kr = io->get_matching_services(MACH_PORT_NULL, matching, &iter);
        if (kr != KERN_SUCCESS || iter == MACH_PORT_NULL) {
            return AMI_OK;
        }

        ami_io_object_t service = MACH_PORT_NULL;
        while ((service = io->iterator_next(iter)) != MACH_PORT_NULL) {
            CFMutableDictionaryRef props = NULL;
            if (io->entry_create_props(service, &props, kCFAllocatorDefault, 0) == KERN_SUCCESS &&
                props != NULL) {
                CFTypeRef stats_raw = CFDictionaryGetValue(props, CFSTR("Statistics"));
                if (stats_raw != NULL && CFGetTypeID(stats_raw) == CFDictionaryGetTypeID()) {
                    CFDictionaryRef stats = (CFDictionaryRef)stats_raw;
                    uint64_t v = 0;
                    out_sample->available = 1u;
                    out_sample->driver_count += 1u;

                    if (ami_cf_dict_read_u64(stats, CFSTR("Bytes (Read)"), &v)) {
                        out_sample->bytes_read += v;
                    }
                    if (ami_cf_dict_read_u64(stats, CFSTR("Bytes (Write)"), &v)) {
                        out_sample->bytes_written += v;
                    }
                    if (ami_cf_dict_read_u64(stats, CFSTR("Operations (Read)"), &v)) {
                        out_sample->ops_read += v;
                    }
                    if (ami_cf_dict_read_u64(stats, CFSTR("Operations (Write)"), &v)) {
                        out_sample->ops_written += v;
                    }
                    if (ami_cf_dict_read_u64(stats, CFSTR("Total Time (Read)"), &v)) {
                        out_sample->read_time_ns += v;
                    }
                    if (ami_cf_dict_read_u64(stats, CFSTR("Total Time (Write)"), &v)) {
                        out_sample->write_time_ns += v;
                    }
                }
                CFRelease(props);
            }
            io->object_release(service);
        }
        io->object_release(iter);
    }

    return AMI_OK;
}

uint32_t ami_tty_isatty(int32_t fd) {
    if (fd < 0) {
        return 0u;
    }
    return isatty(fd) != 0 ? 1u : 0u;
}

ami_status ami_tty_get_winsize(int32_t fd, uint16_t *out_cols, uint16_t *out_rows) {
    if (fd < 0 || out_cols == NULL || out_rows == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    *out_cols = 0;
    *out_rows = 0;
    struct winsize ws;
    memset(&ws, 0, sizeof(ws));
    if (ioctl(fd, TIOCGWINSZ, &ws) != 0 || ws.ws_col == 0 || ws.ws_row == 0) {
        return AMI_ERR_TTY;
    }
    *out_cols = ws.ws_col;
    *out_rows = ws.ws_row;
    return AMI_OK;
}

ami_status ami_tty_enter_raw(int32_t fd, ami_tty_raw_guard *out_guard) {
    if (fd < 0 || out_guard == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    memset(out_guard, 0, sizeof(*out_guard));
    struct termios orig;
    if (tcgetattr(fd, &orig) != 0) {
        return AMI_ERR_TTY;
    }
    memcpy(out_guard->opaque, &orig, sizeof(orig));
    out_guard->fd = fd;

    struct termios raw = orig;
    raw.c_iflag &= ~(tcflag_t)(BRKINT | ICRNL | INPCK | ISTRIP | IXON);
    raw.c_oflag &= ~(tcflag_t)(OPOST);
    raw.c_cflag |= (tcflag_t)(CS8);
    raw.c_lflag &= ~(tcflag_t)(ECHO | ICANON | IEXTEN | ISIG);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;

    if (tcsetattr(fd, TCSAFLUSH, &raw) != 0) {
        return AMI_ERR_TTY;
    }
    out_guard->active = 1u;
    return AMI_OK;
}

ami_status ami_tty_leave_raw(ami_tty_raw_guard *guard) {
    if (guard == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    if (!guard->active) {
        return AMI_OK;
    }
    struct termios orig;
    memcpy(&orig, guard->opaque, sizeof(orig));
    guard->active = 0u;
    if (tcsetattr(guard->fd, TCSAFLUSH, &orig) != 0) {
        return AMI_ERR_TTY;
    }
    return AMI_OK;
}

// ============================================================================
// Native TLS 1.2/1.3 Listener Bridge (Network.framework + Security.framework)
// ============================================================================

static void ami_secure_wipe(void *ptr, size_t len) {
    if (ptr != NULL && len > 0) {
        (void)memset_s(ptr, len, 0, len);
    }
}

static void ami_secure_wipe_mutable_data(NSMutableData *data) {
    if (data != nil && data.length > 0) {
        ami_secure_wipe(data.mutableBytes, data.length);
    }
}

// Minimal RFC 1321 MD5 for legacy OpenSSL PBES1 (pbeWithMD5AndDES-CBC) and
// legacy PEM EVP_BytesToKey without calling deprecated CommonCrypto CC_MD5.
static void ami_md5_transform(uint32_t state[4], const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au,
        0xa8304613u, 0xfd469501u, 0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu,
        0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u, 0xf61e2562u, 0xc040b340u,
        0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
        0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u,
        0x676f02d9u, 0x8d2a4c8au, 0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu,
        0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u, 0x289b7ec6u, 0xeaa127fau,
        0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
        0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u,
        0xffeff47du, 0x85845dd1u, 0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u,
        0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u,
    };
    static const uint8_t s[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    };
    uint32_t m[16];
    for (size_t i = 0; i < 16; i++) {
        size_t j = i * 4;
        m[i] = ((uint32_t)block[j]) | ((uint32_t)block[j + 1] << 8) |
               ((uint32_t)block[j + 2] << 16) | ((uint32_t)block[j + 3] << 24);
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    for (uint32_t i = 0; i < 64; i++) {
        uint32_t f = 0, g = 0;
        if (i < 16) {
            f = (b & c) | ((~b) & d);
            g = i;
        } else if (i < 32) {
            f = (d & b) | ((~d) & c);
            g = (5u * i + 1u) % 16u;
        } else if (i < 48) {
            f = b ^ c ^ d;
            g = (3u * i + 5u) % 16u;
        } else {
            f = c ^ (b | (~d));
            g = (7u * i) % 16u;
        }
        uint32_t tmp = d;
        d = c;
        c = b;
        uint32_t x = a + f + k[i] + m[g];
        b = b + ((x << s[i]) | (x >> (32u - s[i])));
        a = tmp;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    ami_secure_wipe(m, sizeof(m));
}

static void ami_md5(const uint8_t *data, size_t len, uint8_t out16[16]) {
    uint32_t state[4] = {0x67452301u, 0xefcdab89u, 0x98badcfeu, 0x10325476u};
    uint8_t buf[64];
    size_t pos = 0;
    while (len - pos >= 64) {
        ami_md5_transform(state, data + pos);
        pos += 64;
    }
    size_t rem = len - pos;
    memset(buf, 0, sizeof(buf));
    if (rem > 0) {
        memcpy(buf, data + pos, rem);
    }
    buf[rem++] = 0x80u;
    if (rem > 56) {
        ami_md5_transform(state, buf);
        memset(buf, 0, sizeof(buf));
    }
    uint64_t bit_len = (uint64_t)len * 8u;
    for (size_t i = 0; i < 8; i++) {
        buf[56 + i] = (uint8_t)((bit_len >> (i * 8u)) & 0xffu);
    }
    ami_md5_transform(state, buf);
    for (size_t i = 0; i < 4; i++) {
        out16[i * 4 + 0] = (uint8_t)(state[i] & 0xffu);
        out16[i * 4 + 1] = (uint8_t)((state[i] >> 8) & 0xffu);
        out16[i * 4 + 2] = (uint8_t)((state[i] >> 16) & 0xffu);
        out16[i * 4 + 3] = (uint8_t)((state[i] >> 24) & 0xffu);
    }
    ami_secure_wipe(state, sizeof(state));
    ami_secure_wipe(buf, sizeof(buf));
}

// ASN.1 DER reader (strict bounded lengths; rejects indefinite encoding).
typedef struct ami_der_elem {
    uint8_t tag;
    const uint8_t *val;
    size_t len;
    const uint8_t *next;
} ami_der_elem;

static bool ami_der_read(const uint8_t *cur, const uint8_t *end, ami_der_elem *out) {
    if (cur == NULL || end == NULL || cur >= end || out == NULL) {
        return false;
    }
    uint8_t tag = *cur++;
    if ((tag & 0x1fu) == 0x1fu) {
        return false; // high-tag-number form not expected in standard PKCS/X.509 keys
    }
    if (cur >= end) {
        return false;
    }
    uint8_t l0 = *cur++;
    size_t len = 0;
    if ((l0 & 0x80u) == 0u) {
        len = l0;
    } else {
        uint8_t nbytes = (uint8_t)(l0 & 0x7fu);
        if (nbytes == 0u || nbytes > sizeof(size_t) || (size_t)(end - cur) < nbytes) {
            return false;
        }
        if (*cur == 0u) {
            return false; // non-minimal length encoding
        }
        for (uint8_t i = 0; i < nbytes; i++) {
            len = (len << 8) | (size_t)(*cur++);
        }
        if (len < 128u) {
            return false; // non-minimal length encoding
        }
    }
    if ((size_t)(end - cur) < len) {
        return false;
    }
    out->tag = tag;
    out->val = cur;
    out->len = len;
    out->next = cur + len;
    return true;
}

static bool ami_der_read_uint32(const ami_der_elem *elem, uint32_t *out_val) {
    if (elem->tag != 0x02u || elem->len == 0 || elem->len > 5) {
        return false;
    }
    if ((elem->val[0] & 0x80u) != 0u) {
        return false; // negative INTEGER
    }
    uint64_t v = 0;
    for (size_t i = 0; i < elem->len; i++) {
        v = (v << 8) | elem->val[i];
    }
    if (v > UINT32_MAX) {
        return false;
    }
    *out_val = (uint32_t)v;
    return true;
}

static bool ami_oid_eq(const ami_der_elem *elem, const uint8_t *oid, size_t oid_len) {
    return elem->tag == 0x06u && elem->len == oid_len && memcmp(elem->val, oid, oid_len) == 0;
}

static const uint8_t *ami_memmem(const uint8_t *hay, size_t hay_len, const char *needle) {
    size_t nlen = strlen(needle);
    if (nlen == 0 || hay == NULL || hay_len < nlen) {
        return NULL;
    }
    for (size_t i = 0; i <= hay_len - nlen; i++) {
        if (memcmp(hay + i, needle, nlen) == 0) {
            return hay + i;
        }
    }
    return NULL;
}

static int ami_b64_val(uint8_t ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

static NSMutableData *ami_base64_decode_pem(const uint8_t *src, size_t len) {
    if (src == NULL || len == 0) {
        return nil;
    }
    NSMutableData *out = [NSMutableData dataWithCapacity:(len * 3u) / 4u + 4u];
    uint32_t accum = 0;
    int bits = 0;
    int pad = 0;
    bool seen_pad = false;
    for (size_t i = 0; i < len; i++) {
        uint8_t c = src[i];
        if (c == '\r' || c == '\n' || c == ' ' || c == '\t') {
            continue;
        }
        if (c == '=') {
            seen_pad = true;
            pad++;
            if (pad > 2) {
                ami_secure_wipe_mutable_data(out);
                return nil;
            }
            accum <<= 6;
            bits += 6;
            if (bits == 24) {
                uint8_t b0 = (uint8_t)((accum >> 16) & 0xffu);
                [out appendBytes:&b0 length:1];
                if (pad == 1) {
                    uint8_t b1 = (uint8_t)((accum >> 8) & 0xffu);
                    [out appendBytes:&b1 length:1];
                }
                accum = 0;
                bits = 0;
                pad = 0;
            }
            continue;
        }
        if (seen_pad) {
            ami_secure_wipe_mutable_data(out);
            return nil;
        }
        int v = ami_b64_val(c);
        if (v < 0) {
            ami_secure_wipe_mutable_data(out);
            return nil;
        }
        accum = (accum << 6) | (uint32_t)v;
        bits += 6;
        if (bits == 24) {
            uint8_t bytes[3] = {
                (uint8_t)((accum >> 16) & 0xffu),
                (uint8_t)((accum >> 8) & 0xffu),
                (uint8_t)(accum & 0xffu),
            };
            [out appendBytes:bytes length:3];
            ami_secure_wipe(bytes, sizeof(bytes));
            accum = 0;
            bits = 0;
        }
    }
    if (bits != 0 || out.length == 0) {
        ami_secure_wipe_mutable_data(out);
        return nil;
    }
    return out;
}

typedef struct ami_pem_block {
    char label[64];
    const uint8_t *headers;
    size_t headers_len;
    const uint8_t *body;
    size_t body_len;
    size_t next_offset;
} ami_pem_block;

static bool ami_pem_next_block(const uint8_t *buf, size_t len, size_t offset, ami_pem_block *out) {
    if (buf == NULL || offset >= len || out == NULL) {
        return false;
    }
    memset(out, 0, sizeof(*out));
    const char *begin_tag = "-----BEGIN ";
    const uint8_t *p = ami_memmem(buf + offset, len - offset, begin_tag);
    while (p != NULL) {
        // Require start of buffer or preceded by newline
        if (p == buf || p[-1] == '\n' || p[-1] == '\r') {
            break;
        }
        size_t adv = (size_t)(p - buf) + 1u;
        p = ami_memmem(buf + adv, len - adv, begin_tag);
    }
    if (p == NULL) {
        return false;
    }
    const uint8_t *label_start = p + strlen(begin_tag);
    const uint8_t *label_end = ami_memmem(label_start, (size_t)(buf + len - label_start), "-----");
    if (label_end == NULL) {
        return false;
    }
    size_t label_len = (size_t)(label_end - label_start);
    if (label_len == 0 || label_len >= sizeof(out->label)) {
        return false;
    }
    memcpy(out->label, label_start, label_len);
    out->label[label_len] = '\0';

    const uint8_t *after_begin = label_end + 5;
    while (after_begin < buf + len && (*after_begin == ' ' || *after_begin == '\t')) {
        after_begin++;
    }
    if (after_begin >= buf + len || (*after_begin != '\n' && *after_begin != '\r')) {
        return false;
    }
    if (*after_begin == '\r' && after_begin + 1 < buf + len && after_begin[1] == '\n') {
        after_begin += 2;
    } else {
        after_begin += 1;
    }

    char end_marker[84];
    snprintf(end_marker, sizeof(end_marker), "-----END %s-----", out->label);
    const uint8_t *end_pos = ami_memmem(after_begin, (size_t)(buf + len - after_begin), end_marker);
    while (end_pos != NULL) {
        if (end_pos == after_begin || end_pos[-1] == '\n' || end_pos[-1] == '\r') {
            break;
        }
        size_t adv = (size_t)(end_pos - buf) + 1u;
        end_pos = ami_memmem(buf + adv, len - adv, end_marker);
    }
    if (end_pos == NULL) {
        return false;
    }

    // Check for RFC 1421 / legacy OpenSSL headers before a blank line.
    const uint8_t *body_start = after_begin;
    const uint8_t *first_nl = memchr(after_begin, '\n', (size_t)(end_pos - after_begin));
    if (first_nl != NULL) {
        const uint8_t *colon = memchr(after_begin, ':', (size_t)(first_nl - after_begin));
        if (colon != NULL) {
            const uint8_t *blank_lf = ami_memmem(after_begin, (size_t)(end_pos - after_begin), "\n\n");
            const uint8_t *blank_crlf = ami_memmem(after_begin, (size_t)(end_pos - after_begin), "\r\n\r\n");
            const uint8_t *sep = NULL;
            size_t sep_len = 0;
            if (blank_lf != NULL && (blank_crlf == NULL || blank_lf < blank_crlf)) {
                sep = blank_lf;
                sep_len = 2;
            } else if (blank_crlf != NULL) {
                sep = blank_crlf;
                sep_len = 4;
            }
            if (sep != NULL) {
                out->headers = after_begin;
                out->headers_len = (size_t)(sep - after_begin);
                body_start = sep + sep_len;
            }
        }
    }

    out->body = body_start;
    out->body_len = (size_t)(end_pos - body_start);
    out->next_offset = (size_t)(end_pos + strlen(end_marker) - buf);
    return true;
}

static bool ami_hex_decode(const char *hex, size_t hex_len, uint8_t *out, size_t out_len) {
    if (hex_len != out_len * 2u) {
        return false;
    }
    for (size_t i = 0; i < out_len; i++) {
        uint8_t hi = (uint8_t)hex[i * 2u];
        uint8_t lo = (uint8_t)hex[i * 2u + 1u];
        int hv = (hi >= '0' && hi <= '9') ? (hi - '0')
               : (hi >= 'A' && hi <= 'F') ? (hi - 'A' + 10)
               : (hi >= 'a' && hi <= 'f') ? (hi - 'a' + 10)
                                          : -1;
        int lv = (lo >= '0' && lo <= '9') ? (lo - '0')
               : (lo >= 'A' && lo <= 'F') ? (lo - 'A' + 10)
               : (lo >= 'a' && lo <= 'f') ? (lo - 'a' + 10)
                                          : -1;
        if (hv < 0 || lv < 0) {
            return false;
        }
        out[i] = (uint8_t)((hv << 4) | lv);
    }
    return true;
}

static ami_status ami_cc_decrypt_cbc(CCAlgorithm alg, const uint8_t *key, size_t key_len,
                                     const uint8_t *iv, const uint8_t *ct, size_t ct_len,
                                     NSMutableData *__strong *out_plain) {
    if (ct == NULL || ct_len == 0 || out_plain == NULL) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    NSMutableData *plain = [NSMutableData dataWithLength:ct_len + 32u];
    size_t written = 0;
    CCCryptorStatus cst = CCCrypt(kCCDecrypt, alg, kCCOptionPKCS7Padding, key, key_len, iv,
                                  ct, ct_len, plain.mutableBytes, plain.length, &written);
    if (cst != kCCSuccess || written == 0) {
        ami_secure_wipe_mutable_data(plain);
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    [plain setLength:written];
    *out_plain = plain;
    return AMI_OK;
}

static void ami_pkcs12_kdf_sha1(const uint8_t *pass, size_t pass_len,
                                const uint8_t *salt, size_t salt_len,
                                uint32_t iter, uint8_t id,
                                uint8_t *out, size_t out_len) {
    // RFC 7292 Appendix B with SHA-1 (u = 20, v = 64)
    const size_t u = CC_SHA1_DIGEST_LENGTH;
    const size_t v = 64u;
    size_t bmp_len = (pass_len > 0) ? (pass_len + 1u) * 2u : 0u;
    uint8_t *bmp = bmp_len > 0 ? (uint8_t *)malloc(bmp_len) : NULL;
    if (bmp != NULL) {
        for (size_t i = 0; i < pass_len; i++) {
            bmp[i * 2u] = 0u;
            bmp[i * 2u + 1u] = pass[i];
        }
        bmp[pass_len * 2u] = 0u;
        bmp[pass_len * 2u + 1u] = 0u;
    }

    uint8_t buf_d[64];
    memset(buf_d, id, v);
    size_t s_len = salt_len > 0 ? ((salt_len + v - 1u) / v) * v : 0u;
    size_t p_len = bmp_len > 0 ? ((bmp_len + v - 1u) / v) * v : 0u;
    size_t buf_i_len = s_len + p_len;
    uint8_t *buf_i = buf_i_len > 0 ? (uint8_t *)malloc(buf_i_len) : NULL;
    if (buf_i != NULL) {
        for (size_t i = 0; i < s_len; i++) {
            buf_i[i] = salt[i % salt_len];
        }
        for (size_t i = 0; i < p_len; i++) {
            buf_i[s_len + i] = bmp[i % bmp_len];
        }
    }

    uint8_t buf_a[CC_SHA1_DIGEST_LENGTH];
    uint8_t buf_b[64];
    size_t pos = 0;
    while (pos < out_len) {
        CC_SHA1_CTX ctx;
        CC_SHA1_Init(&ctx);
        CC_SHA1_Update(&ctx, buf_d, (CC_LONG)v);
        if (buf_i != NULL && buf_i_len > 0) {
            CC_SHA1_Update(&ctx, buf_i, (CC_LONG)buf_i_len);
        }
        CC_SHA1_Final(buf_a, &ctx);
        for (uint32_t r = 1; r < iter; r++) {
            CC_SHA1(buf_a, (CC_LONG)u, buf_a);
        }
        size_t take = (out_len - pos < u) ? (out_len - pos) : u;
        memcpy(out + pos, buf_a, take);
        pos += take;
        if (pos >= out_len) {
            break;
        }
        for (size_t i = 0; i < v; i++) {
            buf_b[i] = buf_a[i % u];
        }
        for (size_t k = 0; k < buf_i_len; k += v) {
            uint16_t carry = 1u;
            for (size_t j = v; j > 0; j--) {
                size_t idx = k + j - 1u;
                uint16_t sum = (uint16_t)buf_i[idx] + (uint16_t)buf_b[j - 1u] + carry;
                buf_i[idx] = (uint8_t)(sum & 0xffu);
                carry = sum >> 8;
            }
        }
    }
    ami_secure_wipe(buf_a, sizeof(buf_a));
    ami_secure_wipe(buf_b, sizeof(buf_b));
    ami_secure_wipe(buf_d, sizeof(buf_d));
    if (buf_i != NULL) {
        ami_secure_wipe(buf_i, buf_i_len);
        free(buf_i);
    }
    if (bmp != NULL) {
        ami_secure_wipe(bmp, bmp_len);
        free(bmp);
    }
}

static ami_status ami_decrypt_legacy_pem(const uint8_t *headers, size_t headers_len,
                                         const uint8_t *ct, size_t ct_len,
                                         const uint8_t *pass, size_t pass_len,
                                         NSMutableData *__strong *out_plain) {
    if (pass == NULL || pass_len == 0) {
        return AMI_STATUS_TLS_PASSPHRASE_REQUIRED;
    }
    if (headers == NULL || headers_len == 0) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    const uint8_t *dek = ami_memmem(headers, headers_len, "DEK-Info:");
    if (dek == NULL) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    const uint8_t *p = dek + 9;
    const uint8_t *hend = headers + headers_len;
    while (p < hend && (*p == ' ' || *p == '\t')) p++;
    const uint8_t *comma = memchr(p, ',', (size_t)(hend - p));
    if (comma == NULL) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    size_t alg_len = (size_t)(comma - p);
    const uint8_t *iv_hex = comma + 1;
    const uint8_t *iv_end = iv_hex;
    while (iv_end < hend && *iv_end != '\r' && *iv_end != '\n' && *iv_end != ' ' && *iv_end != '\t') {
        iv_end++;
    }
    CCAlgorithm alg = kCCAlgorithmAES;
    size_t key_len = 0;
    size_t iv_len = 0;
    if (alg_len == 11 && memcmp(p, "AES-128-CBC", 11) == 0) {
        alg = kCCAlgorithmAES; key_len = 16; iv_len = 16;
    } else if (alg_len == 11 && memcmp(p, "AES-192-CBC", 11) == 0) {
        alg = kCCAlgorithmAES; key_len = 24; iv_len = 16;
    } else if (alg_len == 11 && memcmp(p, "AES-256-CBC", 11) == 0) {
        alg = kCCAlgorithmAES; key_len = 32; iv_len = 16;
    } else if (alg_len == 12 && memcmp(p, "DES-EDE3-CBC", 12) == 0) {
        alg = kCCAlgorithm3DES; key_len = 24; iv_len = 8;
    } else {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    uint8_t iv[16];
    memset(iv, 0, sizeof(iv));
    if (!ami_hex_decode((const char *)iv_hex, (size_t)(iv_end - iv_hex), iv, iv_len)) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    // EVP_BytesToKey(MD5, salt = iv[0..8], count = 1)
    uint8_t key[32];
    memset(key, 0, sizeof(key));
    uint8_t digest[16];
    size_t have = 0;
    size_t scratch_cap = 16u + pass_len + 8u;
    uint8_t *scratch = (uint8_t *)malloc(scratch_cap);
    if (scratch == NULL) {
        return AMI_ERR_OUT_OF_MEMORY;
    }
    bool first = true;
    while (have < key_len) {
        size_t pos = 0;
        if (!first) {
            memcpy(scratch, digest, 16u);
            pos = 16u;
        }
        memcpy(scratch + pos, pass, pass_len);
        pos += pass_len;
        memcpy(scratch + pos, iv, 8u);
        pos += 8u;
        ami_md5(scratch, pos, digest);
        first = false;
        size_t take = (key_len - have < 16u) ? (key_len - have) : 16u;
        memcpy(key + have, digest, take);
        have += take;
    }
    ami_secure_wipe(scratch, scratch_cap);
    free(scratch);
    ami_secure_wipe(digest, sizeof(digest));

    ami_status st = ami_cc_decrypt_cbc(alg, key, key_len, iv, ct, ct_len, out_plain);
    ami_secure_wipe(key, sizeof(key));
    ami_secure_wipe(iv, sizeof(iv));
    return st;
}

static ami_status ami_decrypt_pkcs8_epki(const uint8_t *der, size_t der_len,
                                         const uint8_t *pass, size_t pass_len,
                                         NSMutableData *__strong *out_plain) {
    static const uint8_t oid_pbes2[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x05u, 0x0du};
    static const uint8_t oid_pbkdf2[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x05u, 0x0cu};
    static const uint8_t oid_hmac_sha1[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x02u, 0x07u};
    static const uint8_t oid_hmac_sha224[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x02u, 0x08u};
    static const uint8_t oid_hmac_sha256[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x02u, 0x09u};
    static const uint8_t oid_hmac_sha384[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x02u, 0x0au};
    static const uint8_t oid_hmac_sha512[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x02u, 0x0bu};
    static const uint8_t oid_aes128_cbc[] = {0x60u, 0x86u, 0x48u, 0x01u, 0x65u, 0x03u, 0x04u, 0x01u, 0x02u};
    static const uint8_t oid_aes192_cbc[] = {0x60u, 0x86u, 0x48u, 0x01u, 0x65u, 0x03u, 0x04u, 0x01u, 0x16u};
    static const uint8_t oid_aes256_cbc[] = {0x60u, 0x86u, 0x48u, 0x01u, 0x65u, 0x03u, 0x04u, 0x01u, 0x2au};
    static const uint8_t oid_des_ede3_cbc[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x03u, 0x07u};
    static const uint8_t oid_pbe_md5_des[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x05u, 0x03u};
    static const uint8_t oid_pbe_sha1_des[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x05u, 0x0au};
    static const uint8_t oid_p12_3des[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x0cu, 0x01u, 0x03u};
    static const uint8_t oid_p12_2des[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x0cu, 0x01u, 0x04u};

    if (pass == NULL || pass_len == 0) {
        return AMI_STATUS_TLS_PASSPHRASE_REQUIRED;
    }

    ami_der_elem top, alg_seq, enc_data;
    if (!ami_der_read(der, der + der_len, &top) || top.tag != 0x30u || top.next != der + der_len) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    if (!ami_der_read(top.val, top.next, &alg_seq) || alg_seq.tag != 0x30u) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    if (!ami_der_read(alg_seq.next, top.next, &enc_data) || enc_data.tag != 0x04u || enc_data.next != top.next) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    ami_der_elem alg_oid, alg_params;
    if (!ami_der_read(alg_seq.val, alg_seq.next, &alg_oid) || alg_oid.tag != 0x06u) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    if (!ami_der_read(alg_oid.next, alg_seq.next, &alg_params) || alg_params.next != alg_seq.next) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    if (ami_oid_eq(&alg_oid, oid_pbes2, sizeof(oid_pbes2))) {
        if (alg_params.tag != 0x30u) return AMI_STATUS_TLS_IDENTITY_INVALID;
        ami_der_elem kdf_seq, enc_seq;
        if (!ami_der_read(alg_params.val, alg_params.next, &kdf_seq) || kdf_seq.tag != 0x30u) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        if (!ami_der_read(kdf_seq.next, alg_params.next, &enc_seq) || enc_seq.tag != 0x30u ||
            enc_seq.next != alg_params.next) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }

        ami_der_elem kdf_oid, kdf_params;
        if (!ami_der_read(kdf_seq.val, kdf_seq.next, &kdf_oid) ||
            !ami_oid_eq(&kdf_oid, oid_pbkdf2, sizeof(oid_pbkdf2))) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        if (!ami_der_read(kdf_oid.next, kdf_seq.next, &kdf_params) ||
            kdf_params.tag != 0x30u || kdf_params.next != kdf_seq.next) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }

        ami_der_elem salt_el, iter_el, opt_el;
        if (!ami_der_read(kdf_params.val, kdf_params.next, &salt_el) || salt_el.tag != 0x04u) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        if (!ami_der_read(salt_el.next, kdf_params.next, &iter_el)) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        uint32_t iter = 0;
        if (!ami_der_read_uint32(&iter_el, &iter) || iter == 0 || iter > 10000000u) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }

        uint32_t explicit_keylen = 0;
        bool has_explicit_keylen = false;
        CCPseudoRandomAlgorithm prf = kCCPRFHmacAlgSHA1;
        const uint8_t *cur = iter_el.next;
        if (cur < kdf_params.next) {
            if (!ami_der_read(cur, kdf_params.next, &opt_el)) {
                return AMI_STATUS_TLS_IDENTITY_INVALID;
            }
            if (opt_el.tag == 0x02u) {
                if (!ami_der_read_uint32(&opt_el, &explicit_keylen) || explicit_keylen == 0) {
                    return AMI_STATUS_TLS_IDENTITY_INVALID;
                }
                has_explicit_keylen = true;
                cur = opt_el.next;
                if (cur < kdf_params.next && !ami_der_read(cur, kdf_params.next, &opt_el)) {
                    return AMI_STATUS_TLS_IDENTITY_INVALID;
                }
            }
            if (cur < kdf_params.next) {
                if (opt_el.tag != 0x30u || opt_el.next != kdf_params.next) {
                    return AMI_STATUS_TLS_IDENTITY_INVALID;
                }
                ami_der_elem prf_oid;
                if (!ami_der_read(opt_el.val, opt_el.next, &prf_oid)) {
                    return AMI_STATUS_TLS_IDENTITY_INVALID;
                }
                if (ami_oid_eq(&prf_oid, oid_hmac_sha1, sizeof(oid_hmac_sha1))) {
                    prf = kCCPRFHmacAlgSHA1;
                } else if (ami_oid_eq(&prf_oid, oid_hmac_sha224, sizeof(oid_hmac_sha224))) {
                    prf = kCCPRFHmacAlgSHA224;
                } else if (ami_oid_eq(&prf_oid, oid_hmac_sha256, sizeof(oid_hmac_sha256))) {
                    prf = kCCPRFHmacAlgSHA256;
                } else if (ami_oid_eq(&prf_oid, oid_hmac_sha384, sizeof(oid_hmac_sha384))) {
                    prf = kCCPRFHmacAlgSHA384;
                } else if (ami_oid_eq(&prf_oid, oid_hmac_sha512, sizeof(oid_hmac_sha512))) {
                    prf = kCCPRFHmacAlgSHA512;
                } else {
                    return AMI_STATUS_TLS_IDENTITY_INVALID;
                }
            }
        }

        ami_der_elem enc_oid, iv_el;
        if (!ami_der_read(enc_seq.val, enc_seq.next, &enc_oid) ||
            !ami_der_read(enc_oid.next, enc_seq.next, &iv_el) ||
            iv_el.tag != 0x04u || iv_el.next != enc_seq.next) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        CCAlgorithm alg = kCCAlgorithmAES;
        size_t key_len = 0;
        size_t iv_len = 0;
        if (ami_oid_eq(&enc_oid, oid_aes128_cbc, sizeof(oid_aes128_cbc))) {
            alg = kCCAlgorithmAES; key_len = 16; iv_len = 16;
        } else if (ami_oid_eq(&enc_oid, oid_aes192_cbc, sizeof(oid_aes192_cbc))) {
            alg = kCCAlgorithmAES; key_len = 24; iv_len = 16;
        } else if (ami_oid_eq(&enc_oid, oid_aes256_cbc, sizeof(oid_aes256_cbc))) {
            alg = kCCAlgorithmAES; key_len = 32; iv_len = 16;
        } else if (ami_oid_eq(&enc_oid, oid_des_ede3_cbc, sizeof(oid_des_ede3_cbc))) {
            alg = kCCAlgorithm3DES; key_len = 24; iv_len = 8;
        } else {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        if (iv_el.len != iv_len || (has_explicit_keylen && explicit_keylen != key_len)) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }

        uint8_t derived[32];
        memset(derived, 0, sizeof(derived));
        if (CCKeyDerivationPBKDF(kCCPBKDF2, (const char *)pass, pass_len,
                                 salt_el.val, salt_el.len, prf, (unsigned)iter,
                                 derived, key_len) != kCCSuccess) {
            ami_secure_wipe(derived, sizeof(derived));
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        ami_status st = ami_cc_decrypt_cbc(alg, derived, key_len, iv_el.val,
                                           enc_data.val, enc_data.len, out_plain);
        ami_secure_wipe(derived, sizeof(derived));
        return st;
    }

    // PBES1 (PKCS#5 v1.5) or PKCS#12 PBE
    if (alg_params.tag != 0x30u) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    ami_der_elem salt_el, iter_el;
    if (!ami_der_read(alg_params.val, alg_params.next, &salt_el) || salt_el.tag != 0x04u ||
        !ami_der_read(salt_el.next, alg_params.next, &iter_el) || iter_el.next != alg_params.next) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    uint32_t iter = 0;
    if (!ami_der_read_uint32(&iter_el, &iter) || iter == 0 || iter > 10000000u) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    if (ami_oid_eq(&alg_oid, oid_pbe_md5_des, sizeof(oid_pbe_md5_des)) ||
        ami_oid_eq(&alg_oid, oid_pbe_sha1_des, sizeof(oid_pbe_sha1_des))) {
        if (salt_el.len != 8u) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        bool use_sha1 = ami_oid_eq(&alg_oid, oid_pbe_sha1_des, sizeof(oid_pbe_sha1_des));
        uint8_t digest[CC_SHA1_DIGEST_LENGTH];
        memset(digest, 0, sizeof(digest));
        size_t in_len = pass_len + salt_el.len;
        uint8_t *in_buf = (uint8_t *)malloc(in_len);
        if (in_buf == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        memcpy(in_buf, pass, pass_len);
        memcpy(in_buf + pass_len, salt_el.val, salt_el.len);
        if (use_sha1) {
            CC_SHA1(in_buf, (CC_LONG)in_len, digest);
            for (uint32_t r = 1; r < iter; r++) {
                CC_SHA1(digest, CC_SHA1_DIGEST_LENGTH, digest);
            }
        } else {
            ami_md5(in_buf, in_len, digest);
            for (uint32_t r = 1; r < iter; r++) {
                ami_md5(digest, 16u, digest);
            }
        }
        ami_secure_wipe(in_buf, in_len);
        free(in_buf);

        ami_status st = ami_cc_decrypt_cbc(kCCAlgorithmDES, digest, 8u, digest + 8u,
                                           enc_data.val, enc_data.len, out_plain);
        ami_secure_wipe(digest, sizeof(digest));
        return st;
    }

    if (ami_oid_eq(&alg_oid, oid_p12_3des, sizeof(oid_p12_3des)) ||
        ami_oid_eq(&alg_oid, oid_p12_2des, sizeof(oid_p12_2des))) {
        bool is_3key = ami_oid_eq(&alg_oid, oid_p12_3des, sizeof(oid_p12_3des));
        uint8_t key[24];
        uint8_t iv[8];
        ami_pkcs12_kdf_sha1(pass, pass_len, salt_el.val, salt_el.len, iter, 1u, key, is_3key ? 24u : 16u);
        if (!is_3key) {
            memcpy(key + 16, key, 8u);
        }
        ami_pkcs12_kdf_sha1(pass, pass_len, salt_el.val, salt_el.len, iter, 2u, iv, 8u);
        ami_status st = ami_cc_decrypt_cbc(kCCAlgorithm3DES, key, 24u, iv,
                                           enc_data.val, enc_data.len, out_plain);
        ami_secure_wipe(key, sizeof(key));
        ami_secure_wipe(iv, sizeof(iv));
        return st;
    }

    return AMI_STATUS_TLS_IDENTITY_INVALID;
}

static size_t ami_ec_coord_len_from_oid(const ami_der_elem *oid_el) {
    static const uint8_t oid_p256[] = {0x2Au, 0x86u, 0x48u, 0xceu, 0x3du, 0x03u, 0x01u, 0x07u};
    static const uint8_t oid_p384[] = {0x2bu, 0x81u, 0x04u, 0x00u, 0x22u};
    static const uint8_t oid_p521[] = {0x2bu, 0x81u, 0x04u, 0x00u, 0x23u};
    if (ami_oid_eq(oid_el, oid_p256, sizeof(oid_p256))) return 32u;
    if (ami_oid_eq(oid_el, oid_p384, sizeof(oid_p384))) return 48u;
    if (ami_oid_eq(oid_el, oid_p521, sizeof(oid_p521))) return 66u;
    return 0u;
}

static SecKeyRef ami_sec_key_from_ec_sec1(const uint8_t *sec1, size_t sec1_len, size_t hint_coord_len) {
    ami_der_elem seq, ver, priv_oct, opt;
    if (!ami_der_read(sec1, sec1 + sec1_len, &seq) || seq.tag != 0x30u || seq.next != sec1 + sec1_len) {
        return NULL;
    }
    if (!ami_der_read(seq.val, seq.next, &ver) || ver.tag != 0x02u || ver.len != 1 || ver.val[0] != 1u) {
        return NULL;
    }
    if (!ami_der_read(ver.next, seq.next, &priv_oct) || priv_oct.tag != 0x04u || priv_oct.len == 0) {
        return NULL;
    }
    size_t coord_len = hint_coord_len;
    const uint8_t *pub_point = NULL;
    size_t pub_point_len = 0;

    const uint8_t *cur = priv_oct.next;
    while (cur < seq.next) {
        if (!ami_der_read(cur, seq.next, &opt)) {
            return NULL;
        }
        if (opt.tag == 0xa0u) {
            ami_der_elem inner;
            if (ami_der_read(opt.val, opt.next, &inner) && inner.tag == 0x06u) {
                size_t c = ami_ec_coord_len_from_oid(&inner);
                if (c != 0u) {
                    coord_len = c;
                }
            }
        } else if (opt.tag == 0xa1u) {
            ami_der_elem bs;
            if (!ami_der_read(opt.val, opt.next, &bs) || bs.tag != 0x03u ||
                bs.next != opt.next || bs.len < 4u || bs.val[0] != 0x00u || bs.val[1] != 0x04u) {
                return NULL;
            }
            pub_point = bs.val + 1;
            pub_point_len = bs.len - 1u;
        }
        cur = opt.next;
    }

    if (pub_point == NULL) {
        return NULL;
    }
    if (coord_len == 0u && (pub_point_len - 1u) % 2u == 0u) {
        coord_len = (pub_point_len - 1u) / 2u;
    }
    if (coord_len == 0u || pub_point_len != 1u + 2u * coord_len || priv_oct.len > coord_len) {
        return NULL;
    }

    NSMutableData *x963 = [NSMutableData dataWithLength:1u + 3u * coord_len];
    uint8_t *dst = (uint8_t *)x963.mutableBytes;
    memcpy(dst, pub_point, pub_point_len);
    size_t pad = coord_len - priv_oct.len;
    if (pad > 0) {
        memset(dst + pub_point_len, 0, pad);
    }
    memcpy(dst + pub_point_len + pad, priv_oct.val, priv_oct.len);

    CFErrorRef err = NULL;
    NSDictionary *attrs = @{
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeECSECPrimeRandom,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPrivate,
    };
    SecKeyRef key = SecKeyCreateWithData((__bridge CFDataRef)x963, (__bridge CFDictionaryRef)attrs, &err);
    ami_secure_wipe_mutable_data(x963);
    if (err != NULL) {
        CFRelease(err);
    }
    return key;
}

static SecKeyRef ami_sec_key_from_rsa_pkcs1(const uint8_t *pkcs1, size_t pkcs1_len) {
    ami_der_elem seq;
    if (!ami_der_read(pkcs1, pkcs1 + pkcs1_len, &seq) || seq.tag != 0x30u || seq.next != pkcs1 + pkcs1_len) {
        return NULL;
    }
    NSData *der = [NSData dataWithBytesNoCopy:(void *)pkcs1 length:pkcs1_len freeWhenDone:NO];
    CFErrorRef err = NULL;
    NSDictionary *attrs = @{
        (__bridge id)kSecAttrKeyType: (__bridge id)kSecAttrKeyTypeRSA,
        (__bridge id)kSecAttrKeyClass: (__bridge id)kSecAttrKeyClassPrivate,
    };
    SecKeyRef key = SecKeyCreateWithData((__bridge CFDataRef)der, (__bridge CFDictionaryRef)attrs, &err);
    if (err != NULL) {
        CFRelease(err);
    }
    return key;
}

static SecKeyRef ami_sec_key_from_pkcs8(const uint8_t *pkcs8, size_t pkcs8_len) {
    static const uint8_t oid_rsa[] = {0x2Au, 0x86u, 0x48u, 0x86u, 0xf7u, 0x0du, 0x01u, 0x01u, 0x01u};
    static const uint8_t oid_ec[] = {0x2Au, 0x86u, 0x48u, 0xceu, 0x3du, 0x02u, 0x01u};

    ami_der_elem top, ver, alg_seq, key_oct;
    if (!ami_der_read(pkcs8, pkcs8 + pkcs8_len, &top) || top.tag != 0x30u || top.next != pkcs8 + pkcs8_len) {
        return NULL;
    }
    if (!ami_der_read(top.val, top.next, &ver) || ver.tag != 0x02u || ver.len != 1 || ver.val[0] != 0u) {
        return NULL;
    }
    if (!ami_der_read(ver.next, top.next, &alg_seq) || alg_seq.tag != 0x30u) {
        return NULL;
    }
    if (!ami_der_read(alg_seq.next, top.next, &key_oct) || key_oct.tag != 0x04u) {
        return NULL;
    }
    ami_der_elem alg_oid;
    if (!ami_der_read(alg_seq.val, alg_seq.next, &alg_oid) || alg_oid.tag != 0x06u) {
        return NULL;
    }
    if (ami_oid_eq(&alg_oid, oid_rsa, sizeof(oid_rsa))) {
        return ami_sec_key_from_rsa_pkcs1(key_oct.val, key_oct.len);
    }
    if (ami_oid_eq(&alg_oid, oid_ec, sizeof(oid_ec))) {
        size_t coord_hint = 0u;
        ami_der_elem curve_el;
        if (alg_oid.next < alg_seq.next && ami_der_read(alg_oid.next, alg_seq.next, &curve_el) &&
            curve_el.tag == 0x06u) {
            coord_hint = ami_ec_coord_len_from_oid(&curve_el);
        }
        return ami_sec_key_from_ec_sec1(key_oct.val, key_oct.len, coord_hint);
    }
    return NULL;
}

static bool ami_verify_cert_matches_priv_key(SecCertificateRef cert, SecKeyRef priv_key) {
    if (cert == NULL || priv_key == NULL) {
        return false;
    }
    SecKeyRef cert_pub = SecCertificateCopyKey(cert);
    SecKeyRef key_pub = SecKeyCopyPublicKey(priv_key);
    if (cert_pub == NULL || key_pub == NULL) {
        if (cert_pub != NULL) CFRelease(cert_pub);
        if (key_pub != NULL) CFRelease(key_pub);
        return false;
    }
    CFDataRef cert_pub_data = SecKeyCopyExternalRepresentation(cert_pub, NULL);
    CFDataRef key_pub_data = SecKeyCopyExternalRepresentation(key_pub, NULL);
    bool match = (cert_pub_data != NULL && key_pub_data != NULL &&
                  CFDataGetLength(cert_pub_data) > 0 &&
                  CFDataGetLength(cert_pub_data) == CFDataGetLength(key_pub_data) &&
                  memcmp(CFDataGetBytePtr(cert_pub_data),
                         CFDataGetBytePtr(key_pub_data),
                         (size_t)CFDataGetLength(cert_pub_data)) == 0);
    if (cert_pub_data != NULL) CFRelease(cert_pub_data);
    if (key_pub_data != NULL) CFRelease(key_pub_data);
    CFRelease(cert_pub);
    CFRelease(key_pub);
    return match;
}

static ami_status ami_load_sec_identity(const ami_tls_config *cfg, sec_identity_t *out_sec_id) {
    *out_sec_id = nil;
    if (cfg->cert_bytes == NULL || cfg->cert_len == 0) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    bool is_pem = (ami_memmem(cfg->cert_bytes, cfg->cert_len, "-----BEGIN ") != NULL);
    if (!is_pem) {
        // PKCS#12 (.p12 / .pfx) binary DER path — strictly in-memory via kSecImportToMemoryOnly.
        if (cfg->key_bytes != NULL && cfg->key_len > 0) {
            return AMI_ERR_INVALID_ARGUMENT;
        }
        NSString *pass_str = @"";
        if (cfg->passphrase_bytes != NULL && cfg->passphrase_len > 0) {
            pass_str = [[NSString alloc] initWithBytes:cfg->passphrase_bytes
                                                length:cfg->passphrase_len
                                              encoding:NSUTF8StringEncoding];
            if (pass_str == nil) {
                return AMI_STATUS_TLS_IDENTITY_INVALID;
            }
        }
        NSData *p12_data = [NSData dataWithBytes:cfg->cert_bytes length:cfg->cert_len];
        NSDictionary *opts = @{
            (__bridge id)kSecImportExportPassphrase: pass_str,
            (__bridge id)kSecImportToMemoryOnly: @YES,
        };
        CFArrayRef items = NULL;
        OSStatus ist = errSecDecode;
        @try {
            ist = SecPKCS12Import((__bridge CFDataRef)p12_data, (__bridge CFDictionaryRef)opts, &items);
        } @catch (NSException *ex) {
            (void)ex;
            ist = errSecDecode;
        }
        pass_str = nil;
        if (ist == errSecAuthFailed) {
            if (items != NULL) CFRelease(items);
            return (cfg->passphrase_bytes == NULL || cfg->passphrase_len == 0)
                       ? AMI_STATUS_TLS_PASSPHRASE_REQUIRED
                       : AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        if (ist != errSecSuccess || items == NULL || CFArrayGetCount(items) == 0) {
            if (items != NULL) CFRelease(items);
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }

        sec_identity_t sec_id = nil;
        CFIndex count = CFArrayGetCount(items);
        for (CFIndex i = 0; i < count; i++) {
            NSDictionary *entry = (__bridge NSDictionary *)CFArrayGetValueAtIndex(items, i);
            if (![entry isKindOfClass:[NSDictionary class]]) continue;
            SecIdentityRef ident = (__bridge SecIdentityRef)entry[(__bridge id)kSecImportItemIdentity];
            if (ident == NULL) continue;
            SecCertificateRef leaf = NULL;
            SecKeyRef priv = NULL;
            if (SecIdentityCopyCertificate(ident, &leaf) != errSecSuccess ||
                SecIdentityCopyPrivateKey(ident, &priv) != errSecSuccess ||
                !ami_verify_cert_matches_priv_key(leaf, priv)) {
                if (leaf != NULL) CFRelease(leaf);
                if (priv != NULL) CFRelease(priv);
                continue;
            }
            CFRelease(leaf);
            CFRelease(priv);
            NSArray *chain = entry[(__bridge id)kSecImportItemCertChain];
            if (chain != nil && chain.count > 1) {
                sec_id = sec_identity_create_with_certificates(ident, (__bridge CFArrayRef)chain);
            } else {
                sec_id = sec_identity_create(ident);
            }
            if (sec_id != nil) break;
        }
        CFRelease(items);
        if (sec_id == nil) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        *out_sec_id = sec_id;
        return AMI_OK;
    }

    // PEM path: decode certificate chain + decrypt/decode private key in memory.
    if (cfg->key_bytes == NULL || cfg->key_len == 0) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    NSMutableArray *certs = [NSMutableArray array];
    size_t cert_off = 0;
    ami_pem_block cblk;
    while (ami_pem_next_block(cfg->cert_bytes, cfg->cert_len, cert_off, &cblk)) {
        cert_off = cblk.next_offset;
        if (strcmp(cblk.label, "CERTIFICATE") != 0) {
            continue;
        }
        NSMutableData *der = ami_base64_decode_pem(cblk.body, cblk.body_len);
        if (der == nil) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        SecCertificateRef cert = SecCertificateCreateWithData(NULL, (__bridge CFDataRef)der);
        ami_secure_wipe_mutable_data(der);
        if (cert == NULL) {
            return AMI_STATUS_TLS_IDENTITY_INVALID;
        }
        [certs addObject:CFBridgingRelease(cert)];
    }
    if (certs.count == 0) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    ami_pem_block kblk;
    if (!ami_pem_next_block(cfg->key_bytes, cfg->key_len, 0, &kblk)) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    NSMutableData *raw_key = ami_base64_decode_pem(kblk.body, kblk.body_len);
    if (raw_key == nil) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    NSMutableData *plain_key = nil;
    bool has_legacy_enc = (kblk.headers != NULL &&
                           ami_memmem(kblk.headers, kblk.headers_len, "Proc-Type:") != NULL &&
                           ami_memmem(kblk.headers, kblk.headers_len, "ENCRYPTED") != NULL);
    SecKeyRef priv_key = NULL;

    if (strcmp(kblk.label, "ENCRYPTED PRIVATE KEY") == 0) {
        ami_status dst = ami_decrypt_pkcs8_epki((const uint8_t *)raw_key.bytes, raw_key.length,
                                                cfg->passphrase_bytes, cfg->passphrase_len,
                                                &plain_key);
        ami_secure_wipe_mutable_data(raw_key);
        if (dst != AMI_OK) {
            ami_secure_wipe_mutable_data(plain_key);
            return dst;
        }
        priv_key = ami_sec_key_from_pkcs8((const uint8_t *)plain_key.bytes, plain_key.length);
        ami_secure_wipe_mutable_data(plain_key);
    } else if (strcmp(kblk.label, "PRIVATE KEY") == 0 && !has_legacy_enc) {
        priv_key = ami_sec_key_from_pkcs8((const uint8_t *)raw_key.bytes, raw_key.length);
        ami_secure_wipe_mutable_data(raw_key);
    } else if ((strcmp(kblk.label, "RSA PRIVATE KEY") == 0 ||
                strcmp(kblk.label, "EC PRIVATE KEY") == 0)) {
        const uint8_t *key_der = (const uint8_t *)raw_key.bytes;
        size_t key_der_len = raw_key.length;
        if (has_legacy_enc) {
            ami_status dst = ami_decrypt_legacy_pem(kblk.headers, kblk.headers_len,
                                                    key_der, key_der_len,
                                                    cfg->passphrase_bytes, cfg->passphrase_len,
                                                    &plain_key);
            ami_secure_wipe_mutable_data(raw_key);
            if (dst != AMI_OK) {
                ami_secure_wipe_mutable_data(plain_key);
                return dst;
            }
            key_der = (const uint8_t *)plain_key.bytes;
            key_der_len = plain_key.length;
        }
        if (strcmp(kblk.label, "RSA PRIVATE KEY") == 0) {
            priv_key = ami_sec_key_from_rsa_pkcs1(key_der, key_der_len);
        } else {
            priv_key = ami_sec_key_from_ec_sec1(key_der, key_der_len, 0u);
        }
        ami_secure_wipe_mutable_data(raw_key);
        ami_secure_wipe_mutable_data(plain_key);
    } else {
        ami_secure_wipe_mutable_data(raw_key);
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    if (priv_key == NULL) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    SecCertificateRef leaf_cert = (__bridge SecCertificateRef)certs[0];
    if (!ami_verify_cert_matches_priv_key(leaf_cert, priv_key)) {
        CFRelease(priv_key);
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    SecIdentityRef identity = SecIdentityCreate(NULL, leaf_cert, priv_key);
    CFRelease(priv_key);
    if (identity == NULL) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }

    sec_identity_t sec_id = (certs.count > 1)
                                ? sec_identity_create_with_certificates(identity, (__bridge CFArrayRef)certs)
                                : sec_identity_create(identity);
    CFRelease(identity);
    if (sec_id == nil) {
        return AMI_STATUS_TLS_IDENTITY_INVALID;
    }
    *out_sec_id = sec_id;
    return AMI_OK;
}

static bool ami_configure_fd_nonblock_cloexec(int32_t fd, bool is_socket) {
    if (fd < 0) return false;
    if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) return false;
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl < 0) return false;
    if (fcntl(fd, F_SETFL, fl | O_NONBLOCK) != 0) return false;
    if (is_socket) {
        int one = 1;
        if (setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &one, (socklen_t)sizeof(one)) != 0) {
            return false;
        }
    }
    return true;
}

typedef struct ami_tls_pump ami_tls_pump;

struct ami_tls_listener {
    nw_listener_t listener;
    sec_identity_t identity;
    dispatch_queue_t queue;
    dispatch_group_t pump_group;
    NSMutableSet<NSValue *> *active_pumps;
    NSMutableSet<nw_connection_t> *pending_conns;
    pthread_mutex_t accept_lock;
    int32_t notify_pipe[2];
    int32_t *accept_ring;
    size_t accept_cap;
    size_t accept_head;
    size_t accept_tail;
    size_t accept_count;
    uint16_t bound_port;
    bool notify_armed;
    bool shutting_down;
};

static pthread_mutex_t g_client_pumps_lock = PTHREAD_MUTEX_INITIALIZER;
static NSMutableSet<NSValue *> *g_client_pumps = nil;

// Pure C pump struct (no @interface ObjC class) so static + dylib linkage in the
// same process never emits duplicate ObjC class runtime warnings.
struct ami_tls_pump {
    atomic_int refcount;
    void *conn;          // nw_connection_t (CFBridgingRetain)
    void *queue;         // dispatch_queue_t (CFBridgingRetain)
    void *read_src;      // dispatch_source_t (CFBridgingRetain)
    void *write_src;     // dispatch_source_t (CFBridgingRetain)
    void *pending_write; // NSMutableData * (CFBridgingRetain)
    ami_tls_listener *listener_owner;
    int32_t bridge_fd;
    int32_t active_sources;
    bool read_src_suspended;
    bool write_src_suspended;
    bool read_src_cancelled;
    bool write_src_cancelled;
    bool conn_cancelled;
    bool conn_cancel_requested;
    bool write_eof_pending;
    bool write_eof_done;
    bool read_eof_sent;
    bool receive_in_flight;
    bool retired;
    size_t pending_write_offset;
    uint8_t read_buf[65536];
};

static ami_tls_pump *ami_tls_pump_retain(ami_tls_pump *p) {
    if (p != NULL) {
        atomic_fetch_add_explicit(&p->refcount, 1, memory_order_relaxed);
    }
    return p;
}

static void ami_tls_pump_release(ami_tls_pump *p) {
    if (p != NULL && atomic_fetch_sub_explicit(&p->refcount, 1, memory_order_acq_rel) == 1) {
        if (p->pending_write != NULL) CFRelease((CFTypeRef)p->pending_write);
        if (p->read_src != NULL) CFRelease((CFTypeRef)p->read_src);
        if (p->write_src != NULL) CFRelease((CFTypeRef)p->write_src);
        if (p->conn != NULL) CFRelease((CFTypeRef)p->conn);
        if (p->queue != NULL) CFRelease((CFTypeRef)p->queue);
        free(p);
    }
}

static void ami_tls_pump_schedule_receive(ami_tls_pump *p);

static void ami_tls_pump_maybe_retire(ami_tls_pump *p) {
    if (p->retired) {
        return;
    }
    if (p->active_sources == 0 && p->bridge_fd >= 0) {
        close(p->bridge_fd);
        p->bridge_fd = -1;
    }
    if (p->active_sources == 0 && p->conn_cancelled) {
        p->retired = true;
        if (p->pending_write != NULL) {
            CFRelease((CFTypeRef)p->pending_write);
            p->pending_write = NULL;
        }
        if (p->conn != NULL) {
            nw_connection_t c = (__bridge nw_connection_t)p->conn;
            nw_connection_set_state_changed_handler(c, NULL);
            CFRelease((CFTypeRef)p->conn);
            p->conn = NULL;
        }
        NSValue *key = [NSValue valueWithPointer:p];
        if (p->listener_owner != NULL) {
            ami_tls_listener *owner = p->listener_owner;
            p->listener_owner = NULL;
            [owner->active_pumps removeObject:key];
            dispatch_group_leave(owner->pump_group);
        } else {
            pthread_mutex_lock(&g_client_pumps_lock);
            if (g_client_pumps != nil) {
                [g_client_pumps removeObject:key];
            }
            pthread_mutex_unlock(&g_client_pumps_lock);
        }
        ami_tls_pump_release(p);
    }
}

static void ami_tls_pump_cancel_sources(ami_tls_pump *p) {
    if (!p->read_src_cancelled && p->read_src != NULL) {
        p->read_src_cancelled = true;
        dispatch_source_t rs = (__bridge dispatch_source_t)p->read_src;
        dispatch_source_cancel(rs);
        if (p->read_src_suspended) {
            p->read_src_suspended = false;
            dispatch_resume(rs);
        }
    }
    if (!p->write_src_cancelled && p->write_src != NULL) {
        p->write_src_cancelled = true;
        dispatch_source_t ws = (__bridge dispatch_source_t)p->write_src;
        dispatch_source_cancel(ws);
        if (p->write_src_suspended) {
            p->write_src_suspended = false;
            dispatch_resume(ws);
        }
    }
}

static void ami_tls_pump_shutdown(ami_tls_pump *p) {
    if (!p->conn_cancel_requested && !p->conn_cancelled && p->conn != NULL) {
        p->conn_cancel_requested = true;
        nw_connection_cancel((__bridge nw_connection_t)p->conn);
    }
    ami_tls_pump_cancel_sources(p);
    ami_tls_pump_maybe_retire(p);
}

static void ami_tls_pump_on_conn_state(ami_tls_pump *p, nw_connection_state_t state) {
    if (state == nw_connection_state_failed) {
        ami_tls_pump_shutdown(p);
    } else if (state == nw_connection_state_cancelled) {
        p->conn_cancelled = true;
        ami_tls_pump_cancel_sources(p);
        ami_tls_pump_maybe_retire(p);
    }
}

static void ami_tls_pump_stop_write(ami_tls_pump *p) {
    p->write_eof_done = true;
    p->write_eof_pending = false;
    if (p->pending_write != NULL) {
        CFRelease((CFTypeRef)p->pending_write);
        p->pending_write = NULL;
        p->pending_write_offset = 0;
    }
    if (!p->write_src_cancelled && p->write_src != NULL) {
        p->write_src_cancelled = true;
        dispatch_source_t ws = (__bridge dispatch_source_t)p->write_src;
        dispatch_source_cancel(ws);
        if (p->write_src_suspended) {
            p->write_src_suspended = false;
            dispatch_resume(ws);
        }
    }
    // Keep read_src and nw_connection alive so buffered outbound replies (e.g.
    // 503/401/413 + Connection: close written before close(client_fd)) drain
    // completely and end with a clean TLS close_notify.
    ami_tls_pump_maybe_retire(p);
}

static void ami_tls_pump_flush_pending_write(ami_tls_pump *p) {
    if (p->write_src_cancelled || p->bridge_fd < 0 || p->pending_write == NULL) {
        return;
    }
    NSMutableData *pw = (__bridge NSMutableData *)p->pending_write;
    const uint8_t *base = (const uint8_t *)pw.bytes;
    size_t total = pw.length;
    while (p->pending_write_offset < total) {
        ssize_t rc = write(p->bridge_fd, base + p->pending_write_offset,
                           total - p->pending_write_offset);
        if (rc > 0) {
            p->pending_write_offset += (size_t)rc;
        } else if (rc < 0 && errno == EINTR) {
            continue;
        } else if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            return;
        } else if (rc < 0 && (errno == EPIPE || errno == ECONNRESET)) {
            ami_tls_pump_stop_write(p);
            return;
        } else {
            ami_tls_pump_shutdown(p);
            return;
        }
    }
    CFRelease((CFTypeRef)p->pending_write);
    p->pending_write = NULL;
    p->pending_write_offset = 0;
    if (!p->write_src_suspended && !p->write_src_cancelled && p->write_src != NULL) {
        p->write_src_suspended = true;
        dispatch_suspend((__bridge dispatch_source_t)p->write_src);
    }
    if (p->write_eof_pending) {
        p->write_eof_pending = false;
        p->write_eof_done = true;
        if (p->bridge_fd >= 0) {
            shutdown(p->bridge_fd, SHUT_WR);
        }
    } else {
        ami_tls_pump_schedule_receive(p);
    }
}

static void ami_tls_pump_schedule_receive(ami_tls_pump *p) {
    if (p->conn_cancelled || p->conn_cancel_requested || p->write_src_cancelled ||
        p->receive_in_flight || p->write_eof_done || p->write_eof_pending ||
        p->pending_write != NULL || p->conn == NULL) {
        return;
    }
    p->receive_in_flight = true;
    ami_tls_pump_retain(p);
    nw_connection_receive((__bridge nw_connection_t)p->conn, 1, 65536,
                          ^(dispatch_data_t content, nw_content_context_t context,
                            bool is_complete, nw_error_t error) {
        (void)context;
        p->receive_in_flight = false;
        if (p->conn_cancelled || p->write_src_cancelled || p->bridge_fd < 0) {
            ami_tls_pump_release(p);
            return;
        }
        if (content != NULL) {
            const void *ptr = NULL;
            size_t len = 0;
            dispatch_data_t map = dispatch_data_create_map(content, &ptr, &len);
            (void)map;
            size_t off = 0;
            bool would_block = false;
            while (off < len) {
                ssize_t rc = write(p->bridge_fd, (const uint8_t *)ptr + off, len - off);
                if (rc > 0) {
                    off += (size_t)rc;
                } else if (rc < 0 && errno == EINTR) {
                    continue;
                } else if (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
                    would_block = true;
                    break;
                } else if (rc < 0 && (errno == EPIPE || errno == ECONNRESET)) {
                    ami_tls_pump_stop_write(p);
                    ami_tls_pump_release(p);
                    return;
                } else {
                    ami_tls_pump_shutdown(p);
                    ami_tls_pump_release(p);
                    return;
                }
            }
            if (would_block) {
                NSMutableData *rem = [NSMutableData dataWithBytes:(const uint8_t *)ptr + off
                                                           length:len - off];
                p->pending_write = (void *)CFBridgingRetain(rem);
                p->pending_write_offset = 0;
                p->write_eof_pending = is_complete;
                if (p->write_src_suspended && !p->write_src_cancelled && p->write_src != NULL) {
                    p->write_src_suspended = false;
                    dispatch_resume((__bridge dispatch_source_t)p->write_src);
                }
                ami_tls_pump_release(p);
                return;
            }
        }
        if (is_complete) {
            p->write_eof_done = true;
            if (p->bridge_fd >= 0) {
                shutdown(p->bridge_fd, SHUT_WR);
            }
            ami_tls_pump_release(p);
            return;
        }
        if (error != NULL) {
            ami_tls_pump_shutdown(p);
            ami_tls_pump_release(p);
            return;
        }
        ami_tls_pump_schedule_receive(p);
        ami_tls_pump_release(p);
    });
}

static void ami_tls_pump_handle_read_ready(ami_tls_pump *p) {
    if (p->read_src_cancelled || p->read_eof_sent || p->conn_cancelled ||
        p->conn_cancel_requested || p->bridge_fd < 0 || p->conn == NULL) {
        return;
    }
    ssize_t n = read(p->bridge_fd, p->read_buf, sizeof(p->read_buf));
    if (n > 0) {
        if (!p->read_src_suspended && !p->read_src_cancelled && p->read_src != NULL) {
            p->read_src_suspended = true;
            dispatch_suspend((__bridge dispatch_source_t)p->read_src);
        }
        dispatch_data_t payload = dispatch_data_create(p->read_buf, (size_t)n,
                                                       (__bridge dispatch_queue_t)p->queue,
                                                       DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        ami_tls_pump_retain(p);
        nw_connection_send((__bridge nw_connection_t)p->conn, payload,
                           NW_CONNECTION_DEFAULT_MESSAGE_CONTEXT, false,
                           ^(nw_error_t error) {
            if (error != NULL || p->conn_cancelled || p->conn_cancel_requested) {
                ami_tls_pump_shutdown(p);
                ami_tls_pump_release(p);
                return;
            }
            if (p->read_src_suspended && !p->read_src_cancelled &&
                !p->read_eof_sent && p->read_src != NULL) {
                p->read_src_suspended = false;
                dispatch_resume((__bridge dispatch_source_t)p->read_src);
            }
            ami_tls_pump_release(p);
        });
    } else if (n == 0) {
        p->read_eof_sent = true;
        if (!p->read_src_cancelled && p->read_src != NULL) {
            p->read_src_cancelled = true;
            dispatch_source_cancel((__bridge dispatch_source_t)p->read_src);
        }
        ami_tls_pump_retain(p);
        nw_connection_send((__bridge nw_connection_t)p->conn, NULL,
                           NW_CONNECTION_FINAL_MESSAGE_CONTEXT, true,
                           ^(nw_error_t error) {
            (void)error;
            if (!p->conn_cancel_requested && !p->conn_cancelled && p->conn != NULL) {
                p->conn_cancel_requested = true;
                nw_connection_cancel((__bridge nw_connection_t)p->conn);
            }
            ami_tls_pump_release(p);
        });
    } else {
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
            return;
        }
        ami_tls_pump_shutdown(p);
    }
}

static ami_tls_pump *ami_tls_pump_create_and_start(nw_connection_t c, int32_t fd,
                                                   dispatch_queue_t q,
                                                   ami_tls_listener *owner) {
    ami_tls_pump *p = (ami_tls_pump *)calloc(1, sizeof(ami_tls_pump));
    if (p == NULL) {
        return NULL;
    }
    atomic_init(&p->refcount, 1);
    p->conn = (void *)CFBridgingRetain(c);
    p->queue = (void *)CFBridgingRetain(q);
    p->listener_owner = owner;
    p->bridge_fd = fd;
    p->read_src_suspended = true;
    p->write_src_suspended = true;

    dispatch_source_t rs = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, (uintptr_t)fd, 0, q);
    dispatch_source_t ws = dispatch_source_create(DISPATCH_SOURCE_TYPE_WRITE, (uintptr_t)fd, 0, q);
    if (rs == nil || ws == nil) {
        if (rs != nil) {
            dispatch_source_cancel(rs);
            dispatch_resume(rs);
        }
        if (ws != nil) {
            dispatch_source_cancel(ws);
            dispatch_resume(ws);
        }
        ami_tls_pump_release(p);
        return NULL;
    }
    p->read_src = (void *)CFBridgingRetain(rs);
    p->write_src = (void *)CFBridgingRetain(ws);
    p->active_sources = 2;

    ami_tls_pump_retain(p); // owned by read_src cancel handler
    dispatch_source_set_event_handler(rs, ^{
        ami_tls_pump_handle_read_ready(p);
    });
    dispatch_source_set_cancel_handler(rs, ^{
        if (p->read_src != NULL) {
            CFRelease((CFTypeRef)p->read_src);
            p->read_src = NULL;
        }
        p->active_sources--;
        ami_tls_pump_maybe_retire(p);
        ami_tls_pump_release(p);
    });

    ami_tls_pump_retain(p); // owned by write_src cancel handler
    dispatch_source_set_event_handler(ws, ^{
        ami_tls_pump_flush_pending_write(p);
    });
    dispatch_source_set_cancel_handler(ws, ^{
        if (p->write_src != NULL) {
            CFRelease((CFTypeRef)p->write_src);
            p->write_src = NULL;
        }
        p->active_sources--;
        ami_tls_pump_maybe_retire(p);
        ami_tls_pump_release(p);
    });

    NSValue *key = [NSValue valueWithPointer:p];
    if (owner != NULL) {
        [owner->active_pumps addObject:key];
    } else {
        pthread_mutex_lock(&g_client_pumps_lock);
        if (g_client_pumps == nil) {
            g_client_pumps = [NSMutableSet set];
        }
        [g_client_pumps addObject:key];
        pthread_mutex_unlock(&g_client_pumps_lock);
    }

    p->read_src_suspended = false;
    dispatch_resume(rs);
    ami_tls_pump_schedule_receive(p);
    return p;
}

static void ami_drain_notify_pipe_locked(ami_tls_listener *lst) {
    uint8_t drain[64];
    for (;;) {
        ssize_t r = read(lst->notify_pipe[0], drain, sizeof(drain));
        if (r > 0) continue;
        if (r < 0 && errno == EINTR) continue;
        break;
    }
    lst->notify_armed = false;
}

ami_status ami_tls_listener_create(const ami_tls_config *cfg,
                                   ami_tls_listener **out_listener) {
    if (out_listener != NULL) {
        *out_listener = NULL;
    }
    if (cfg == NULL || out_listener == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        sec_identity_t sec_id = nil;
        ami_status id_st = ami_load_sec_identity(cfg, &sec_id);
        if (id_st != AMI_OK) {
            return id_st;
        }

        size_t cap = cfg->backlog == 0 ? 128u : (size_t)cfg->backlog;
        if (cap > 4096u) cap = 4096u;

        ami_tls_listener *lst = (ami_tls_listener *)calloc(1, sizeof(ami_tls_listener));
        if (lst == NULL) {
            return AMI_ERR_OUT_OF_MEMORY;
        }
        lst->notify_pipe[0] = -1;
        lst->notify_pipe[1] = -1;
        lst->accept_cap = cap;
        lst->accept_ring = (int32_t *)calloc(cap, sizeof(int32_t));
        if (lst->accept_ring == NULL) {
            free(lst);
            return AMI_ERR_OUT_OF_MEMORY;
        }
        for (size_t i = 0; i < cap; i++) {
            lst->accept_ring[i] = -1;
        }

        if (pipe(lst->notify_pipe) != 0 ||
            !ami_configure_fd_nonblock_cloexec(lst->notify_pipe[0], false) ||
            !ami_configure_fd_nonblock_cloexec(lst->notify_pipe[1], false)) {
            if (lst->notify_pipe[0] >= 0) close(lst->notify_pipe[0]);
            if (lst->notify_pipe[1] >= 0) close(lst->notify_pipe[1]);
            free(lst->accept_ring);
            free(lst);
            return AMI_STATUS_TLS_BIND_FAILED;
        }

        if (pthread_mutex_init(&lst->accept_lock, NULL) != 0) {
            close(lst->notify_pipe[0]);
            close(lst->notify_pipe[1]);
            free(lst->accept_ring);
            free(lst);
            return AMI_ERR_OUT_OF_MEMORY;
        }

        lst->identity = sec_id;
        dispatch_queue_attr_t qattr = dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
        lst->queue = dispatch_queue_create("com.aminfer.tls.listener", qattr);
        lst->pump_group = dispatch_group_create();
        lst->active_pumps = [NSMutableSet set];
        lst->pending_conns = [NSMutableSet set];

        nw_parameters_t params = nw_parameters_create_secure_tcp(
            ^(nw_protocol_options_t tls_opts) {
                sec_protocol_options_t sec_opts = nw_tls_copy_sec_protocol_options(tls_opts);
                sec_protocol_options_set_local_identity(sec_opts, sec_id);
                sec_protocol_options_set_min_tls_protocol_version(sec_opts, tls_protocol_version_TLSv12);
                sec_protocol_options_set_max_tls_protocol_version(sec_opts, tls_protocol_version_TLSv13);
            },
            ^(nw_protocol_options_t tcp_opts) {
                nw_tcp_options_set_no_delay(tcp_opts, true);
            });
        if (params == nil) {
            ami_tls_listener_destroy(lst);
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        nw_parameters_set_reuse_local_address(params, true);

        char ip_buf[32];
        char port_buf[16];
        snprintf(ip_buf, sizeof(ip_buf), "%u.%u.%u.%u",
                 cfg->host_ipv4[0], cfg->host_ipv4[1], cfg->host_ipv4[2], cfg->host_ipv4[3]);
        snprintf(port_buf, sizeof(port_buf), "%u", (unsigned)cfg->port);
        nw_endpoint_t local_ep = nw_endpoint_create_host(ip_buf, port_buf);
        if (local_ep == nil) {
            ami_tls_listener_destroy(lst);
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        nw_parameters_set_local_endpoint(params, local_ep);

        nw_listener_t nw_lst = nw_listener_create(params);
        if (nw_lst == nil) {
            ami_tls_listener_destroy(lst);
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        lst->listener = nw_lst;
        nw_listener_set_queue(nw_lst, lst->queue);

        dispatch_semaphore_t ready_sem = dispatch_semaphore_create(0);
        __block bool ready_ok = false;
        __block bool startup_signaled = false;
        __block uint16_t bound = 0;

        nw_listener_set_state_changed_handler(nw_lst, ^(nw_listener_state_t state, nw_error_t error) {
            if (state == nw_listener_state_ready) {
                bound = nw_listener_get_port(nw_lst);
                ready_ok = (bound > 0);
                if (!startup_signaled) {
                    startup_signaled = true;
                    dispatch_semaphore_signal(ready_sem);
                }
            } else if (state == nw_listener_state_failed ||
                       (state == nw_listener_state_waiting && error != NULL)) {
                ready_ok = false;
                if (!startup_signaled) {
                    startup_signaled = true;
                    dispatch_semaphore_signal(ready_sem);
                }
            }
        });

        nw_listener_set_new_connection_handler(nw_lst, ^(nw_connection_t conn) {
            pthread_mutex_lock(&lst->accept_lock);
            bool down = lst->shutting_down;
            pthread_mutex_unlock(&lst->accept_lock);
            if (down) {
                nw_connection_cancel(conn);
                return;
            }
            [lst->pending_conns addObject:conn];
            dispatch_group_enter(lst->pump_group);
            nw_connection_set_queue(conn, lst->queue);

            __block ami_tls_pump *pump = NULL;
            __block bool pending_retired = false;
            nw_connection_set_state_changed_handler(conn, ^(nw_connection_state_t state, nw_error_t err) {
                (void)err;
                if (pump != NULL) {
                    ami_tls_pump_on_conn_state(pump, state);
                    return;
                }
                if (state == nw_connection_state_ready) {
                    [lst->pending_conns removeObject:conn];
                    pending_retired = true;

                    int fds[2] = {-1, -1};
                    if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
                        nw_connection_cancel(conn);
                        dispatch_group_leave(lst->pump_group);
                        return;
                    }
                    if (!ami_configure_fd_nonblock_cloexec(fds[0], true) ||
                        !ami_configure_fd_nonblock_cloexec(fds[1], true)) {
                        close(fds[0]);
                        close(fds[1]);
                        nw_connection_cancel(conn);
                        dispatch_group_leave(lst->pump_group);
                        return;
                    }

                    pthread_mutex_lock(&lst->accept_lock);
                    bool accept_ok = !lst->shutting_down && (lst->accept_count < lst->accept_cap);
                    if (accept_ok) {
                        lst->accept_ring[lst->accept_tail] = fds[1];
                        lst->accept_tail = (lst->accept_tail + 1u) % lst->accept_cap;
                        lst->accept_count++;
                        if (!lst->notify_armed) {
                            uint8_t sig = 1u;
                            ssize_t wr = write(lst->notify_pipe[1], &sig, 1u);
                            if (wr == 1 || (wr < 0 && errno == EAGAIN)) {
                                lst->notify_armed = true;
                            }
                        }
                    }
                    pthread_mutex_unlock(&lst->accept_lock);

                    if (!accept_ok) {
                        close(fds[0]);
                        close(fds[1]);
                        nw_connection_cancel(conn);
                        dispatch_group_leave(lst->pump_group);
                        return;
                    }

                    pump = ami_tls_pump_create_and_start(conn, fds[0], lst->queue, lst);
                    if (pump == NULL) {
                        close(fds[0]);
                        nw_connection_cancel(conn);
                        dispatch_group_leave(lst->pump_group);
                        return;
                    }
                } else if (state == nw_connection_state_failed) {
                    nw_connection_cancel(conn);
                } else if (state == nw_connection_state_cancelled) {
                    if (!pending_retired) {
                        pending_retired = true;
                        [lst->pending_conns removeObject:conn];
                        nw_connection_set_state_changed_handler(conn, NULL);
                        dispatch_group_leave(lst->pump_group);
                    }
                }
            });
            nw_connection_start(conn);
        });

        nw_listener_start(nw_lst);
        long wait_rc = dispatch_semaphore_wait(ready_sem, dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC));
        if (wait_rc != 0 || !ready_ok || bound == 0) {
            ami_tls_listener_destroy(lst);
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        lst->bound_port = bound;
        *out_listener = lst;
        return AMI_OK;
    }
}

void ami_tls_listener_destroy(ami_tls_listener *listener) {
    if (listener == NULL) {
        return;
    }
    @autoreleasepool {
        pthread_mutex_lock(&listener->accept_lock);
        listener->shutting_down = true;
        while (listener->accept_count > 0) {
            int32_t fd = listener->accept_ring[listener->accept_head];
            listener->accept_ring[listener->accept_head] = -1;
            listener->accept_head = (listener->accept_head + 1u) % listener->accept_cap;
            listener->accept_count--;
            if (fd >= 0) close(fd);
        }
        pthread_mutex_unlock(&listener->accept_lock);

        if (listener->listener != nil && listener->queue != nil) {
            dispatch_semaphore_t cancel_sem = dispatch_semaphore_create(0);
            dispatch_sync(listener->queue, ^{
                nw_listener_set_new_connection_handler(listener->listener, NULL);
                nw_listener_set_state_changed_handler(listener->listener, ^(nw_listener_state_t st, nw_error_t err) {
                    (void)err;
                    if (st == nw_listener_state_cancelled || st == nw_listener_state_failed) {
                        dispatch_semaphore_signal(cancel_sem);
                    }
                });
                nw_listener_cancel(listener->listener);
                for (nw_connection_t pconn in [listener->pending_conns allObjects]) {
                    nw_connection_cancel(pconn);
                }
                for (NSValue *v in [listener->active_pumps allObjects]) {
                    ami_tls_pump *p = (ami_tls_pump *)v.pointerValue;
                    if (p != NULL) {
                        ami_tls_pump_shutdown(p);
                    }
                }
            });
            (void)dispatch_semaphore_wait(cancel_sem, dispatch_time(DISPATCH_TIME_NOW, 2LL * NSEC_PER_SEC));
        }

        if (listener->pump_group != nil) {
            (void)dispatch_group_wait(listener->pump_group, dispatch_time(DISPATCH_TIME_NOW, 2LL * NSEC_PER_SEC));
        }

        if (listener->queue != nil) {
            dispatch_sync(listener->queue, ^{
                if (listener->listener != nil) {
                    nw_listener_set_state_changed_handler(listener->listener, NULL);
                }
            });
        }

        if (listener->notify_pipe[0] >= 0) close(listener->notify_pipe[0]);
        if (listener->notify_pipe[1] >= 0) close(listener->notify_pipe[1]);
        pthread_mutex_destroy(&listener->accept_lock);
        free(listener->accept_ring);
        listener->listener = nil;
        listener->identity = nil;
        listener->active_pumps = nil;
        listener->pending_conns = nil;
        listener->pump_group = nil;
        listener->queue = nil;
        free(listener);
    }
}

uint16_t ami_tls_listener_bound_port(const ami_tls_listener *listener) {
    if (listener == NULL) return 0;
    return listener->bound_port;
}

int32_t ami_tls_listener_notify_fd(const ami_tls_listener *listener) {
    if (listener == NULL) return -1;
    return listener->notify_pipe[0];
}

ami_status ami_tls_listener_pop_accepted_fd(ami_tls_listener *listener, int32_t *out_fd) {
    if (out_fd != NULL) {
        *out_fd = -1;
    }
    if (listener == NULL || out_fd == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&listener->accept_lock);
    if (listener->accept_count > 0) {
        int32_t fd = listener->accept_ring[listener->accept_head];
        listener->accept_ring[listener->accept_head] = -1;
        listener->accept_head = (listener->accept_head + 1u) % listener->accept_cap;
        listener->accept_count--;
        *out_fd = fd;
        if (listener->accept_count == 0) {
            ami_drain_notify_pipe_locked(listener);
        }
    } else {
        *out_fd = -1;
        ami_drain_notify_pipe_locked(listener);
    }
    pthread_mutex_unlock(&listener->accept_lock);
    return AMI_OK;
}

ami_status ami_tls_client_connect_socketpair(const uint8_t host_ipv4[4],
                                             uint16_t port,
                                             uint16_t min_tls_version,
                                             uint16_t max_tls_version,
                                             int32_t *out_client_fd,
                                             uint16_t *out_tls_version) {
    if (out_client_fd != NULL) *out_client_fd = -1;
    if (out_tls_version != NULL) *out_tls_version = 0;
    if (host_ipv4 == NULL || port == 0 || out_client_fd == NULL || out_tls_version == NULL) {
        return AMI_ERR_INVALID_ARGUMENT;
    }

    @autoreleasepool {
        dispatch_queue_attr_t cq_attr = dispatch_queue_attr_make_with_autorelease_frequency(
            DISPATCH_QUEUE_SERIAL, DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
        dispatch_queue_t cq = dispatch_queue_create("com.aminfer.tls.client", cq_attr);
        uint16_t min_v = min_tls_version != 0 ? min_tls_version : (uint16_t)tls_protocol_version_TLSv12;
        uint16_t max_v = max_tls_version != 0 ? max_tls_version : (uint16_t)tls_protocol_version_TLSv13;

        nw_parameters_t params = nw_parameters_create_secure_tcp(
            ^(nw_protocol_options_t tls_opts) {
                sec_protocol_options_t sec_opts = nw_tls_copy_sec_protocol_options(tls_opts);
                sec_protocol_options_set_min_tls_protocol_version(sec_opts, (tls_protocol_version_t)min_v);
                sec_protocol_options_set_max_tls_protocol_version(sec_opts, (tls_protocol_version_t)max_v);
                sec_protocol_options_set_verify_block(sec_opts,
                    ^(sec_protocol_metadata_t metadata, sec_trust_t trust,
                      sec_protocol_verify_complete_t complete) {
                        (void)metadata;
                        (void)trust;
                        complete(true);
                    }, cq);
            },
            ^(nw_protocol_options_t tcp_opts) {
                nw_tcp_options_set_no_delay(tcp_opts, true);
            });
        if (params == nil) {
            return AMI_STATUS_TLS_BIND_FAILED;
        }

        char ip_buf[32];
        char port_buf[16];
        snprintf(ip_buf, sizeof(ip_buf), "%u.%u.%u.%u",
                 host_ipv4[0], host_ipv4[1], host_ipv4[2], host_ipv4[3]);
        snprintf(port_buf, sizeof(port_buf), "%u", (unsigned)port);
        nw_endpoint_t ep = nw_endpoint_create_host(ip_buf, port_buf);
        if (ep == nil) {
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        nw_connection_t conn = nw_connection_create(ep, params);
        if (conn == nil) {
            return AMI_STATUS_TLS_BIND_FAILED;
        }
        nw_connection_set_queue(conn, cq);

        dispatch_semaphore_t sem = dispatch_semaphore_create(0);
        __block bool ready_ok = false;
        __block bool signaled = false;
        __block uint16_t negotiated = 0;
        __block ami_tls_pump *pump = NULL;

        nw_connection_set_state_changed_handler(conn, ^(nw_connection_state_t state, nw_error_t err) {
            if (pump != NULL) {
                ami_tls_pump_on_conn_state(pump, state);
                return;
            }
            if (state == nw_connection_state_ready) {
                nw_protocol_definition_t tls_def = nw_protocol_copy_tls_definition();
                nw_protocol_metadata_t proto_md = nw_connection_copy_protocol_metadata(conn, tls_def);
                if (proto_md != nil && nw_protocol_metadata_is_tls(proto_md)) {
                    sec_protocol_metadata_t sec_md = nw_tls_copy_sec_protocol_metadata(proto_md);
                    negotiated = (uint16_t)sec_protocol_metadata_get_negotiated_tls_protocol_version(sec_md);
                }
                ready_ok = true;
                if (!signaled) {
                    signaled = true;
                    dispatch_semaphore_signal(sem);
                }
            } else if (state == nw_connection_state_failed ||
                       (state == nw_connection_state_waiting && err != NULL)) {
                ready_ok = false;
                if (!signaled) {
                    signaled = true;
                    dispatch_semaphore_signal(sem);
                }
            }
        });
        nw_connection_start(conn);

        long wrc = dispatch_semaphore_wait(sem, dispatch_time(DISPATCH_TIME_NOW, 5LL * NSEC_PER_SEC));
        if (wrc != 0 || !ready_ok) {
            dispatch_sync(cq, ^{
                nw_connection_set_state_changed_handler(conn, NULL);
                nw_connection_cancel(conn);
            });
            return AMI_STATUS_TLS_BIND_FAILED;
        }

        int fds[2] = {-1, -1};
        if (socketpair(AF_UNIX, SOCK_STREAM, 0, fds) != 0) {
            dispatch_sync(cq, ^{
                nw_connection_set_state_changed_handler(conn, NULL);
                nw_connection_cancel(conn);
            });
            return AMI_ERR_OUT_OF_MEMORY;
        }
        if (!ami_configure_fd_nonblock_cloexec(fds[0], true) ||
            !ami_configure_fd_nonblock_cloexec(fds[1], true)) {
            close(fds[0]);
            close(fds[1]);
            dispatch_sync(cq, ^{
                nw_connection_set_state_changed_handler(conn, NULL);
                nw_connection_cancel(conn);
            });
            return AMI_STATUS_TLS_BIND_FAILED;
        }

        int32_t bridge_fd = fds[0];
        int32_t client_fd = fds[1];
        __block bool start_ok = false;
        dispatch_sync(cq, ^{
            ami_tls_pump *p = ami_tls_pump_create_and_start(conn, bridge_fd, cq, NULL);
            if (p != NULL) {
                pump = p;
                start_ok = true;
            } else {
                nw_connection_set_state_changed_handler(conn, NULL);
                nw_connection_cancel(conn);
            }
        });
        if (!start_ok) {
            close(bridge_fd);
            close(client_fd);
            return AMI_ERR_OUT_OF_MEMORY;
        }

        *out_client_fd = client_fd;
        *out_tls_version = negotiated;
        return AMI_OK;
    }
}
