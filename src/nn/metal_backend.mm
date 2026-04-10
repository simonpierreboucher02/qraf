#include "nn/metal_backend.h"
#include "core/logging.h"

#if QRAF_HAS_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <cstring>
#include <string>

namespace qraf {
namespace metal {

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;

// Pipeline states for each kernel
static id<MTLComputePipelineState> g_matvec_pipeline = nil;
static id<MTLComputePipelineState> g_softmax_pipeline = nil;
static id<MTLComputePipelineState> g_rms_norm_pipeline = nil;
static id<MTLComputePipelineState> g_silu_pipeline = nil;
static id<MTLComputePipelineState> g_add_pipeline = nil;
static id<MTLComputePipelineState> g_mul_pipeline = nil;

static bool g_initialized = false;

static id<MTLComputePipelineState> make_pipeline(const char* name) {
    NSError* error = nil;
    id<MTLFunction> fn = [g_library newFunctionWithName:
                          [NSString stringWithUTF8String:name]];
    if (!fn) {
        log::error("Metal: function '%s' not found in library", name);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [g_device newComputePipelineStateWithFunction:fn error:&error];
    if (!pipeline) {
        log::error("Metal: failed to create pipeline for '%s': %s",
                   name, [[error localizedDescription] UTF8String]);
        return nil;
    }
    return pipeline;
}

bool init() {
    if (g_initialized) return true;

    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            log::warn("Metal: no GPU device available");
            return false;
        }

        log::info("Metal GPU: %s", [[g_device name] UTF8String]);

        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            log::error("Metal: failed to create command queue");
            return false;
        }

        // Load shader library from compiled metallib or source
        NSError* error = nil;

        // Try loading from source file next to the binary
        NSString* shaderPath = nil;
        NSArray* searchPaths = @[
            @"metal_shaders.metal",
            @"../src/nn/metal_shaders.metal",
            @"../../src/nn/metal_shaders.metal",
        ];

        for (NSString* path in searchPaths) {
            if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
                shaderPath = path;
                break;
            }
        }

        if (shaderPath) {
            NSString* source = [NSString stringWithContentsOfFile:shaderPath
                                                        encoding:NSUTF8StringEncoding
                                                           error:&error];
            if (source) {
                MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
                opts.fastMathEnabled = YES;
                g_library = [g_device newLibraryWithSource:source options:opts error:&error];
            }
        }

        if (!g_library) {
            // Try default library
            g_library = [g_device newDefaultLibrary];
        }

        if (!g_library) {
            log::warn("Metal: no shader library found (GPU kernels disabled)");
            log::warn("Metal: place metal_shaders.metal next to the binary");
            return false;
        }

        // Create pipeline states
        g_matvec_pipeline   = make_pipeline("matvec_f32");
        g_softmax_pipeline  = make_pipeline("softmax_pass1");
        g_rms_norm_pipeline = make_pipeline("rms_norm_f32");
        g_silu_pipeline     = make_pipeline("silu_f32");
        g_add_pipeline      = make_pipeline("add_f32");
        g_mul_pipeline      = make_pipeline("mul_f32");

        g_initialized = true;
        log::info("Metal backend initialized with %lu pipelines",
                  (g_matvec_pipeline ? 1UL : 0) + (g_softmax_pipeline ? 1UL : 0) +
                  (g_rms_norm_pipeline ? 1UL : 0) + (g_silu_pipeline ? 1UL : 0) +
                  (g_add_pipeline ? 1UL : 0) + (g_mul_pipeline ? 1UL : 0));
        return true;
    }
}

void shutdown() {
    g_matvec_pipeline = nil;
    g_softmax_pipeline = nil;
    g_rms_norm_pipeline = nil;
    g_silu_pipeline = nil;
    g_add_pipeline = nil;
    g_mul_pipeline = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
    g_initialized = false;
}

bool is_available() { return g_initialized && g_matvec_pipeline != nil; }

// ─── Helper: run a compute command ───
static void run_command(id<MTLComputePipelineState> pipeline,
                        void (^encode_block)(id<MTLComputeCommandEncoder>),
                        MTLSize grid, MTLSize group) {
    @autoreleasepool {
        id<MTLCommandBuffer> buf = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [buf computeCommandEncoder];
        [enc setComputePipelineState:pipeline];
        encode_block(enc);
        [enc dispatchThreads:grid threadsPerThreadgroup:group];
        [enc endEncoding];
        [buf commit];
        [buf waitUntilCompleted];
    }
}

// ─── MatVec GPU ───
void matvec_f32(const f32* W, const f32* x, f32* y, u32 out_dim, u32 in_dim) {
    if (!g_matvec_pipeline) return;

    @autoreleasepool {
        size_t w_size = (size_t)out_dim * in_dim * sizeof(f32);
        size_t x_size = (size_t)in_dim * sizeof(f32);
        size_t y_size = (size_t)out_dim * sizeof(f32);

        id<MTLBuffer> w_buf = [g_device newBufferWithBytes:W length:w_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:x_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> y_buf = [g_device newBufferWithLength:y_size options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_matvec_pipeline];
        [enc setBuffer:w_buf offset:0 atIndex:0];
        [enc setBuffer:x_buf offset:0 atIndex:1];
        [enc setBuffer:y_buf offset:0 atIndex:2];
        [enc setBytes:&out_dim length:sizeof(u32) atIndex:3];
        [enc setBytes:&in_dim length:sizeof(u32) atIndex:4];

        NSUInteger tpg = MIN(g_matvec_pipeline.maxTotalThreadsPerThreadgroup, 256UL);
        [enc dispatchThreads:MTLSizeMake(out_dim, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(y, [y_buf contents], y_size);
    }
}

// ─── Softmax GPU ───
void softmax(f32* x, int size) {
    if (!g_softmax_pipeline || size < 64) return;

    @autoreleasepool {
        size_t buf_size = (size_t)size * sizeof(f32);
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> scratch = [g_device newBufferWithLength:8 options:MTLResourceStorageModeShared];
        u32 sz = (u32)size;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_softmax_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:scratch offset:0 atIndex:1];
        [enc setBytes:&sz length:sizeof(u32) atIndex:2];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(x, [x_buf contents], buf_size);
    }
}

// ─── RMSNorm GPU ───
void rms_norm(f32* x, const f32* weight, int size, f32 eps) {
    if (!g_rms_norm_pipeline) return;

    @autoreleasepool {
        size_t buf_size = (size_t)size * sizeof(f32);
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> w_buf = [g_device newBufferWithBytes:weight length:buf_size options:MTLResourceStorageModeShared];
        u32 sz = (u32)size;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_rms_norm_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:w_buf offset:0 atIndex:1];
        [enc setBytes:&sz length:sizeof(u32) atIndex:2];
        [enc setBytes:&eps length:sizeof(f32) atIndex:3];
        [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(x, [x_buf contents], buf_size);
    }
}

// ─── SiLU GPU ───
void silu(f32* x, int size) {
    if (!g_silu_pipeline) return;

    @autoreleasepool {
        size_t buf_size = (size_t)size * sizeof(f32);
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:buf_size options:MTLResourceStorageModeShared];
        u32 sz = (u32)size;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_silu_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBytes:&sz length:sizeof(u32) atIndex:1];

        NSUInteger tpg = MIN(g_silu_pipeline.maxTotalThreadsPerThreadgroup, 256UL);
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(x, [x_buf contents], buf_size);
    }
}

// ─── Add GPU ───
void add(f32* x, const f32* y, int size) {
    if (!g_add_pipeline) return;

    @autoreleasepool {
        size_t buf_size = (size_t)size * sizeof(f32);
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> y_buf = [g_device newBufferWithBytes:y length:buf_size options:MTLResourceStorageModeShared];
        u32 sz = (u32)size;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_add_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:y_buf offset:0 atIndex:1];
        [enc setBytes:&sz length:sizeof(u32) atIndex:2];

        NSUInteger tpg = MIN(g_add_pipeline.maxTotalThreadsPerThreadgroup, 256UL);
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(x, [x_buf contents], buf_size);
    }
}

// ─── Mul GPU ───
void mul(f32* x, const f32* y, int size) {
    if (!g_mul_pipeline) return;

    @autoreleasepool {
        size_t buf_size = (size_t)size * sizeof(f32);
        id<MTLBuffer> x_buf = [g_device newBufferWithBytes:x length:buf_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> y_buf = [g_device newBufferWithBytes:y length:buf_size options:MTLResourceStorageModeShared];
        u32 sz = (u32)size;

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_mul_pipeline];
        [enc setBuffer:x_buf offset:0 atIndex:0];
        [enc setBuffer:y_buf offset:0 atIndex:1];
        [enc setBytes:&sz length:sizeof(u32) atIndex:2];

        NSUInteger tpg = MIN(g_mul_pipeline.maxTotalThreadsPerThreadgroup, 256UL);
        [enc dispatchThreads:MTLSizeMake(size, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        memcpy(x, [x_buf contents], buf_size);
    }
}

} // namespace metal
} // namespace qraf

#endif // QRAF_HAS_METAL
