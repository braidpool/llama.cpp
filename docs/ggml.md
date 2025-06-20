# GGML Tensor Library

GGML is a minimalistic tensor library for machine learning applications, designed for efficient tensor operations, automatic differentiation, and basic optimization algorithms.

## Overview

GGML provides:
- A comprehensive set of tensor operations
- Automatic differentiation capabilities
- Basic optimization algorithms
- Multi-backend support (CPU, GPU, accelerators)
- Quantized data types for memory efficiency
- Computation graph execution

The library is designed around a computation graph approach where operations are defined symbolically and executed later, enabling efficient memory management and optimization.

## Getting Started

### Include Headers

```c
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"  // For memory allocation helpers
```

### Basic Usage Pattern

```c
// 1. Initialize context
struct ggml_init_params params = {
    .mem_size   = 16*1024*1024,  // 16MB
    .mem_buffer = NULL,          // Auto-allocate
    .no_alloc   = false
};
struct ggml_context * ctx = ggml_init(params);

// 2. Create tensors and define computation
struct ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
struct ggml_tensor * result = ggml_add(ctx, a, b);

// 3. Build computation graph
struct ggml_cgraph * graph = ggml_new_graph(ctx);
ggml_build_forward_expand(graph, result);

// 4. Set up backend and allocate memory
ggml_backend_t backend = ggml_backend_init_best();
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

// 5. Set input data
ggml_backend_tensor_set(a, input_data_a, 0, ggml_nbytes(a));
ggml_backend_tensor_set(b, input_data_b, 0, ggml_nbytes(b));

// 6. Execute computation
ggml_backend_graph_compute(backend, graph);

// 7. Get results
ggml_backend_tensor_get(result, output_data, 0, ggml_nbytes(result));

// 8. Cleanup
ggml_backend_buffer_free(buffer);
ggml_backend_free(backend);
ggml_free(ctx);
```

## Core Concepts

### Tensors

Tensors are the fundamental data structure in GGML, supporting up to 4 dimensions:

```c
// Create tensors of different dimensions
struct ggml_tensor * scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
struct ggml_tensor * vector = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
struct ggml_tensor * matrix = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1000, 512);
struct ggml_tensor * tensor3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 64, 3);
struct ggml_tensor * tensor4d = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 224, 224, 3, 32);
```

### Data Types

GGML supports various data types optimized for different use cases:

**Floating Point:**
- `GGML_TYPE_F32` - 32-bit float
- `GGML_TYPE_F16` - 16-bit half precision
- `GGML_TYPE_BF16` - bfloat16
- `GGML_TYPE_F64` - 64-bit double

**Integer:**
- `GGML_TYPE_I8`, `GGML_TYPE_I16`, `GGML_TYPE_I32`, `GGML_TYPE_I64`
- `GGML_TYPE_UINT8`, `GGML_TYPE_UINT16`, `GGML_TYPE_UINT32`, `GGML_TYPE_UINT64`

**Quantized Types (for memory efficiency):**
- `GGML_TYPE_Q4_0`, `GGML_TYPE_Q4_1` - 4-bit quantization
- `GGML_TYPE_Q5_0`, `GGML_TYPE_Q5_1` - 5-bit quantization
- `GGML_TYPE_Q8_0`, `GGML_TYPE_Q8_1` - 8-bit quantization
- `GGML_TYPE_Q2_K`, `GGML_TYPE_Q3_K`, `GGML_TYPE_Q4_K`, etc. - K-quantization variants

### Computation Graph

Operations in GGML are lazy - they build a computation graph that is executed later:

```c
// These operations only define the graph structure
struct ggml_tensor * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1000);
struct ggml_tensor * y = ggml_mul(ctx, x, x);           // y = x²
struct ggml_tensor * z = ggml_add_inplace(ctx, y, x);   // y = x² + x (in-place)
struct ggml_tensor * result = ggml_sum(ctx, z);         // sum all elements

// Actual computation happens here
ggml_backend_graph_compute(backend, graph);
```

## Common Operations

### Basic Arithmetic

```c
// Element-wise operations
struct ggml_tensor * sum = ggml_add(ctx, a, b);      // a + b
struct ggml_tensor * diff = ggml_sub(ctx, a, b);     // a - b
struct ggml_tensor * prod = ggml_mul(ctx, a, b);     // a * b (element-wise)
struct ggml_tensor * quot = ggml_div(ctx, a, b);     // a / b

// In-place variants (modify first operand)
ggml_add_inplace(ctx, a, b);  // a += b
ggml_sub_inplace(ctx, a, b);  // a -= b
```

### Matrix Operations

```c
// Matrix multiplication
struct ggml_tensor * matmul = ggml_mul_mat(ctx, A, B);

// Transpose
struct ggml_tensor * At = ggml_transpose(ctx, A);

// Reshape
struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, tensor, new_rows, new_cols);
```

### Activation Functions

```c
// Common activation functions
struct ggml_tensor * relu_out = ggml_relu(ctx, x);
struct ggml_tensor * gelu_out = ggml_gelu(ctx, x);
struct ggml_tensor * silu_out = ggml_silu(ctx, x);
struct ggml_tensor * tanh_out = ggml_tanh(ctx, x);

// Softmax
struct ggml_tensor * softmax_out = ggml_soft_max(ctx, x);
```

### Normalization

```c
// Layer normalization
struct ggml_tensor * norm_out = ggml_norm(ctx, x);

// RMS normalization
struct ggml_tensor * rms_out = ggml_rms_norm(ctx, x);

// Group normalization
struct ggml_tensor * group_norm_out = ggml_group_norm(ctx, x, num_groups);
```

## Backend Management

### Backend Types

GGML supports multiple compute backends:

```c
// CPU backend (always available)
ggml_backend_t cpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, NULL);

// GPU backend (if available)
ggml_backend_t gpu_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, NULL);

// Automatic selection (prefers GPU if available)
ggml_backend_t best_backend = ggml_backend_init_best();
```

### Memory Management

```c
// Allocate tensors to backend buffer
ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

// Manual allocation for specific buffer types
ggml_backend_buffer_type_t buf_type = ggml_backend_get_default_buffer_type(backend);
ggml_backend_buffer_t manual_buffer = ggml_backend_buft_alloc_buffer(buf_type, size);
```

### Multi-Backend Scheduling

For complex scenarios with multiple backends:

```c
// Create backend scheduler
ggml_backend_t backends[] = {gpu_backend, cpu_backend};
ggml_backend_sched_t sched = ggml_backend_sched_new(backends, NULL, 2, 
                                                    GGML_DEFAULT_GRAPH_SIZE, 
                                                    false, true);

// Use scheduler for computation
ggml_backend_sched_graph_compute(sched, graph);

// Cleanup
ggml_backend_sched_free(sched);
```

## Memory Optimization

### Graph Allocator

For efficient memory usage with multiple graphs:

```c
// Create graph allocator
ggml_gallocr_t gallocr = ggml_gallocr_new(buffer_type);

// Reserve memory for worst-case graph
ggml_gallocr_reserve(gallocr, max_graph);

// Allocate and compute multiple graphs efficiently
for (int i = 0; i < num_iterations; i++) {
    struct ggml_cgraph * graph = build_graph(ctx, batch_data[i]);
    ggml_gallocr_alloc_graph(gallocr, graph);
    ggml_backend_graph_compute(backend, graph);
}

ggml_gallocr_free(gallocr);
```

### Tensor Flags

Control memory allocation behavior:

```c
// Mark as input (allocated at graph start)
ggml_set_input(input_tensor);

// Mark as output (never freed during graph execution)
ggml_set_output(output_tensor);

// Mark as parameter (trainable)
ggml_set_param(weight_tensor);
```

## Utility Functions

### Tensor Information

```c
// Get tensor properties
int64_t num_elements = ggml_nelements(tensor);
size_t bytes = ggml_nbytes(tensor);
int n_dims = ggml_n_dims(tensor);
bool is_contiguous = ggml_is_contiguous(tensor);

// Get tensor shape
printf("Shape: [%lld, %lld, %lld, %lld]\n", 
       tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
```

### Data Access

```c
// Set/get tensor data
ggml_backend_tensor_set(tensor, data, offset, size);
ggml_backend_tensor_get(tensor, data, offset, size);

// For host tensors, direct access is possible
float * tensor_data = ggml_get_data_f32(tensor);
```

### Type Utilities

```c
// Check data type properties
bool is_quantized = ggml_is_quantized(GGML_TYPE_Q4_0);
size_t type_size = ggml_type_size(GGML_TYPE_F32);
const char * type_name = ggml_type_name(GGML_TYPE_F16);
```

## Error Handling

```c
// Check operation status
enum ggml_status status = ggml_backend_graph_compute(backend, graph);
if (status != GGML_STATUS_SUCCESS) {
    printf("Computation failed: %s\n", ggml_status_to_string(status));
}
```

## Best Practices

1. **Memory Management**: Always free contexts, backends, and buffers to prevent memory leaks
2. **Graph Reuse**: Build graphs once and reuse them for better performance
3. **Backend Selection**: Use `ggml_backend_init_best()` for automatic backend selection
4. **Quantization**: Use quantized types for inference to reduce memory usage
5. **In-place Operations**: Use in-place variants when possible to reduce memory pressure
6. **Error Checking**: Always check return values for resource allocation and computation

## Thread Safety

GGML contexts are not thread-safe. For multi-threaded applications:
- Use separate contexts per thread, or
- Implement external synchronization
- Backend operations are generally thread-safe within a single context

## Advanced Features

### Custom Operations

```c
// Define custom operations using the custom operation interface
struct ggml_tensor * custom_op = ggml_map_custom1(ctx, input, custom_func, 
                                                  GGML_N_TASKS_MAX, user_data);
```

### Gradient Computation

```c
// Enable gradient computation
ggml_set_param(ctx, weight);

// Build forward and backward graphs
struct ggml_cgraph * forward_graph = ggml_new_graph(ctx);
struct ggml_cgraph * backward_graph = ggml_graph_dup(ctx, forward_graph);

// Compute gradients
ggml_graph_compute_with_ctx(ctx, backward_graph, n_threads);
```

This documentation covers the essential aspects of using GGML for tensor operations and machine learning applications. For specific backend details and advanced memory management, see the companion documentation files.