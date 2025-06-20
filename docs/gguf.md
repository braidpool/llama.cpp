# GGUF File Format Library

GGUF (GGML Universal Format) is a binary file format designed for storing and loading machine learning models and their associated metadata. It provides efficient serialization of tensors along with key-value metadata.

## Overview

GGUF provides:
- Efficient binary storage for tensors and metadata
- Support for various data types and arrays
- Flexible key-value metadata system
- Memory-efficient loading with optional tensor allocation
- Cross-platform compatibility

## File Format Structure

GGUF files have the following binary structure:

```
1. File magic "GGUF" (4 bytes)
2. File version (uint32_t)
3. Number of tensors (int64_t)
4. Number of key-value pairs (int64_t)
5. Key-value pairs:
   - Key (string: length + data)
   - Value type (gguf_type as int32_t)
   - Value data (type-specific)
6. Tensor metadata:
   - Name (string)
   - Number of dimensions (uint32_t)
   - Dimension sizes (int64_t array)
   - Data type (ggml_type as int32_t)
   - Data offset (uint64_t)
7. Tensor data (optional, aligned)
```

## Getting Started

### Include Header

```c
#include "gguf.h"
#include "ggml.h"  // Often needed for tensor operations
```

### Basic Reading

```c
// Open and read a GGUF file
struct gguf_init_params params = {
    .no_alloc = false,  // Allocate tensor data
    .ctx = NULL         // Will create ggml_context automatically
};

struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);
if (!gguf_ctx) {
    fprintf(stderr, "Failed to load GGUF file\n");
    return -1;
}

// Get basic information
uint32_t version = gguf_get_version(gguf_ctx);
int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
int64_t n_kv = gguf_get_n_kv(gguf_ctx);

printf("GGUF version: %u\n", version);
printf("Number of tensors: %lld\n", n_tensors);
printf("Number of KV pairs: %lld\n", n_kv);

// Cleanup
gguf_free(gguf_ctx);
```

### Memory-Efficient Reading

```c
// Read only metadata, no tensor allocation
struct ggml_context * model_ctx = NULL;
struct gguf_init_params params = {
    .no_alloc = true,   // Don't allocate tensor data
    .ctx = &model_ctx   // Get the ggml context
};

struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);

// Manually allocate tensors as needed
// ... (see tensor access section)

gguf_free(gguf_ctx);
ggml_free(model_ctx);
```

## Data Types

GGUF supports the following data types for metadata:

```c
enum gguf_type {
    GGUF_TYPE_UINT8,    // unsigned 8-bit integer
    GGUF_TYPE_INT8,     // signed 8-bit integer
    GGUF_TYPE_UINT16,   // unsigned 16-bit integer
    GGUF_TYPE_INT16,    // signed 16-bit integer
    GGUF_TYPE_UINT32,   // unsigned 32-bit integer
    GGUF_TYPE_INT32,    // signed 32-bit integer
    GGUF_TYPE_FLOAT32,  // 32-bit float
    GGUF_TYPE_BOOL,     // boolean (stored as int8)
    GGUF_TYPE_STRING,   // UTF-8 string
    GGUF_TYPE_ARRAY,    // array of other types
    GGUF_TYPE_UINT64,   // unsigned 64-bit integer
    GGUF_TYPE_INT64,    // signed 64-bit integer
    GGUF_TYPE_FLOAT64   // 64-bit double
};
```

## Working with Metadata

### Reading Key-Value Pairs

```c
// Iterate through all KV pairs
int64_t n_kv = gguf_get_n_kv(gguf_ctx);
for (int64_t i = 0; i < n_kv; i++) {
    const char * key = gguf_get_key(gguf_ctx, i);
    enum gguf_type type = gguf_get_kv_type(gguf_ctx, i);
    
    printf("Key: %s, Type: %s\n", key, gguf_type_name(type));
    
    // Read value based on type
    switch (type) {
        case GGUF_TYPE_STRING: {
            const char * str_val = gguf_get_val_str(gguf_ctx, i);
            printf("  Value: %s\n", str_val);
            break;
        }
        case GGUF_TYPE_INT32: {
            int32_t int_val = gguf_get_val_i32(gguf_ctx, i);
            printf("  Value: %d\n", int_val);
            break;
        }
        case GGUF_TYPE_FLOAT32: {
            float float_val = gguf_get_val_f32(gguf_ctx, i);
            printf("  Value: %f\n", float_val);
            break;
        }
        // ... handle other types
    }
}
```

### Finding Specific Keys

```c
// Find a specific key
int64_t key_id = gguf_find_key(gguf_ctx, "model.name");
if (key_id >= 0) {
    const char * model_name = gguf_get_val_str(gguf_ctx, key_id);
    printf("Model name: %s\n", model_name);
} else {
    printf("Model name not found\n");
}

// Common model metadata
key_id = gguf_find_key(gguf_ctx, "general.name");
if (key_id >= 0) {
    const char * general_name = gguf_get_val_str(gguf_ctx, key_id);
    printf("General name: %s\n", general_name);
}
```

### Working with Arrays

```c
// Read array metadata
int64_t array_key = gguf_find_key(gguf_ctx, "some.array");
if (array_key >= 0 && gguf_get_kv_type(gguf_ctx, array_key) == GGUF_TYPE_ARRAY) {
    enum gguf_type arr_type = gguf_get_arr_type(gguf_ctx, array_key);
    size_t arr_n = gguf_get_arr_n(gguf_ctx, array_key);
    
    printf("Array type: %s, length: %zu\n", gguf_type_name(arr_type), arr_n);
    
    // Access array data
    const void * arr_data = gguf_get_arr_data(gguf_ctx, array_key);
    
    // For string arrays
    if (arr_type == GGUF_TYPE_STRING) {
        for (size_t i = 0; i < arr_n; i++) {
            const char * str = gguf_get_arr_str(gguf_ctx, array_key, i);
            printf("  [%zu]: %s\n", i, str);
        }
    }
    // For numeric arrays, cast arr_data to appropriate type
    else if (arr_type == GGUF_TYPE_FLOAT32) {
        const float * float_array = (const float *)arr_data;
        for (size_t i = 0; i < arr_n; i++) {
            printf("  [%zu]: %f\n", i, float_array[i]);
        }
    }
}
```

## Working with Tensors

### Reading Tensor Metadata

```c
int64_t n_tensors = gguf_get_n_tensors(gguf_ctx);
for (int64_t i = 0; i < n_tensors; i++) {
    const char * name = gguf_get_tensor_name(gguf_ctx, i);
    enum ggml_type type = gguf_get_tensor_type(gguf_ctx, i);
    size_t size = gguf_get_tensor_size(gguf_ctx, i);
    size_t offset = gguf_get_tensor_offset(gguf_ctx, i);
    
    printf("Tensor: %s\n", name);
    printf("  Type: %s\n", ggml_type_name(type));
    printf("  Size: %zu bytes\n", size);
    printf("  Offset: %zu\n", offset);
}
```

### Finding Specific Tensors

```c
// Find a tensor by name
int64_t tensor_id = gguf_find_tensor(gguf_ctx, "model.embed_tokens.weight");
if (tensor_id >= 0) {
    enum ggml_type type = gguf_get_tensor_type(gguf_ctx, tensor_id);
    size_t size = gguf_get_tensor_size(gguf_ctx, tensor_id);
    printf("Found tensor: type=%s, size=%zu\n", ggml_type_name(type), size);
}
```

### Loading Tensor Data

When using `no_alloc = false`, tensors are automatically loaded:

```c
struct gguf_init_params params = {.no_alloc = false, .ctx = NULL};
struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);

// Tensors are accessible through the ggml context
struct ggml_context * model_ctx = *params.ctx;
struct ggml_tensor * tensor = ggml_get_tensor(model_ctx, "model.embed_tokens.weight");
if (tensor) {
    // Tensor data is already loaded and accessible
    float * data = ggml_get_data_f32(tensor);
    // Use tensor data...
}
```

## Writing GGUF Files

### Creating a New GGUF Context

```c
// Create empty GGUF context
struct gguf_context * gguf_ctx = gguf_init_empty();

// Set metadata
gguf_set_val_str(gguf_ctx, "general.name", "MyModel");
gguf_set_val_str(gguf_ctx, "general.description", "A custom model");
gguf_set_val_u32(gguf_ctx, "general.version", 1);
gguf_set_val_f32(gguf_ctx, "training.learning_rate", 0.001f);

// Set arrays
const char * tokens[] = {"hello", "world", "!"};
gguf_set_arr_str(gguf_ctx, "tokenizer.tokens", tokens, 3);

float weights[] = {0.1f, 0.2f, 0.3f, 0.4f};
gguf_set_arr_data(gguf_ctx, "model.weights", GGUF_TYPE_FLOAT32, weights, 4);
```

### Adding Tensors

```c
// Create a ggml context for tensors
struct ggml_init_params init_params = {
    .mem_size = 1024 * 1024,  // 1MB
    .mem_buffer = NULL,
    .no_alloc = false
};
struct ggml_context * model_ctx = ggml_init(init_params);

// Create and populate tensors
struct ggml_tensor * weights = ggml_new_tensor_2d(model_ctx, GGML_TYPE_F32, 512, 1024);
ggml_set_name(weights, "model.layers.0.weight");

// Set tensor data
float * weight_data = ggml_get_data_f32(weights);
// ... populate weight_data ...

// Add tensor to GGUF
gguf_add_tensor(gguf_ctx, weights);
```

### Writing to File

```c
// Method 1: Write everything at once
bool success = gguf_write_to_file(gguf_ctx, "output.gguf", false);
if (!success) {
    fprintf(stderr, "Failed to write GGUF file\n");
}

// Method 2: Write metadata first, then tensor data
gguf_write_to_file(gguf_ctx, "output.gguf", true);  // metadata only

FILE * file = fopen("output.gguf", "ab");
// Write tensor data manually
fclose(file);

// Method 3: Prepare data in memory first
size_t meta_size = gguf_get_meta_size(gguf_ctx);
void * meta_data = malloc(meta_size);
gguf_get_meta_data(gguf_ctx, meta_data);

FILE * file = fopen("output.gguf", "wb");
fwrite(meta_data, 1, meta_size, file);
// Write tensor data...
fclose(file);
free(meta_data);
```

## Advanced Features

### Custom Alignment

```c
// Set custom alignment for tensor data
gguf_set_val_u32(gguf_ctx, "general.alignment", 64);  // 64-byte alignment
```

### Modifying Existing Files

```c
// Load existing file
struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);

// Remove a key
int64_t removed_id = gguf_remove_key(gguf_ctx, "old.parameter");

// Update values
gguf_set_val_str(gguf_ctx, "general.version", "2.0");

// Change tensor type (updates offsets automatically)
gguf_set_tensor_type(gguf_ctx, "model.weight", GGML_TYPE_Q4_0);

// Write back to file
gguf_write_to_file(gguf_ctx, "model_updated.gguf", false);
```

### Copying Metadata

```c
struct gguf_context * src_ctx = gguf_init_from_file("source.gguf", params);
struct gguf_context * dst_ctx = gguf_init_empty();

// Copy all KV pairs from source to destination
gguf_set_kv(dst_ctx, src_ctx);

gguf_free(src_ctx);
gguf_free(dst_ctx);
```

## Utility Functions

### Type Information

```c
// Get type name for display
const char * type_name = gguf_type_name(GGUF_TYPE_FLOAT32);  // Returns "f32"

// Get file information
size_t alignment = gguf_get_alignment(gguf_ctx);
size_t data_offset = gguf_get_data_offset(gguf_ctx);
```

### File Validation

```c
// Basic validation
if (gguf_get_version(gguf_ctx) != GGUF_VERSION) {
    fprintf(stderr, "Unsupported GGUF version\n");
}

// Check for required metadata
if (gguf_find_key(gguf_ctx, "general.name") < 0) {
    fprintf(stderr, "Missing required metadata: general.name\n");
}
```

## Error Handling

```c
// Always check for NULL returns
struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);
if (!gguf_ctx) {
    fprintf(stderr, "Failed to load GGUF file\n");
    return -1;
}

// Check key existence before access
int64_t key_id = gguf_find_key(gguf_ctx, "some.key");
if (key_id < 0) {
    fprintf(stderr, "Key not found: some.key\n");
} else {
    // Safe to access
    const char * value = gguf_get_val_str(gguf_ctx, key_id);
}
```

## Best Practices

1. **Memory Management**: Always call `gguf_free()` to prevent memory leaks
2. **Key Validation**: Use `gguf_find_key()` before accessing values
3. **Type Checking**: Verify types with `gguf_get_kv_type()` before casting
4. **Alignment**: Use proper alignment for optimal performance
5. **Version Checking**: Always check GGUF version compatibility
6. **Error Handling**: Check return values and handle errors gracefully

## Common Use Cases

### Loading a Pre-trained Model

```c
struct gguf_init_params params = {.no_alloc = false, .ctx = NULL};
struct gguf_context * gguf_ctx = gguf_init_from_file("model.gguf", params);

// Read model metadata
const char * model_name = gguf_get_val_str(gguf_ctx, 
    gguf_find_key(gguf_ctx, "general.name"));
int64_t vocab_size = gguf_get_val_i64(gguf_ctx, 
    gguf_find_key(gguf_ctx, "model.vocab_size"));

// Access model tensors
struct ggml_context * model_ctx = *params.ctx;
struct ggml_tensor * embeddings = ggml_get_tensor(model_ctx, "model.embed_tokens.weight");

// Use model for inference...
```

### Converting Between Formats

```c
// Read from one format
struct gguf_context * src = gguf_init_from_file("input.gguf", params);

// Create new GGUF with modified metadata
struct gguf_context * dst = gguf_init_empty();
gguf_set_kv(dst, src);  // Copy metadata
gguf_set_val_str(dst, "converted.tool", "my_converter");

// Copy tensors with potential type conversion
// ... tensor processing ...

// Write new format
gguf_write_to_file(dst, "output.gguf", false);
```

GGUF provides a robust and efficient way to store and load machine learning models with comprehensive metadata support. It's designed to be both human-readable (in terms of metadata) and optimized for fast loading and minimal memory usage.