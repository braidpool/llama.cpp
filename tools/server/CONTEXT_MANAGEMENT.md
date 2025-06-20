# Context Management API Implementation

This document describes the Context Management API implementation that was added to llama.cpp server.

## Overview

The Context Management API provides a flexible, content-addressed system for managing chunks of context in llama.cpp. It allows fine-grained control over what content is loaded in the model's context window, enabling efficient memory management and dynamic context switching.

## Implementation Status

✅ **COMPLETED** - Full implementation with all planned features:

- **Content-addressed storage** using SHA256 hashes
- **Automatic deduplication** of identical content
- **Multiple positioning strategies** for chunk placement
- **Save/restore operations** for memory management
- **Memory compaction** to reduce fragmentation
- **Batch operations** for efficiency
- **Comprehensive API endpoints** with full functionality
- **Extensive test suite** and Python examples

## Files Added/Modified

### Core Implementation
- `tools/server/context-manager.hpp` - Complete context management implementation
- `tools/server/server.cpp` - API endpoints integration (via patch)
- `docs/context.md` - API documentation

### Tests and Examples
- `tools/server/tests/unit/test_context_management.py` - Comprehensive test suite
- `tools/server/test_context_api.py` - Simple verification script
- `llama_context.py` - Full-featured Python example and library

### Planning Documents
- `context_claude_plan.txt` - Original implementation plan

## API Endpoints

The implementation includes all planned endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/context` | GET | Get context information and statistics |
| `/context` | POST | Add new content chunk |
| `/context/:hash` | POST | Perform action on chunk (save/restore/erase) |
| `/context/batch` | POST | Perform multiple operations |
| `/context/compact` | POST | Force memory compaction |
| `/context/gc` | POST | Run garbage collection |

## Key Features Implemented

### 1. SHA256 Content Addressing
- Automatic content deduplication
- Content integrity verification
- Git-like content management
- No manual ID management required

### 2. Flexible Positioning
- `auto` - Append at end (default)
- `best_fit` - Find smallest suitable gap
- `first_fit` - Use first available gap
- `compact_first` - Compact then append
- `specific_pos` - Place at specific position
- `after:<hash>` - Position after another chunk
- `before:<hash>` - Position before another chunk

### 3. Memory Management
- **Save/Restore**: Move chunks between memory and disk
- **Compaction**: Eliminate fragmentation by rebuilding sequence
- **Garbage Collection**: Auto-save old chunks based on access time
- **Fragmentation Tracking**: Monitor and optimize memory usage

### 4. Unified Attention
- All chunks use sequence ID 0 for cross-chunk attention
- Maintains llama.cpp's attention mechanism
- Enables complex multi-document reasoning

## Testing

The implementation includes comprehensive testing:

### Quick Verification
```bash
# Start server with context management
./llama-server -m <model_path>

# Run simple verification
python3 tools/server/test_context_api.py
```

### Full Test Suite
```bash
# Run comprehensive tests
python3 -m pytest tools/server/tests/unit/test_context_management.py -v
```

### Python Example
```bash
# Run feature demonstration
python3 llama_context.py
```

## Usage Examples

### Basic Usage
```python
import requests

# Add content
response = requests.post('http://localhost:8080/context', json={
    'content': 'Your text content here',
    'metadata': {'type': 'user_input'}
})
chunk_hash = response.json()['hash']

# Save to disk
requests.post(f'http://localhost:8080/context/{chunk_hash}?action=save')

# Restore to memory
requests.post(f'http://localhost:8080/context/{chunk_hash}?action=restore')
```

### Advanced Context Management
```python
from llama_context import LlamaContextManager, ConversationManager

# High-level API
ctx = LlamaContextManager()
conv = ConversationManager(ctx)

# Set up conversation modes
conv.set_system_prompt("You are a helpful assistant.", "general")
conv.set_system_prompt("You are a code expert.", "programming")

# Switch between modes
conv.switch_mode("programming")
conv.add_user_message("How do I implement quicksort?")
```

## Architecture Details

### ChunkManager Class
- **Thread-safe** with shared_mutex for concurrent access
- **Position tracking** via start/end positions
- **Gap management** for efficient space utilization
- **Automatic compaction** when fragmentation exceeds threshold

### ContentAddressing Class
- **Modern OpenSSL EVP API** for SHA256 computation
- **Hash validation** and format checking
- **Git-like file organization** (chunks/xx/chunk_hash.kv)

### ChunkStorage Class
- **Metadata + KV data** storage format
- **Atomic save/restore** operations
- **Directory structure** management

## Integration with llama.cpp

The implementation uses modern llama.cpp APIs:

- `llama_state_get_size()` / `llama_state_get_data()` for state save
- `llama_state_set_data()` for state restore  
- `llama_memory_seq_rm()` for sequence removal
- `llama_batch` system for token encoding
- `llama_get_memory()` for memory operations

## Performance Considerations

### Memory Efficiency
- **Lazy loading**: Only active chunks in memory
- **Automatic compaction**: Reduces fragmentation overhead
- **LRU eviction**: Manages memory pressure

### I/O Optimization
- **Batch operations**: Minimize API roundtrips
- **Binary storage**: Efficient KV cache serialization
- **Directory sharding**: Scales to many chunks

## Security and Reliability

### Content Integrity
- **SHA256 verification**: Detect corruption
- **Atomic operations**: Prevent partial state
- **Error handling**: Graceful failure recovery

### Resource Management
- **Memory limits**: Configurable thresholds
- **Timeout handling**: Prevents hanging operations
- **Cleanup on errors**: No resource leaks

## Future Enhancements

The current implementation provides a solid foundation for:

- **Search functionality**: Content-based chunk discovery
- **Versioning**: Git-like content history
- **Streaming**: Large document processing
- **Replication**: Multi-node context sharing
- **Analytics**: Usage pattern analysis

## Build Requirements

The implementation requires:
- **OpenSSL**: For SHA256 computation
- **C++17**: For filesystem and shared_mutex
- **llama.cpp**: Latest version with memory API

## Conclusion

This implementation successfully delivers all planned features for the Context Management API. It provides a robust, efficient, and user-friendly system for managing LLM context that integrates seamlessly with llama.cpp's existing architecture.

The comprehensive test suite and examples demonstrate the API's capabilities and provide a foundation for integration into larger applications requiring sophisticated context management.