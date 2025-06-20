# Context Management API

The Context Management API provides a flexible, content-addressed system for managing chunks of context in llama.cpp. This allows
fine-grained control over what content is loaded in the model's context window, enabling efficient memory management and dynamic
context switching.

## Overview

The context management system uses SHA256 content addressing, similar to Git, where each chunk of content is identified by its hash.
This provides several benefits:

- **Automatic deduplication** - identical content is stored only once
- **Content integrity** - corruption can be detected by hash verification
- **Cache-friendly** - clients can cache chunks by hash
- **No ID management** - no need to track arbitrary IDs

## Key Concepts

### Chunks
A chunk is a piece of content (text) that can be independently managed. Each chunk has:
- **Hash** - SHA256 hash of the content (64 character hex string)
- **Content** - The original text
- **Tokens** - Tokenized representation
- **Position** - Location in the KV cache (`token_start_pos`, `token_end_pos`)
- **Status** - `loaded` (in memory), `saved` (on disk), or `empty`
- **Metadata** - User-defined JSON metadata

### Unified Attention
All chunks share the same sequence ID (0), meaning tokens can attend to any other tokens across all loaded chunks. This differs from
the slot mechanism where each slot is isolated.

### Memory Management
The system automatically handles:
- **Fragmentation** - Compacts memory when fragmentation exceeds threshold
- **Position allocation** - Various strategies for placing chunks
- **Save/restore** - Chunks can be saved to disk and restored later
- **Garbage collection** - LRU eviction and auto-save of old chunks

## API Endpoints

### GET /context
Returns information about all chunks and memory usage.

**Response:**
```json
{
  "total_context": 40960,
  "used_context": 8192,
  "free_context": 32768,
  "fragmentation": 0.15,
  "chunks": [
    {
      "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
      "token_start_pos": 0,
      "token_end_pos": 1000,
      "token_size": 1000,
      "memory_size": 1000,
      "status": "loaded",
      "save_file": "chunk_e3b0c44298fc1c14.kv",
      "content_preview": "This is the beginning of the content...",
      "created_at": "2024-01-15T10:30:00Z",
      "last_accessed": "2024-01-15T10:35:00Z",
      "metadata": {
        "type": "system_prompt",
        "version": "1.0"
      }
    }
  ],
  "gaps": [
    {"start": 1000, "end": 2000, "size": 1000}
  ],
  "content_index": {
    "e3b0c44298fc1c14...": {"type": "system_prompt", "tokens": 1000, "status": "loaded"}
  }
}

POST /context

Adds a new chunk of content to the context.

Request:
{
  "content": "The quick brown fox jumps over the lazy dog",
  "position": "auto",              // auto|best_fit|first_fit|compact_first|specific_pos|after:<hash>|before:<hash>
  "preferred_position": 5000,      // Only used with position="specific_pos"
  "tokenize": true,                // Whether to tokenize the content
  "tokens": [123, 456, 789],       // Pre-tokenized tokens (if tokenize=false)
  "metadata": {                    // User-defined metadata
    "source": "document.txt",
    "section": "introduction"
  }
}

Response:
{
  "hash": "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
  "token_size": 12,
  "memory_size": 12,
  "position": {"start": 1000, "end": 1012},
  "status": "loaded",
  "deduplication": false,
  "tokens": 12
}

POST /context/:hash?action=save

Saves a chunk to disk, freeing memory while preserving the content.

Response:
{
  "success": true,
  "hash": "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
  "action": "save"
}

POST /context/:hash?action=restore

Restores a previously saved chunk back into memory.

Response:
{
  "success": true,
  "hash": "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
  "action": "restore"
}

POST /context/:hash?action=erase

Permanently removes a chunk from both memory and disk.

Response:
{
  "success": true,
  "hash": "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592",
  "action": "erase"
}

POST /context/batch

Performs multiple operations in a single request.

Request:
{
  "operations": [
    {"action": "save", "hash": "abc123..."},
    {"action": "restore", "hash": "def456..."},
    {"action": "erase", "hash": "ghi789..."}
  ],
  "compact_after": true
}

Response:
[
  {"action": "save", "hash": "abc123...", "success": true},
  {"action": "restore", "hash": "def456...", "success": true},
  {"action": "erase", "hash": "ghi789...", "success": false}
]

POST /context/compact

Forces memory compaction to eliminate fragmentation.

Response:
{
  "success": true
}

POST /context/gc

Runs garbage collection to:
- Save chunks that haven't been accessed recently
- Evict excess chunks if memory limit exceeded

Response:
{
  "success": true
}

Position Strategies

When adding a chunk, you can specify how it should be positioned:

- auto (default) - Append at the end of existing content
- best_fit - Find the smallest gap that fits the chunk
- first_fit - Use the first available gap
- compact_first - Compact memory then append
- specific_pos - Place at a specific position
- after:<hash> - Place immediately after another chunk
- before:<hash> - Place immediately before another chunk

Usage Examples

Basic Usage

import requests
import json

# Add a system prompt
response = requests.post('http://localhost:8080/context', json={
    'content': 'You are a helpful AI assistant.',
    'metadata': {'type': 'system_prompt'}
})
system_hash = response.json()['hash']

# Add user context
response = requests.post('http://localhost:8080/context', json={
    'content': 'The user is working on a Python project.',
    'position': f'after:{system_hash}',
    'metadata': {'type': 'user_context'}
})
context_hash = response.json()['hash']

# Check memory usage
response = requests.get('http://localhost:8080/context')
print(f"Used: {response.json()['used_context']} / {response.json()['total_context']}")

# Save context to disk to free memory
requests.post(f'http://localhost:8080/context/{context_hash}?action=save')

# Later, restore it
requests.post(f'http://localhost:8080/context/{context_hash}?action=restore')

Dynamic Context Management

class ContextManager:
    def __init__(self, base_url='http://localhost:8080'):
        self.base_url = base_url
        self.chunks = {}

    def add_chunk(self, content, chunk_type, **kwargs):
        """Add a chunk and track its hash"""
        response = requests.post(f'{self.base_url}/context', json={
            'content': content,
            'metadata': {'type': chunk_type},
            **kwargs
        })
        data = response.json()
        self.chunks[chunk_type] = data['hash']
        return data['hash']

    def swap_context(self, old_type, new_content, new_type):
        """Replace one context chunk with another"""
        # Save old chunk
        if old_type in self.chunks:
            requests.post(f'{self.base_url}/context/{self.chunks[old_type]}?action=save')

        # Add new chunk in its place
        return self.add_chunk(new_content, new_type, position='compact_first')

    def get_memory_info(self):
        """Get current memory usage"""
        response = requests.get(f'{self.base_url}/context')
        return response.json()

# Usage
ctx = ContextManager()

# Add base context
ctx.add_chunk("You are an AI assistant.", "system")
ctx.add_chunk("User prefers concise answers.", "preferences")

# Swap context based on task
ctx.swap_context("preferences", "User wants detailed technical explanations.", "preferences")

Content Deduplication

# Adding the same content multiple times returns the same hash
content = "Important context that might be added multiple times"

hash1 = requests.post('http://localhost:8080/context',
                     json={'content': content}).json()['hash']

hash2 = requests.post('http://localhost:8080/context',
                     json={'content': content}).json()['hash']

assert hash1 == hash2  # Same content = same hash

Batch Operations

# Efficiently reorganize context
requests.post('http://localhost:8080/context/batch', json={
    'operations': [
        {'action': 'save', 'hash': old_context_hash},
        {'action': 'restore', 'hash': relevant_context_hash},
        {'action': 'erase', 'hash': outdated_context_hash}
    ],
    'compact_after': True
})

