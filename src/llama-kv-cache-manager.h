#pragma once

#include "llama.h"
#include "llama-memory.h"

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;

struct llama_context;
struct llama_model;
struct llama_vocab;

// Type definitions
using llama_tokens = std::vector<llama_token>;

//
// KV Cache Manager - Content-addressed chunk management for llama.cpp
//

enum class llama_chunk_status {
    LOADED,
    SAVED,
    EMPTY
};

enum class llama_position_strategy {
    APPEND,           // Add at end (fastest)
    BEST_FIT,         // Find smallest suitable gap
    FIRST_FIT,        // Use first available gap
    COMPACT_FIRST,    // Compact then append
    SPECIFIC_POS,     // Use specific position
    AFTER_CHUNK,      // Position after another chunk
    BEFORE_CHUNK      // Position before another chunk
};

struct llama_chunk_options {
    llama_position_strategy position = llama_position_strategy::APPEND;
    llama_pos preferred_position = -1;
    std::string relative_to_hash;  // For AFTER_CHUNK/BEFORE_CHUNK
    bool tokenize = true;
    llama_tokens tokens;  // Pre-tokenized tokens if tokenize=false
    json metadata;
};

struct llama_chunk_info {
    std::string hash;                   // SHA256 of original content
    std::string content;                // Original text content
    llama_tokens tokens;                // Tokenized content
    llama_seq_id seq_id;                // Sequence ID in KV cache
    llama_pos start_pos = -1;          // Position in KV cache
    llama_pos end_pos = -1;            // End position in KV cache
    llama_chunk_status status = llama_chunk_status::EMPTY;
    std::string save_file;             // Disk file path
    json metadata;                     // User metadata
    std::chrono::time_point<std::chrono::system_clock> created_at;
    std::chrono::time_point<std::chrono::system_clock> last_accessed;

    json to_json() const;
};

// Forward declaration
class llama_kv_cache_manager_impl;

class llama_kv_cache_manager {
public:
    explicit llama_kv_cache_manager(llama_context * ctx, const std::string & storage_path = "");
    ~llama_kv_cache_manager();

    // Chunk management
    std::string add_chunk(const std::string & content, const llama_chunk_options & opts = {});
    bool save_chunk(const std::string & hash);
    bool restore_chunk(const std::string & hash);
    bool erase_chunk(const std::string & hash);

    // Batch operations
    json batch_operations(const json & request_body);

    // Memory management
    void compact();
    void garbage_collect();

    // Queries
    json get_context_info() const;
    json get_chunk_info(const std::string & hash) const;
    std::string get_chunk_content(const std::string & hash) const;
    json search_chunks(const json & criteria) const;

    // Configuration
    void set_fragmentation_threshold(float threshold);
    void set_auto_compact_enabled(bool enabled);
    void set_max_memory_chunks(size_t max_chunks);

private:
    std::unique_ptr<llama_kv_cache_manager_impl> impl;
};

// Utility functions
class llama_content_addressing {
public:
    static std::string compute_hash(const std::string & content);
    static std::string generate_filename(const std::string & hash);
    static bool validate_hash(const std::string & hash);
};