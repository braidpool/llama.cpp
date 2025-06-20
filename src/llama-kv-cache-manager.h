#pragma once

#include "llama.h"
#include "llama-memory.h"
#include "llama-batch.h"

#include <nlohmann/json.hpp>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::ordered_json;
using llama_tokens = std::vector<llama_token>;

struct llama_context;
struct llama_model;
class llama_kv_cache_unified;

//
// Minimal KV Cache Manager - Simple content tracking wrapper
// 
// This is a minimal wrapper around llama_kv_cache_unified that:
// - Tracks content by hash
// - Manages active/inactive state via sequence IDs
// - Provides save/restore to system memory
// - Does NOT duplicate any KV cache state
//

enum class llama_chunk_status {
    ACTIVE,    // Using ACTIVE_SEQ_ID, visible to inference
    INACTIVE,  // Using INACTIVE_SEQ_ID, hidden from inference  
    SYSTEM,    // Offloaded to system RAM
    EMPTY      // Deleted/placeholder state
};

struct llama_chunk_info {
    std::string hash;                   // SHA256 of content
    std::string content;                // Original text
    llama_tokens tokens;                // Tokenized content
    llama_seq_id seq_id = -1;          // Current sequence ID
    llama_chunk_status status = llama_chunk_status::EMPTY;
    std::vector<uint8_t> system_cache;  // KV data when offloaded
    json metadata;                      // User metadata
    llama_pos kv_start_pos = -1;       // Start position in KV cache
    llama_pos kv_end_pos = -1;         // End position in KV cache
    
    json to_json() const;
};

struct llama_chunk_result {
    bool success = false;
    std::string hash;
    std::string error_message;
    
    static llama_chunk_result ok(const std::string & chunk_hash) {
        return {true, chunk_hash, ""};
    }
    
    static llama_chunk_result error(const std::string & message) {
        return {false, "", message};
    }
};

struct llama_chunk_options {
    bool tokenize = true;
    llama_tokens tokens;  // Pre-tokenized if tokenize=false
    json metadata;
};

class llama_kv_cache_manager : public llama_memory_i {
public:
    // Constructor - takes existing unified cache
    llama_kv_cache_manager(llama_kv_cache_unified * unified_cache);
    ~llama_kv_cache_manager() = default;

    // Set context info (called by server after construction)
    void set_context_info(const llama_model * model, llama_context * ctx);

    //
    // llama_memory_i interface - delegate to unified cache
    //
    llama_memory_state_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_state_ptr init_full() override;
    llama_memory_state_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;
    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    size_t get_memory_size() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1)       override;

    //
    // Chunk management API (required by server)
    //
    llama_chunk_result add_chunk_with_result(const std::string & content, const llama_chunk_options & opts = {});
    
    bool save_chunk(const std::string & hash);      // Save to disk (stub for now)
    bool restore_chunk(const std::string & hash);   // Restore from disk (stub for now)
    bool erase_chunk(const std::string & hash);     // Remove chunk completely
    
    bool activate_chunk(const std::string & hash);   // Move to ACTIVE_SEQ_ID
    bool deactivate_chunk(const std::string & hash); // Move to INACTIVE_SEQ_ID
    bool unload_chunk(const std::string & hash);     // Move to system RAM
    
    void compact();          // Delegate to unified cache
    void garbage_collect();  // Clean up empty chunks
    
    // Query methods
    json get_context_info() const;
    json get_chunk_info(const std::string & hash) const;
    std::string get_chunk_content(const std::string & hash) const;
    
    // Slicing support
    struct SliceResult {
        bool success;
        std::vector<std::string> new_chunk_hashes;
        std::string error_message;
        
        static SliceResult ok(const std::vector<std::string> & hashes) {
            return {true, hashes, ""};
        }
        
        static SliceResult error(const std::string & message) {
            return {false, {}, message};
        }
    };
    
    SliceResult slice_chunk(const std::string & chunk_hash, const std::vector<size_t> & slice_positions);
    
    // Debug
    void audit_kv_cache_state() const;

private:
    // Fixed sequence IDs for activation control
    static constexpr llama_seq_id ACTIVE_SEQ_ID = 0;
    static constexpr llama_seq_id INACTIVE_SEQ_ID = 1;
    
    // Reference to unified cache (no ownership)
    llama_kv_cache_unified * unified_cache;
    
    // Context info
    const llama_model * model = nullptr;
    llama_context * ctx = nullptr;
    
    // Simple chunk tracking
    mutable std::shared_mutex chunks_mutex;
    std::unordered_map<std::string, llama_chunk_info> chunks;
    
    // Helper methods
    bool tokenize_content(const std::string & content, std::vector<llama_token> & tokens);
    bool populate_kv_cache_with_tokens(const std::vector<llama_token> & tokens, llama_seq_id seq_id);
};

// Utility class for content hashing
class llama_content_addressing {
public:
    static std::string compute_hash(const std::string & content);
    static bool validate_hash(const std::string & hash);
};