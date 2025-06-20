#include "llama-kv-cache-manager.h"
#include "llama-kv-cache-unified.h"
#include "llama-io.h"
#include "llama-context.h"
#include "llama-vocab.h"
#include "llama-model.h"
#include "llama-impl.h"
// Remove common.h - not available in library code

#include <openssl/evp.h>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cassert>

//
// llama_chunk_info
//

json llama_chunk_info::to_json() const {
    json result = {
        {"hash", hash},
        {"status", status == llama_chunk_status::ACTIVE ? "active" :
                   status == llama_chunk_status::INACTIVE ? "inactive" :
                   status == llama_chunk_status::SYSTEM ? "system" : "empty"},
        {"token_count", tokens.size()},
        {"content_length", content.length()},
        {"metadata", metadata}
    };

    if (kv_start_pos >= 0 && kv_end_pos >= 0) {
        result["kv_start_pos"] = kv_start_pos;
        result["kv_end_pos"] = kv_end_pos;
    }

    if (status == llama_chunk_status::SYSTEM) {
        result["system_cache_size"] = system_cache.size();
    }

    return result;
}

//
// llama_content_addressing
//

std::string llama_content_addressing::compute_hash(const std::string & content) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len;

    EVP_MD_CTX * ctx = EVP_MD_CTX_new();
    if (!ctx) return "";

    if (EVP_DigestInit_ex(ctx, EVP_sha256(), nullptr) != 1 ||
        EVP_DigestUpdate(ctx, content.data(), content.size()) != 1 ||
        EVP_DigestFinal_ex(ctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return "";
    }

    EVP_MD_CTX_free(ctx);

    std::stringstream ss;
    for (unsigned int i = 0; i < hash_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }

    return ss.str();
}

bool llama_content_addressing::validate_hash(const std::string & hash) {
    if (hash.length() != 64) return false;
    return std::all_of(hash.begin(), hash.end(),
                      [](char c) { return std::isxdigit(c); });
}

//
// llama_kv_cache_manager
//

llama_kv_cache_manager::llama_kv_cache_manager(llama_kv_cache_unified * unified_cache)
    : unified_cache(unified_cache) {
    LLAMA_LOG_INFO("%s: minimal chunk manager initialized\n", __func__);
}

void llama_kv_cache_manager::set_context_info(const llama_model * model, llama_context * ctx) {
    this->model = model;
    this->ctx = ctx;
    LLAMA_LOG_INFO("%s: context info set\n", __func__);
}

//
// llama_memory_i interface - all delegated to unified cache
//

llama_memory_state_ptr llama_kv_cache_manager::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    return unified_cache->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_state_ptr llama_kv_cache_manager::init_full() {
    return unified_cache->init_full();
}

llama_memory_state_ptr llama_kv_cache_manager::init_update(llama_context * lctx, bool optimize) {
    return unified_cache->init_update(lctx, optimize);
}

bool llama_kv_cache_manager::get_can_shift() const {
    return unified_cache->get_can_shift();
}

void llama_kv_cache_manager::clear(bool data) {
    unified_cache->clear(data);

    // Also clear our chunk tracking
    std::unique_lock lock(chunks_mutex);
    chunks.clear();
}

bool llama_kv_cache_manager::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    return unified_cache->seq_rm(seq_id, p0, p1);
}

void llama_kv_cache_manager::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    unified_cache->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_kv_cache_manager::seq_keep(llama_seq_id seq_id) {
    unified_cache->seq_keep(seq_id);
}

void llama_kv_cache_manager::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    unified_cache->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_manager::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    unified_cache->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_manager::seq_pos_min(llama_seq_id seq_id) const {
    return unified_cache->seq_pos_min(seq_id);
}

llama_pos llama_kv_cache_manager::seq_pos_max(llama_seq_id seq_id) const {
    return unified_cache->seq_pos_max(seq_id);
}

size_t llama_kv_cache_manager::get_memory_size() const {
    return unified_cache->get_memory_size();
}

void llama_kv_cache_manager::state_write(llama_io_write_i & io, llama_seq_id seq_id) const {
    // Write unified cache state
    unified_cache->state_write(io, seq_id);

    // Write chunk info
    std::shared_lock lock(chunks_mutex);
    uint32_t chunk_count = chunks.size();
    io.write(&chunk_count, sizeof(chunk_count));

    for (const auto & [hash, chunk] : chunks) {
        uint32_t hash_len = hash.length();
        io.write(&hash_len, sizeof(hash_len));
        io.write(hash.data(), hash_len);

        uint32_t content_len = chunk.content.length();
        io.write(&content_len, sizeof(content_len));
        io.write(chunk.content.data(), content_len);

        uint32_t status_int = static_cast<uint32_t>(chunk.status);
        io.write(&status_int, sizeof(status_int));

        io.write(&chunk.seq_id, sizeof(chunk.seq_id));

        // Write metadata as JSON string
        std::string meta_str = chunk.metadata.dump();
        uint32_t meta_len = meta_str.length();
        io.write(&meta_len, sizeof(meta_len));
        io.write(meta_str.data(), meta_len);
    }
}

void llama_kv_cache_manager::state_read(llama_io_read_i & io, llama_seq_id seq_id) {
    // Read unified cache state
    unified_cache->state_read(io, seq_id);

    // Read chunk info
    std::unique_lock lock(chunks_mutex);
    chunks.clear();

    uint32_t chunk_count;
    io.read_to(&chunk_count, sizeof(chunk_count));

    for (uint32_t i = 0; i < chunk_count; i++) {
        uint32_t hash_len;
        io.read_to(&hash_len, sizeof(hash_len));
        std::string hash(hash_len, '\0');
        io.read_to(hash.data(), hash_len);

        uint32_t content_len;
        io.read_to(&content_len, sizeof(content_len));
        std::string content(content_len, '\0');
        io.read_to(content.data(), content_len);

        llama_chunk_info chunk;
        chunk.hash = hash;
        chunk.content = content;

        uint32_t status_int;
        io.read_to(&status_int, sizeof(status_int));
        chunk.status = static_cast<llama_chunk_status>(status_int);

        io.read_to(&chunk.seq_id, sizeof(chunk.seq_id));

        // Read metadata
        uint32_t meta_len;
        io.read_to(&meta_len, sizeof(meta_len));
        if (meta_len > 0) {
            std::string meta_str(meta_len, '\0');
            io.read_to(meta_str.data(), meta_len);
            chunk.metadata = json::parse(meta_str);
        }

        // Re-tokenize content
        tokenize_content(content, chunk.tokens);

        chunks[hash] = std::move(chunk);
    }
}

//
// Chunk management implementation
//

llama_chunk_result llama_kv_cache_manager::add_chunk_with_result(const std::string & content, const llama_chunk_options & opts) {
    // Compute hash
    std::string hash = llama_content_addressing::compute_hash(content);

    // Check if already exists
    {
        std::shared_lock lock(chunks_mutex);
        if (chunks.find(hash) != chunks.end()) {
            return llama_chunk_result::ok(hash);
        }
    }

    // Prepare tokens
    std::vector<llama_token> tokens;
    if (opts.tokenize) {
        if (!tokenize_content(content, tokens)) {
            return llama_chunk_result::error("Failed to tokenize content");
        }
    } else {
        tokens = opts.tokens;
    }

    if (tokens.empty()) {
        return llama_chunk_result::error("No tokens in content");
    }

    // Only populate KV cache if this is not an inference output
    // Inference outputs are already in the KV cache from generation
    bool is_inference_output = opts.metadata.contains("type") &&
                              opts.metadata["type"] == "inference_output";

    if (!is_inference_output) {
        // Try to add tokens to KV cache using active sequence
        if (!populate_kv_cache_with_tokens(tokens, ACTIVE_SEQ_ID)) {
            return llama_chunk_result::error("Failed to populate KV cache");
        }
    }

    // Create and store chunk info
    std::unique_lock lock(chunks_mutex);

    llama_chunk_info chunk;
    chunk.hash = hash;
    chunk.content = content;
    chunk.tokens = tokens;
    chunk.seq_id = ACTIVE_SEQ_ID;
    chunk.status = llama_chunk_status::ACTIVE;
    chunk.metadata = opts.metadata;

    // Extract KV cache position info from metadata if available
    if (opts.metadata.contains("kv_cache_start_pos")) {
        chunk.kv_start_pos = opts.metadata["kv_cache_start_pos"];
    }
    if (opts.metadata.contains("kv_cache_end_pos")) {
        chunk.kv_end_pos = opts.metadata["kv_cache_end_pos"];
    }

    chunks[hash] = std::move(chunk);

    LLAMA_LOG_INFO("%s: added chunk %s (%zu tokens)\n", __func__, hash.substr(0, 8).c_str(), tokens.size());

    return llama_chunk_result::ok(hash);
}

bool llama_kv_cache_manager::save_chunk(const std::string & hash) {
    // Stub - disk persistence not implemented in minimal version
    LLAMA_LOG_WARN("%s: disk save not implemented for chunk %s\n", __func__, hash.substr(0, 8).c_str());
    return false;
}

bool llama_kv_cache_manager::restore_chunk(const std::string & hash) {
    // Stub - disk persistence not implemented in minimal version
    LLAMA_LOG_WARN("%s: disk restore not implemented for chunk %s\n", __func__, hash.substr(0, 8).c_str());
    return false;
}

bool llama_kv_cache_manager::erase_chunk(const std::string & hash) {
    std::unique_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return false;
    }

    // Remove from KV cache by clearing the sequence
    if (it->second.seq_id >= 0) {
        unified_cache->seq_rm(it->second.seq_id, -1, -1);
    }

    chunks.erase(it);

    LLAMA_LOG_INFO("%s: erased chunk %s\n", __func__, hash.substr(0, 8).c_str());

    return true;
}

bool llama_kv_cache_manager::activate_chunk(const std::string & hash) {
    std::unique_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return false;
    }

    auto & chunk = it->second;

    if (chunk.status == llama_chunk_status::ACTIVE) {
        return true; // Already active
    }

    if (chunk.status == llama_chunk_status::INACTIVE) {
        // Move from inactive to active sequence
        unified_cache->seq_cp(INACTIVE_SEQ_ID, ACTIVE_SEQ_ID, -1, -1);
        unified_cache->seq_rm(INACTIVE_SEQ_ID, -1, -1);
        chunk.seq_id = ACTIVE_SEQ_ID;
        chunk.status = llama_chunk_status::ACTIVE;

        LLAMA_LOG_INFO("%s: activated chunk %s\n", __func__, hash.substr(0, 8).c_str());
        return true;
    }

    if (chunk.status == llama_chunk_status::SYSTEM) {
        // Need to restore from system cache
        // TODO: Implement system restore
        LLAMA_LOG_WARN("%s: system restore not yet implemented for chunk %s\n", __func__, hash.substr(0, 8).c_str());
        return false;
    }

    return false;
}

bool llama_kv_cache_manager::deactivate_chunk(const std::string & hash) {
    std::unique_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return false;
    }

    auto & chunk = it->second;

    if (chunk.status == llama_chunk_status::INACTIVE) {
        return true; // Already inactive
    }

    if (chunk.status == llama_chunk_status::ACTIVE) {
        // Move from active to inactive sequence
        unified_cache->seq_cp(ACTIVE_SEQ_ID, INACTIVE_SEQ_ID, -1, -1);
        unified_cache->seq_rm(ACTIVE_SEQ_ID, -1, -1);
        chunk.seq_id = INACTIVE_SEQ_ID;
        chunk.status = llama_chunk_status::INACTIVE;

        LLAMA_LOG_INFO("%s: deactivated chunk %s\n", __func__, hash.substr(0, 8).c_str());
        return true;
    }

    return false;
}

bool llama_kv_cache_manager::unload_chunk(const std::string & hash) {
    std::unique_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return false;
    }

    auto & chunk = it->second;

    if (chunk.status == llama_chunk_status::SYSTEM) {
        return true; // Already unloaded
    }

    // TODO: Implement actual KV cache serialization
    // For now, just mark as unloaded and clear from cache
    if (chunk.seq_id >= 0) {
        unified_cache->seq_rm(chunk.seq_id, -1, -1);
    }

    chunk.status = llama_chunk_status::SYSTEM;
    chunk.seq_id = -1;

    LLAMA_LOG_INFO("%s: unloaded chunk %s\n", __func__, hash.substr(0, 8).c_str());

    return true;
}

void llama_kv_cache_manager::compact() {
    // Delegate to unified cache - it handles defragmentation
    if (ctx) {
        unified_cache->init_update(ctx, true); // optimize=true triggers defrag
    }
}

void llama_kv_cache_manager::garbage_collect() {
    std::unique_lock lock(chunks_mutex);

    // Remove empty chunks
    for (auto it = chunks.begin(); it != chunks.end();) {
        if (it->second.status == llama_chunk_status::EMPTY) {
            it = chunks.erase(it);
        } else {
            ++it;
        }
    }
}

//
// Query methods
//

json llama_kv_cache_manager::get_context_info() const {
    std::shared_lock lock(chunks_mutex);

    json result = {
        {"total_chunks", chunks.size()},
        {"chunks", json::array()}
    };

    size_t active_count = 0;
    size_t inactive_count = 0;
    size_t system_count = 0;

    for (const auto & [hash, chunk] : chunks) {
        result["chunks"].push_back(chunk.to_json());

        switch (chunk.status) {
            case llama_chunk_status::ACTIVE:   active_count++; break;
            case llama_chunk_status::INACTIVE: inactive_count++; break;
            case llama_chunk_status::SYSTEM:   system_count++; break;
            default: break;
        }
    }

    result["active_chunks"] = active_count;
    result["inactive_chunks"] = inactive_count;
    result["system_chunks"] = system_count;
    result["kv_cache_size"] = unified_cache->get_size();
    result["kv_cache_used"] = unified_cache->get_cells().get_used();

    return result;
}

json llama_kv_cache_manager::get_chunk_info(const std::string & hash) const {
    std::shared_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return json::object();
    }

    return it->second.to_json();
}

std::string llama_kv_cache_manager::get_chunk_content(const std::string & hash) const {
    std::shared_lock lock(chunks_mutex);

    auto it = chunks.find(hash);
    if (it == chunks.end()) {
        return "";
    }

    return it->second.content;
}

//
// Slicing
//

llama_kv_cache_manager::SliceResult llama_kv_cache_manager::slice_chunk(const std::string & chunk_hash, const std::vector<size_t> & slice_positions) {
    std::shared_lock lock(chunks_mutex);

    auto it = chunks.find(chunk_hash);
    if (it == chunks.end()) {
        return SliceResult::error("Chunk not found");
    }

    const auto & chunk = it->second;

    // Validate slice positions
    for (size_t pos : slice_positions) {
        if (pos >= chunk.content.length()) {
            return SliceResult::error("Slice position out of bounds");
        }
    }

    // Create slices
    std::vector<std::string> slice_hashes;
    std::vector<size_t> positions = slice_positions;
    positions.push_back(chunk.content.length()); // Add end position
    std::sort(positions.begin(), positions.end());

    size_t start = 0;
    for (size_t end : positions) {
        if (end > start) {
            std::string slice_content = chunk.content.substr(start, end - start);

            // Add slice as new chunk
            llama_chunk_options opts;
            opts.metadata = chunk.metadata;
            opts.metadata["parent_chunk"] = chunk_hash;
            opts.metadata["slice_start"] = start;
            opts.metadata["slice_end"] = end;

            lock.unlock(); // Release read lock before write operation
            auto result = add_chunk_with_result(slice_content, opts);
            lock.lock();

            if (result.success) {
                slice_hashes.push_back(result.hash);
            }

            start = end;
        }
    }

    return SliceResult::ok(slice_hashes);
}

//
// Helper methods
//

bool llama_kv_cache_manager::tokenize_content(const std::string & content, std::vector<llama_token> & tokens) {
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: no context available\n", __func__);
        return false;
    }

    // Use llama.cpp tokenization API
    const int n_tokens_max = content.length() + 2; // rough estimate
    tokens.resize(n_tokens_max);

    const int n_tokens = llama_tokenize(
        llama_model_get_vocab(llama_get_model(ctx)),
        content.c_str(),
        content.length(),
        tokens.data(),
        n_tokens_max,
        true,  // add_special
        true   // parse_special
    );

    if (n_tokens < 0) {
        LLAMA_LOG_ERROR("%s: tokenization failed\n", __func__);
        return false;
    }

    tokens.resize(n_tokens);
    return !tokens.empty();
}

bool llama_kv_cache_manager::populate_kv_cache_with_tokens(const std::vector<llama_token> & tokens, llama_seq_id seq_id) {
    if (!ctx) {
        LLAMA_LOG_ERROR("%s: no context available\n", __func__);
        return false;
    }

    // Get the current position for this sequence
    llama_pos start_pos = seq_pos_max(seq_id) + 1;
    if (start_pos < 0) {
        start_pos = 0; // Empty sequence
    }

    LLAMA_LOG_INFO("%s: populating KV cache for seq_id %d starting at position %d with %zu tokens\n", 
                   __func__, seq_id, start_pos, tokens.size());

    // Create a batch with the tokens
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    // Manually populate batch fields
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = start_pos + i;  // Use correct starting position
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_id;
        batch.logits[i] = false;
    }
    batch.n_tokens = tokens.size();

    // Process the batch through the model to populate KV cache
    int ret = llama_decode(ctx, batch);

    llama_batch_free(batch);

    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: decode failed with code %d\n", __func__, ret);
        return false;
    }

    return true;
}

//
// Debug
//

void llama_kv_cache_manager::audit_kv_cache_state() const {
    LLAMA_LOG_INFO("=== KV Cache Audit ===\n");
    LLAMA_LOG_INFO("Cache size: %u, used: %u\n",
                   unified_cache->get_size(),
                   unified_cache->get_cells().get_used());

    std::shared_lock lock(chunks_mutex);
    LLAMA_LOG_INFO("Tracked chunks: %zu\n", chunks.size());

    for (const auto & [hash, chunk] : chunks) {
        LLAMA_LOG_INFO("  Chunk %s: status=%s, tokens=%zu, seq_id=%d\n",
                      hash.substr(0, 8).c_str(),
                      chunk.status == llama_chunk_status::ACTIVE ? "active" :
                      chunk.status == llama_chunk_status::INACTIVE ? "inactive" :
                      chunk.status == llama_chunk_status::SYSTEM ? "system" : "empty",
                      chunk.tokens.size(),
                      chunk.seq_id);
    }

    LLAMA_LOG_INFO("===================\n");
}
