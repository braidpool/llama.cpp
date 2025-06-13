#include "llama-kv-cache-manager.h"

#include <openssl/evp.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>

#include "llama-context.h"
#include "llama-vocab.h"

//
// Implementation classes
//

class llama_chunk_storage {
  private:
    std::string base_path;

  public:
    explicit llama_chunk_storage(const std::string & path) : base_path(path) {
        if (!path.empty()) {
            try {
                std::filesystem::create_directories(base_path + "/chunks");
            } catch (const std::exception &) {
                base_path.clear();
            }
        }
    }

    std::string get_chunk_path(const std::string & hash) const {
        if (base_path.empty()) {
            return "";
        }
        std::string prefix   = hash.substr(0, 2);
        std::string filename = "chunk_" + hash + ".kv";
        return base_path + "/chunks/" + prefix + "/" + filename;
    }

    bool save_chunk_state(const std::string & hash, const llama_chunk_info & chunk, llama_context * ctx) {
        if (base_path.empty()) {
            return false;
        }

        std::string path     = get_chunk_path(hash);
        std::string dir_path = std::filesystem::path(path).parent_path();

        try {
            std::filesystem::create_directories(dir_path);
        } catch (const std::exception &) {
            return false;
        }

        // Save metadata
        std::string   meta_path = path + ".meta";
        std::ofstream meta_file(meta_path);
        if (!meta_file) {
            return false;
        }

        json meta_data = {
            { "hash",        hash                },
            { "content",     chunk.content       },
            { "metadata",    chunk.metadata      },
            { "seq_id",      chunk.seq_id        },
            { "token_count", chunk.tokens.size() }
        };

        meta_file << meta_data.dump(4);
        meta_file.close();

        // Save KV cache state using llama.cpp APIs
        if (chunk.seq_id >= 0) {
            return ctx->state_seq_save_file(chunk.seq_id, path.c_str(), chunk.tokens.data(), chunk.tokens.size()) > 0;
        }

        return true;
    }

    bool restore_chunk_state(const std::string & hash, llama_chunk_info & chunk, llama_context * ctx) {
        if (base_path.empty()) {
            return false;
        }

        std::string path      = get_chunk_path(hash);
        std::string meta_path = path + ".meta";

        // Load metadata
        std::ifstream meta_file(meta_path);
        if (!meta_file) {
            return false;
        }

        json meta_data;
        meta_file >> meta_data;
        meta_file.close();

        chunk.hash     = meta_data["hash"];
        chunk.content  = meta_data["content"];
        chunk.metadata = meta_data["metadata"];
        chunk.seq_id   = meta_data["seq_id"];

        // Restore KV cache state using llama.cpp APIs
        if (std::filesystem::exists(path)) {
            llama_tokens tokens_out;
            size_t       n_token_capacity = meta_data["token_count"];
            tokens_out.resize(n_token_capacity);
            size_t n_token_count_out = 0;

            size_t result = ctx->state_seq_load_file(chunk.seq_id, path.c_str(), tokens_out.data(), n_token_capacity,
                                                     &n_token_count_out);

            if (result > 0) {
                tokens_out.resize(n_token_count_out);
                chunk.tokens = tokens_out;
                return true;
            }
        }

        return false;
    }

    bool erase_chunk(const std::string & hash) {
        if (base_path.empty()) {
            return true;
        }

        std::string path      = get_chunk_path(hash);
        std::string meta_path = path + ".meta";

        bool success = true;
        if (std::filesystem::exists(path)) {
            success &= std::filesystem::remove(path);
        }
        if (std::filesystem::exists(meta_path)) {
            success &= std::filesystem::remove(meta_path);
        }

        return success;
    }
};

class llama_kv_cache_manager_impl {
  private:
    llama_context *     ctx;
    const llama_model * model;
    const llama_vocab * vocab;
    llama_memory_t      memory;
    uint32_t            n_ctx;

    // Chunk management
    std::unordered_map<std::string, llama_chunk_info> chunks;

    // Storage
    llama_chunk_storage storage;

    // Configuration
    float  fragmentation_threshold = 0.20f;
    bool   auto_compact_enabled    = true;
    size_t max_memory_chunks       = 10;

    // Thread safety
    mutable std::shared_mutex manager_mutex;

    // Performance metrics
    std::atomic<size_t>                            total_chunks{ 0 };
    std::atomic<size_t>                            loaded_chunks{ 0 };
    std::atomic<size_t>                            saved_chunks{ 0 };
    std::chrono::high_resolution_clock::time_point start_time;

    llama_seq_id allocate_seq_id() {
        // All chunks share the same sequence so cross attention works
        return 0;
    }

    llama_tokens tokenize_content(const std::string & content, bool add_special) {
        // Use llama.cpp tokenization API directly
        std::vector<llama_token> tokens;
        int n_tokens = llama_tokenize(vocab, content.c_str(), content.length(), nullptr, 0, add_special, false);
        if (n_tokens < 0) {
            n_tokens = -n_tokens;
        }

        tokens.resize(n_tokens);
        int actual_tokens =
            llama_tokenize(vocab, content.c_str(), content.length(), tokens.data(), n_tokens, add_special, false);
        if (actual_tokens < 0) {
            actual_tokens = -actual_tokens;
        }

        tokens.resize(actual_tokens);
        return tokens;
    }

    bool store_tokens_in_sequence(const llama_tokens & tokens, llama_seq_id seq_id, llama_pos start_pos) {
        if (tokens.empty()) {
            return true;
        }

        // Create a batch for the tokens
        llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
        if (!batch.token) {
            return false;
        }

        // Add tokens to batch for this sequence - implement common_batch_add inline
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i]     = tokens[i];
            batch.pos[i]       = start_pos + i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i]    = false;
        }
        batch.n_tokens = tokens.size();

        // Decode the batch to populate KV cache
        int result = llama_decode(ctx, batch);
        llama_batch_free(batch);

        return result == 0;
    }

    json chunk_to_json(const llama_chunk_info & chunk) const {
        json result = {
            { "hash",      chunk.hash                                                               },
            { "seq_id",    chunk.seq_id                                                             },
            { "position", {
                { "start", chunk.token_start_pos },
                { "end",   chunk.token_end_pos }
            }},
            { "token_size",      chunk.tokens.size() },
            { "memory_size",     chunk.memory_size      },
            { "status",    chunk.status == llama_chunk_status::ACTIVE   ? "active" :
                        chunk.status == llama_chunk_status::INACTIVE ? "inactive" :
                        chunk.status == llama_chunk_status::SYSTEM   ? "system" :
                        chunk.status == llama_chunk_status::DISK     ? "disk" :
                                                                       "empty" },
            { "metadata",  chunk.metadata                                                           }
        };

        if (!chunk.save_file.empty()) {
            result["save_file"] = chunk.save_file;
        }

        if (!chunk.content.empty() && chunk.content.length() <= 100) {
            result["content_preview"] = chunk.content.substr(0, 100) + (chunk.content.length() > 100 ? "..." : "");
        }

        // Convert timestamps to ISO 8601 format
        auto time_to_string = [](const auto & tp) {
            auto              time_t = std::chrono::system_clock::to_time_t(tp);
            std::stringstream ss;
            ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
            return ss.str();
        };

        result["created_at"]    = time_to_string(chunk.created_at);
        result["last_accessed"] = time_to_string(chunk.last_accessed);

        return result;
    }

  public:
    explicit llama_kv_cache_manager_impl(llama_context * ctx, const std::string & storage_path) :
        ctx(ctx),
        storage(storage_path) {
        model      = &ctx->get_model();
        vocab      = llama_model_get_vocab(model);
        memory     = ctx->get_memory();
        n_ctx      = ctx->n_ctx();
        start_time = std::chrono::high_resolution_clock::now();
    }

    std::string add_chunk(const std::string & content, const llama_chunk_options & opts) {
        std::unique_lock lock(manager_mutex);

        // Compute content hash
        std::string hash = llama_content_addressing::compute_hash(content);

        // Check if content already exists (deduplication)
        auto existing = chunks.find(hash);
        if (existing != chunks.end()) {
            existing->second.last_accessed = std::chrono::system_clock::now();
            return hash;
        }

        // Tokenize content
        llama_tokens tokens = opts.tokenize ? tokenize_content(content, true) : opts.tokens;

        // Allocate sequence ID (shared sequence 0)
        llama_seq_id seq_id = allocate_seq_id();

        // Determine placement - append at the end of the shared sequence
        llama_pos start_pos = llama_memory_seq_pos_max(memory, seq_id);
        if (start_pos < 0) {
            start_pos = 0;
        } else {
            start_pos += 1;
        }

        // Create chunk info
        llama_chunk_info chunk;
        chunk.hash          = hash;
        chunk.content       = content;
        chunk.tokens        = tokens;
        chunk.seq_id        = seq_id;
        chunk.token_start_pos = start_pos;
        chunk.token_end_pos   = start_pos + tokens.size();
        chunk.memory_size     = tokens.size();
        chunk.status        = llama_chunk_status::ACTIVE;
        chunk.metadata      = opts.metadata;
        chunk.created_at    = std::chrono::system_clock::now();
        chunk.last_accessed = chunk.created_at;

        // Store in KV cache
        if (!store_tokens_in_sequence(tokens, seq_id, start_pos)) {
            return "";
        }

        // Update mappings
        chunks[hash] = chunk;

        loaded_chunks++;
        total_chunks++;

        return hash;
    }

    bool save_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);

        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }
        
        // Can only save chunks that are in VRAM (active or inactive)
        if (it->second.status != llama_chunk_status::ACTIVE && 
            it->second.status != llama_chunk_status::INACTIVE) {
            return false;
        }

        // Save using proper KV cache state APIs
        if (!storage.save_chunk_state(hash, it->second, ctx)) {
            return false;
        }

        // Update status and remove from active memory
        it->second.status    = llama_chunk_status::DISK;
        it->second.save_file = storage.get_chunk_path(hash);

        // Remove tokens from the shared sequence
        if (it->second.seq_id >= 0) {
            memory->seq_rm(it->second.seq_id, it->second.token_start_pos, it->second.token_end_pos);
            it->second.seq_id = -1;
        }

        loaded_chunks--;
        saved_chunks++;

        return true;
    }

    bool restore_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);

        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }
        
        // Handle different source states
        if (it->second.status == llama_chunk_status::DISK) {
            // Restore from disk
            if (!storage.restore_chunk_state(hash, it->second, ctx)) {
                return false;
            }
        } else if (it->second.status == llama_chunk_status::SYSTEM) {
            // Restore from system RAM - tokens and content are already in memory
            // Just need to put back in KV cache
        } else if (it->second.status == llama_chunk_status::ACTIVE || 
                   it->second.status == llama_chunk_status::INACTIVE) {
            // Already in VRAM, just activate it
            it->second.status = llama_chunk_status::ACTIVE;
            return true;
        } else {
            return false;
        }

        // Find position to place the chunk
        llama_pos start_pos = 0;
        if (it->second.status == llama_chunk_status::DISK || 
            it->second.status == llama_chunk_status::SYSTEM) {
            // For disk/system chunks, find the end of current context
            llama_pos max_pos = llama_memory_seq_pos_max(memory, 0);
            start_pos = (max_pos >= 0) ? max_pos + 1 : 0;
        }
        if (start_pos < 0) {
            return false;
        }

        // Allocate sequence ID (always use 0 for shared context)
        it->second.seq_id = 0;
        it->second.token_start_pos = start_pos;
        it->second.token_end_pos = start_pos + it->second.tokens.size();

        // Restore tokens to KV cache
        if (!store_tokens_in_sequence(it->second.tokens, it->second.seq_id, start_pos)) {
            it->second.seq_id = -1;
            return false;
        }
        // Update status and counters based on previous state
        auto prev_status = it->second.status;
        it->second.status = llama_chunk_status::ACTIVE;
        it->second.last_accessed = std::chrono::system_clock::now();
        
        if (prev_status == llama_chunk_status::DISK) {
            it->second.save_file.clear();
            saved_chunks--;
            loaded_chunks++;
        } else if (prev_status == llama_chunk_status::SYSTEM) {
            // Moving from system RAM back to VRAM
            it->second.system_cache.clear();
            loaded_chunks++;
        }

        return true;
    }

    bool erase_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);

        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }

        // Remove from KV cache if in VRAM
        if (it->second.status == llama_chunk_status::ACTIVE || 
            it->second.status == llama_chunk_status::INACTIVE) {
            memory->seq_rm(it->second.seq_id, it->second.token_start_pos, it->second.token_end_pos);
            loaded_chunks--;
        } else if (it->second.status == llama_chunk_status::DISK) {
            saved_chunks--;
        }

        // Erase from disk
        storage.erase_chunk(hash);

        // Remove from chunks map
        chunks.erase(it);
        total_chunks--;

        return true;
    }

    void compact() {
        std::unique_lock lock(manager_mutex);

        // Request KV cache optimization through memory interface
        auto update_state = memory->init_update(ctx, true);
        if (update_state && update_state->get_status() != LLAMA_MEMORY_STATUS_NO_UPDATE) {
            update_state->apply();
        }
    }

    void garbage_collect() {
        std::unique_lock lock(manager_mutex);

        // LRU eviction if over memory limit
        if (loaded_chunks > max_memory_chunks) {
            std::vector<std::pair<std::chrono::time_point<std::chrono::system_clock>, std::string>> access_times;

            for (const auto & [hash, chunk] : chunks) {
                if (chunk.status == llama_chunk_status::ACTIVE || 
                    chunk.status == llama_chunk_status::INACTIVE) {
                    access_times.push_back({ chunk.last_accessed, hash });
                }
            }

            std::sort(access_times.begin(), access_times.end());

            size_t to_evict = loaded_chunks - max_memory_chunks;
            for (size_t i = 0; i < to_evict && i < access_times.size(); ++i) {
                save_chunk(access_times[i].second);
            }
        }
    }

    json get_context_info() const {
        std::shared_lock lock(manager_mutex);

        llama_pos used_context = llama_memory_seq_pos_max(memory, 0);
        if (used_context >= 0) {
            used_context += 1;  // positions are 0-based
        } else {
            used_context = 0;
        }

        auto now    = std::chrono::high_resolution_clock::now();
        auto uptime = std::chrono::duration_cast<std::chrono::seconds>(now - start_time).count();

        json result = {
            { "total_context",  n_ctx                },
            { "used_context",   used_context         },
            { "free_context",   n_ctx - used_context },
            { "total_chunks",   total_chunks.load()  },
            { "loaded_chunks",  loaded_chunks.load() },
            { "saved_chunks",   saved_chunks.load()  },
            { "uptime_seconds", uptime               },
            { "fragmentation",  0.0f                 }
        };

        // Add chunk list
        json chunks_array = json::array();
        for (const auto & [hash, chunk] : chunks) {
            chunks_array.push_back(chunk_to_json(chunk));
        }
        result["chunks"] = chunks_array;

        // Placeholder gap information - not yet tracked
        result["gaps"] = json::array();

        // Build content index mapping hash -> metadata
        json index = json::object();
        for (const auto & [hash, chunk] : chunks) {
            index[hash] = {
                { "tokens",   chunk.tokens.size() },
                { "status",   chunk.status == llama_chunk_status::ACTIVE   ? "active" :
                            chunk.status == llama_chunk_status::INACTIVE ? "inactive" :
                            chunk.status == llama_chunk_status::SYSTEM   ? "system" :
                            chunk.status == llama_chunk_status::DISK     ? "disk" : "empty" },
                { "metadata", chunk.metadata }
            };
        }
        result["content_index"] = index;

        return result;
    }

    json get_chunk_info(const std::string & hash) const {
        std::shared_lock lock(manager_mutex);

        auto it = chunks.find(hash);
        if (it != chunks.end()) {
            return chunk_to_json(it->second);
        }

        return json::object();
    }

    std::string get_chunk_content(const std::string & hash) const {
        std::shared_lock lock(manager_mutex);

        auto it = chunks.find(hash);
        if (it != chunks.end()) {
            return it->second.content;
        }

        return "";
    }

    json search_chunks(const json & criteria) const {
        std::shared_lock lock(manager_mutex);

        json results = json::array();

        for (const auto & [hash, chunk] : chunks) {
            bool matches = true;

            // Check metadata criteria
            if (criteria.contains("metadata")) {
                for (const auto & [key, value] : criteria["metadata"].items()) {
                    if (!chunk.metadata.contains(key) || chunk.metadata[key] != value) {
                        matches = false;
                        break;
                    }
                }
            }

            // Check status criteria
            if (criteria.contains("status")) {
                std::string required_status = criteria["status"];
                std::string chunk_status = chunk.status == llama_chunk_status::ACTIVE   ? "active" :
                                          chunk.status == llama_chunk_status::INACTIVE ? "inactive" :
                                          chunk.status == llama_chunk_status::SYSTEM   ? "system" :
                                          chunk.status == llama_chunk_status::DISK     ? "disk" : "unknown";
                if (chunk_status != required_status) {
                    matches = false;
                }
            }

            if (matches) {
                results.push_back(chunk_to_json(chunk));
            }
        }

        return results;
    }

    bool activate_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);
        
        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }
        
        // Can only activate chunks that are inactive in VRAM
        if (it->second.status != llama_chunk_status::INACTIVE) {
            return false;
        }
        
        it->second.status = llama_chunk_status::ACTIVE;
        it->second.last_accessed = std::chrono::system_clock::now();
        return true;
    }
    
    bool deactivate_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);
        
        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }
        
        // Can only deactivate chunks that are active in VRAM
        if (it->second.status != llama_chunk_status::ACTIVE) {
            return false;
        }
        
        it->second.status = llama_chunk_status::INACTIVE;
        it->second.last_accessed = std::chrono::system_clock::now();
        return true;
    }
    
    bool unload_chunk(const std::string & hash) {
        std::unique_lock lock(manager_mutex);
        
        auto it = chunks.find(hash);
        if (it == chunks.end()) {
            return false;
        }
        
        // Can only unload chunks that are in VRAM (active or inactive)
        if (it->second.status != llama_chunk_status::ACTIVE && 
            it->second.status != llama_chunk_status::INACTIVE) {
            return false;
        }
        
        // Save KV cache state to system memory
        size_t state_size = ctx->state_seq_get_size(it->second.seq_id);
        if (state_size == 0) {
            return false;
        }
        
        it->second.system_cache.resize(state_size);
        size_t actual_size = ctx->state_seq_get_data(
            it->second.seq_id, 
            it->second.system_cache.data(), 
            state_size
        );
        
        if (actual_size != state_size) {
            it->second.system_cache.clear();
            return false;
        }
        
        // Remove from KV cache
        if (it->second.seq_id >= 0) {
            memory->seq_rm(it->second.seq_id, it->second.token_start_pos, it->second.token_end_pos);
            it->second.seq_id = -1;
        }
        
        it->second.status = llama_chunk_status::SYSTEM;
        it->second.last_accessed = std::chrono::system_clock::now();
        loaded_chunks--;
        
        return true;
    }

    json batch_operations(const json & request_body) {
        json results = json::array();

        const json & operations = request_body["operations"];
        for (const auto & op : operations) {
            std::string action = op["action"];
            std::string hash   = op.value("hash", "");

            json result = {
                { "action",  action },
                { "hash",    hash   },
                { "success", false  }
            };

            if (action == "save") {
                result["success"] = save_chunk(hash);
            } else if (action == "restore") {
                result["success"] = restore_chunk(hash);
            } else if (action == "erase") {
                result["success"] = erase_chunk(hash);
            } else if (action == "activate") {
                result["success"] = activate_chunk(hash);
            } else if (action == "deactivate") {
                result["success"] = deactivate_chunk(hash);
            } else if (action == "unload") {
                result["success"] = unload_chunk(hash);
            }

            results.push_back(result);
        }

        // Compact after batch operations if requested
        if (request_body.contains("compact_after") && request_body["compact_after"]) {
            compact();
        }

        return results;
    }

    // Configuration setters
    void set_fragmentation_threshold(float threshold) { fragmentation_threshold = threshold; }

    void set_auto_compact_enabled(bool enabled) { auto_compact_enabled = enabled; }

    void set_max_memory_chunks(size_t max_chunks) { max_memory_chunks = max_chunks; }
};

//
// Public API implementation
//

json llama_chunk_info::to_json() const {
    json result = {
        { "hash",      hash },
        { "seq_id",    seq_id },
        { "position", {
            { "start", token_start_pos },
            { "end",   token_end_pos }
        }},
        { "token_size",  tokens.size() },
        { "memory_size", memory_size },
        { "status",  status == llama_chunk_status::ACTIVE   ? "active" :
                    status == llama_chunk_status::INACTIVE ? "inactive" :
                    status == llama_chunk_status::SYSTEM   ? "system" :
                    status == llama_chunk_status::DISK     ? "disk" : "empty" },
        { "metadata", metadata }
    };

    if (!save_file.empty()) {
        result["save_file"] = save_file;
    }

    if (!content.empty() && content.length() <= 100) {
        result["content_preview"] = content.substr(0, 100) + (content.length() > 100 ? "..." : "");
    }

    auto time_to_string = [](const auto & tp) {
        auto              time_t = std::chrono::system_clock::to_time_t(tp);
        std::stringstream ss;
        ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%SZ");
        return ss.str();
    };

    result["created_at"]    = time_to_string(created_at);
    result["last_accessed"] = time_to_string(last_accessed);

    return result;
}

llama_kv_cache_manager::llama_kv_cache_manager(llama_context * ctx, const std::string & storage_path) :
    impl(std::make_unique<llama_kv_cache_manager_impl>(ctx, storage_path)) {}

llama_kv_cache_manager::~llama_kv_cache_manager() = default;

std::string llama_kv_cache_manager::add_chunk(const std::string & content, const llama_chunk_options & opts) {
    return impl->add_chunk(content, opts);
}

bool llama_kv_cache_manager::save_chunk(const std::string & hash) {
    return impl->save_chunk(hash);
}

bool llama_kv_cache_manager::restore_chunk(const std::string & hash) {
    return impl->restore_chunk(hash);
}

bool llama_kv_cache_manager::erase_chunk(const std::string & hash) {
    return impl->erase_chunk(hash);
}

bool llama_kv_cache_manager::activate_chunk(const std::string & hash) {
    return impl->activate_chunk(hash);
}

bool llama_kv_cache_manager::deactivate_chunk(const std::string & hash) {
    return impl->deactivate_chunk(hash);
}

bool llama_kv_cache_manager::unload_chunk(const std::string & hash) {
    return impl->unload_chunk(hash);
}

json llama_kv_cache_manager::batch_operations(const json & request_body) {
    return impl->batch_operations(request_body);
}

void llama_kv_cache_manager::compact() {
    impl->compact();
}

void llama_kv_cache_manager::garbage_collect() {
    impl->garbage_collect();
}

json llama_kv_cache_manager::get_context_info() const {
    return impl->get_context_info();
}

json llama_kv_cache_manager::get_chunk_info(const std::string & hash) const {
    return impl->get_chunk_info(hash);
}

std::string llama_kv_cache_manager::get_chunk_content(const std::string & hash) const {
    return impl->get_chunk_content(hash);
}

json llama_kv_cache_manager::search_chunks(const json & criteria) const {
    return impl->search_chunks(criteria);
}

void llama_kv_cache_manager::set_fragmentation_threshold(float threshold) {
    impl->set_fragmentation_threshold(threshold);
}

void llama_kv_cache_manager::set_auto_compact_enabled(bool enabled) {
    impl->set_auto_compact_enabled(enabled);
}

void llama_kv_cache_manager::set_max_memory_chunks(size_t max_chunks) {
    impl->set_max_memory_chunks(max_chunks);
}

//
// Utility functions
//

std::string llama_content_addressing::compute_hash(const std::string & content) {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int  hash_len;

    EVP_MD_CTX * mdctx = EVP_MD_CTX_new();
    if (!mdctx) {
        return "";
    }

    if (EVP_DigestInit_ex(mdctx, EVP_sha256(), nullptr) != 1) {
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    if (EVP_DigestUpdate(mdctx, content.c_str(), content.length()) != 1) {
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    if (EVP_DigestFinal_ex(mdctx, hash, &hash_len) != 1) {
        EVP_MD_CTX_free(mdctx);
        return "";
    }

    EVP_MD_CTX_free(mdctx);

    // Convert to hex string
    std::stringstream ss;
    for (unsigned int i = 0; i < hash_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int) hash[i];
    }
    return ss.str();
}

std::string llama_content_addressing::generate_filename(const std::string & hash) {
    return "chunk_" + hash.substr(0, 16) + ".kv";
}

bool llama_content_addressing::validate_hash(const std::string & hash) {
    return hash.length() == 64 && std::all_of(hash.begin(), hash.end(), [](char c) { return std::isxdigit(c); });
}
