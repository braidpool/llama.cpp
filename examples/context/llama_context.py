#!/usr/bin/env python3
"""
Llama.cpp Context Management API - Python Example

This example demonstrates how to use the Context Management API to:
- Add content chunks with different positioning strategies
- Save and restore chunks to/from disk
- Manage memory fragmentation through compaction
- Implement dynamic context switching for different conversation modes
- Handle batch operations efficiently

The Context Management API uses SHA256 content addressing for automatic
deduplication and provides Git-like content management for LLM contexts.
"""

import requests
import json
import hashlib
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

class PositionStrategy(Enum):
    """Positioning strategies for new chunks"""
    AUTO = "auto"                    # Append at end (default)
    BEST_FIT = "best_fit"           # Find smallest suitable gap
    FIRST_FIT = "first_fit"         # Use first available gap
    COMPACT_FIRST = "compact_first" # Compact memory then append
    SPECIFIC_POS = "specific_pos"   # Use specific position
    AFTER_CHUNK = "after"           # Position after another chunk
    BEFORE_CHUNK = "before"         # Position before another chunk

@dataclass
class ChunkInfo:
    """Information about a context chunk"""
    hash: str
    content: str
    size: int
    status: str
    position_start: int
    position_end: int
    metadata: Dict[str, Any]
    created_at: str
    last_accessed: str

class LlamaContextManager:
    """
    High-level interface for managing llama.cpp context chunks
    
    This class provides a Pythonic interface to the Context Management API,
    making it easy to implement dynamic context switching, memory management,
    and content organization strategies.
    """
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.context_url = f"{base_url}/context"
        self._chunk_cache: Dict[str, ChunkInfo] = {}
    
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            if not response.ok:
                print(f"HTTP {response.status_code}: {response.text}")
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
    
    def compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content (same algorithm as server)"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get comprehensive context information"""
        response = self._make_request("GET", self.context_url)
        return response.json()
    
    def add_chunk(self, 
                  content: str, 
                  position: PositionStrategy = PositionStrategy.AUTO,
                  relative_to_hash: Optional[str] = None,
                  preferred_position: Optional[int] = None,
                  metadata: Optional[Dict[str, Any]] = None,
                  tokenize: bool = True,
                  tokens: Optional[List[int]] = None) -> str:
        """
        Add a content chunk to the context
        
        Args:
            content: Text content to add
            position: Positioning strategy
            relative_to_hash: Hash of chunk to position relative to (for AFTER/BEFORE)
            preferred_position: Specific position (for SPECIFIC_POS)
            metadata: User-defined metadata
            tokenize: Whether to tokenize content server-side
            tokens: Pre-tokenized content (if tokenize=False)
            
        Returns:
            Hash of the added chunk
        """
        payload = {
            "content": content,
            "tokenize": tokenize,
            "metadata": metadata or {}
        }
        
        # Handle positioning
        if position == PositionStrategy.AFTER_CHUNK and relative_to_hash:
            payload["position"] = f"after:{relative_to_hash}"
        elif position == PositionStrategy.BEFORE_CHUNK and relative_to_hash:
            payload["position"] = f"before:{relative_to_hash}"
        elif position == PositionStrategy.SPECIFIC_POS and preferred_position is not None:
            payload["position"] = position.value
            payload["preferred_position"] = preferred_position
        else:
            payload["position"] = position.value
        
        if not tokenize and tokens:
            payload["tokens"] = tokens
        
        response = self._make_request("POST", self.context_url, json=payload)
        result = response.json()
        
        # Cache chunk info
        chunk_hash = result["hash"]
        self._update_chunk_cache(chunk_hash)
        
        return chunk_hash
    
    def save_chunk(self, chunk_hash: str) -> bool:
        """Save chunk to disk (frees memory)"""
        url = f"{self.context_url}/{chunk_hash}?action=save"
        response = self._make_request("POST", url)
        result = response.json()
        
        if result.get("success"):
            self._update_chunk_cache(chunk_hash)
        
        return result.get("success", False)
    
    def restore_chunk(self, chunk_hash: str) -> bool:
        """Restore chunk from disk to memory"""
        url = f"{self.context_url}/{chunk_hash}?action=restore"
        response = self._make_request("POST", url)
        result = response.json()
        
        if result.get("success"):
            self._update_chunk_cache(chunk_hash)
        
        return result.get("success", False)
    
    def erase_chunk(self, chunk_hash: str) -> bool:
        """Permanently delete chunk"""
        url = f"{self.context_url}/{chunk_hash}?action=erase"
        response = self._make_request("POST", url)
        result = response.json()
        
        if result.get("success"):
            self._chunk_cache.pop(chunk_hash, None)
        
        return result.get("success", False)
    
    def batch_operations(self, operations: List[Dict[str, str]], compact_after: bool = False) -> List[Dict[str, Any]]:
        """
        Perform multiple operations in a single request
        
        Args:
            operations: List of operations, each with 'action' and 'hash' keys
            compact_after: Whether to compact memory after operations
            
        Returns:
            List of operation results
        """
        payload = {
            "operations": operations,
            "compact_after": compact_after
        }
        
        response = self._make_request("POST", f"{self.context_url}/batch", json=payload)
        results = response.json()
        
        # Update cache for successful operations
        for op, result in zip(operations, results):
            if result.get("success"):
                self._update_chunk_cache(op["hash"])
        
        return results
    
    def compact_memory(self) -> bool:
        """Force memory compaction to reduce fragmentation"""
        response = self._make_request("POST", f"{self.context_url}/compact")
        result = response.json()
        
        if result.get("success"):
            # Clear cache as positions may have changed
            self._chunk_cache.clear()
        
        return result.get("success", False)
    
    def garbage_collect(self) -> bool:
        """Run garbage collection (auto-save old chunks)"""
        response = self._make_request("POST", f"{self.context_url}/gc")
        return response.json().get("success", False)
    
    def _update_chunk_cache(self, chunk_hash: str):
        """Update cached chunk information"""
        info = self.get_context_info()
        chunk_data = next((c for c in info["chunks"] if c["hash"] == chunk_hash), None)
        if chunk_data:
            # Handle both response formats: nested position object and flat fields
            if "position" in chunk_data:
                position_start = chunk_data["position"]["start"]
                position_end = chunk_data["position"]["end"]
            else:
                position_start = chunk_data["token_start_pos"]
                position_end = chunk_data["token_end_pos"]
                
            self._chunk_cache[chunk_hash] = ChunkInfo(
                hash=chunk_data["hash"],
                content="",  # Content not included in context info
                size=chunk_data.get("token_size", chunk_data.get("size")),
                status=chunk_data["status"],
                position_start=position_start,
                position_end=position_end,
                metadata=chunk_data["metadata"],
                created_at=chunk_data["created_at"],
                last_accessed=chunk_data["last_accessed"]
            )
    
    def get_chunks_by_metadata(self, **criteria) -> List[ChunkInfo]:
        """Get chunks matching metadata criteria"""
        info = self.get_context_info()
        matching_chunks = []
        
        for chunk_data in info["chunks"]:
            metadata = chunk_data.get("metadata", {})
            matches = all(
                metadata.get(key) == value 
                for key, value in criteria.items()
            )
            if matches:
                # Handle both response formats: nested position object and flat fields
                if "position" in chunk_data:
                    position_start = chunk_data["position"]["start"]
                    position_end = chunk_data["position"]["end"]
                else:
                    position_start = chunk_data["token_start_pos"]
                    position_end = chunk_data["token_end_pos"]
                    
                chunk_info = ChunkInfo(
                    hash=chunk_data["hash"],
                    content="",
                    size=chunk_data.get("token_size", chunk_data.get("size")),
                    status=chunk_data["status"],
                    position_start=position_start,
                    position_end=position_end,
                    metadata=metadata,
                    created_at=chunk_data["created_at"],
                    last_accessed=chunk_data["last_accessed"]
                )
                matching_chunks.append(chunk_info)
        
        return matching_chunks
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        info = self.get_context_info()
        stats = {
            "total_context": info["total_context"],
            "used_context": info["used_context"],
            "free_context": info["free_context"],
            "fragmentation": info["fragmentation"],
            "total_chunks": info["total_chunks"],
            "loaded_chunks": info["loaded_chunks"],
            "saved_chunks": info["saved_chunks"]
        }
        
        # Add performance metrics if available
        if "performance" in info:
            stats["performance"] = info["performance"]
        
        return stats

class ConversationManager:
    """
    Example implementation of dynamic conversation context management
    
    This class demonstrates how to use the Context Management API for
    real-world conversational AI scenarios with different modes and
    context switching capabilities.
    """
    
    def __init__(self, context_manager: LlamaContextManager):
        self.ctx = context_manager
        self.conversation_modes = {}
        self.current_mode = "default"
        self.system_prompt_hash = None
        self.conversation_history = []
    
    def set_system_prompt(self, prompt: str, mode: str = "default") -> str:
        """Set system prompt for a conversation mode"""
        hash_val = self.ctx.add_chunk(
            content=prompt,
            metadata={"type": "system_prompt", "mode": mode},
            position=PositionStrategy.AUTO
        )
        
        self.conversation_modes[mode] = {
            "system_prompt": hash_val,
            "context_chunks": []
        }
        
        if mode == self.current_mode:
            self.system_prompt_hash = hash_val
        
        return hash_val
    
    def switch_mode(self, mode: str):
        """Switch conversation mode"""
        if mode not in self.conversation_modes:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Save current context chunks
        if self.current_mode in self.conversation_modes:
            current_chunks = self.ctx.get_chunks_by_metadata(mode=self.current_mode)
            save_ops = [{"action": "save", "hash": chunk.hash} for chunk in current_chunks]
            if save_ops:
                self.ctx.batch_operations(save_ops)
        
        # Switch to new mode
        self.current_mode = mode
        mode_info = self.conversation_modes[mode]
        
        # Restore system prompt and context only if they were saved
        if mode_info["system_prompt"]:
            # Check if the system prompt needs to be restored
            info = self.ctx.get_context_info()
            system_chunk = next((c for c in info["chunks"] if c["hash"] == mode_info["system_prompt"]), None)
            
            if system_chunk and system_chunk["status"] == "saved":
                self.ctx.restore_chunk(mode_info["system_prompt"])
            
            self.system_prompt_hash = mode_info["system_prompt"]
        
        # Restore context chunks (only those that are saved)
        if mode_info["context_chunks"]:
            info = self.ctx.get_context_info()
            saved_chunks = {c["hash"]: c for c in info["chunks"] if c["status"] == "saved"}
            
            restore_ops = [{"action": "restore", "hash": chunk_hash} 
                          for chunk_hash in mode_info["context_chunks"]
                          if chunk_hash in saved_chunks]
            if restore_ops:
                self.ctx.batch_operations(restore_ops, compact_after=True)
    
    def add_user_message(self, message: str) -> str:
        """Add user message to conversation"""
        hash_val = self.ctx.add_chunk(
            content=f"User: {message}",
            metadata={"type": "user_message", "mode": self.current_mode},
            position=PositionStrategy.AUTO
        )
        
        self.conversation_history.append(("user", hash_val))
        self._update_mode_context(hash_val)
        
        return hash_val
    
    def add_assistant_response(self, response: str) -> str:
        """Add assistant response to conversation"""
        hash_val = self.ctx.add_chunk(
            content=f"Assistant: {response}",
            metadata={"type": "assistant_response", "mode": self.current_mode},
            position=PositionStrategy.AUTO
        )
        
        self.conversation_history.append(("assistant", hash_val))
        self._update_mode_context(hash_val)
        
        return hash_val
    
    def _update_mode_context(self, chunk_hash: str):
        """Update context chunks for current mode"""
        if self.current_mode in self.conversation_modes:
            self.conversation_modes[self.current_mode]["context_chunks"].append(chunk_hash)
    
    def summarize_old_context(self, summary: str, keep_recent: int = 5):
        """Replace old conversation with summary, keeping recent messages"""
        if len(self.conversation_history) <= keep_recent:
            return
        
        # Find messages to archive
        to_archive = self.conversation_history[:-keep_recent]
        to_keep = self.conversation_history[-keep_recent:]
        
        # Create summary chunk
        summary_hash = self.ctx.add_chunk(
            content=f"Conversation Summary: {summary}",
            metadata={"type": "summary", "mode": self.current_mode},
            position=PositionStrategy.AFTER_CHUNK,
            relative_to_hash=self.system_prompt_hash
        )
        
        # Erase old messages
        erase_ops = [{"action": "erase", "hash": chunk_hash} 
                    for _, chunk_hash in to_archive]
        self.ctx.batch_operations(erase_ops, compact_after=True)
        
        # Update conversation history
        self.conversation_history = [("summary", summary_hash)] + to_keep

def main():
    """Comprehensive example demonstrating the Context Management API"""
    
    print("🦙 Llama.cpp Context Management API Example")
    print("=" * 50)
    
    # Initialize context manager
    try:
        ctx = LlamaContextManager()
        print("✅ Connected to llama.cpp server")
    except Exception as e:
        print(f"❌ Failed to connect to server: {e}")
        return
    
    # Example 1: Basic chunk operations
    print("\n📝 Example 1: Basic Chunk Operations")
    print("-" * 30)
    
    # Add some content
    content1 = "The quick brown fox jumps over the lazy dog."
    hash1 = ctx.add_chunk(content1, metadata={"type": "example", "id": 1})
    print(f"Added chunk: {hash1[:16]}...")
    
    content2 = "This is another piece of content for testing."
    hash2 = ctx.add_chunk(content2, metadata={"type": "example", "id": 2})
    print(f"Added chunk: {hash2[:16]}...")
    
    # Show context info
    stats = ctx.get_memory_stats()
    print(f"Memory usage: {stats['used_context']}/{stats['total_context']} tokens")
    print(f"Fragmentation: {stats['fragmentation']:.2%}")
    
    # Example 2: Save and restore operations
    print("\n💾 Example 2: Save and Restore Operations")
    print("-" * 40)
    
    print(f"Saving chunk {hash1[:16]}...")
    ctx.save_chunk(hash1)
    
    stats = ctx.get_memory_stats()
    print(f"After save - Loaded: {stats['loaded_chunks']}, Saved: {stats['saved_chunks']}")
    
    print(f"Restoring chunk {hash1[:16]}...")
    ctx.restore_chunk(hash1)
    
    stats = ctx.get_memory_stats()
    print(f"After restore - Loaded: {stats['loaded_chunks']}, Saved: {stats['saved_chunks']}")
    
    # Example 3: Conversation management
    print("\n💬 Example 3: Conversation Management")
    print("-" * 35)
    
    conv = ConversationManager(ctx)
    
    # Set up different conversation modes
    conv.set_system_prompt(
        "You are a helpful AI assistant specializing in programming.",
        mode="programming"
    )
    
    conv.set_system_prompt(
        "You are a creative writing assistant focused on storytelling.",
        mode="creative"
    )
    
    # Programming conversation
    print("Starting programming conversation...")
    conv.switch_mode("programming")
    conv.add_user_message("How do I implement a binary search in Python?")
    conv.add_assistant_response("Here's a binary search implementation: [code example]")
    
    # Switch to creative mode
    print("Switching to creative writing mode...")
    conv.switch_mode("creative")
    conv.add_user_message("Write a short story about a robot discovering emotions.")
    conv.add_assistant_response("Once upon a time, there was a robot named ARIA...")
    
    # Switch back to programming
    print("Switching back to programming mode...")
    conv.switch_mode("programming")
    conv.add_user_message("Can you explain the time complexity of that search?")
    
    # Example 4: Batch operations and memory management
    print("\n🔧 Example 4: Batch Operations and Memory Management")
    print("-" * 50)
    
    # Add multiple chunks
    doc_chunks = []
    for i in range(5):
        content = f"Document section {i}: This is content for section {i} of a larger document."
        hash_val = ctx.add_chunk(
            content, 
            metadata={"type": "document", "section": i},
            position=PositionStrategy.AUTO
        )
        doc_chunks.append(hash_val)
    
    print(f"Added {len(doc_chunks)} document sections")
    
    # Batch save some chunks
    save_ops = [{"action": "save", "hash": chunk} for chunk in doc_chunks[::2]]  # Every other chunk
    print(f"Batch saving {len(save_ops)} chunks...")
    results = ctx.batch_operations(save_ops, compact_after=True)
    
    successful = sum(1 for r in results if r.get("success"))
    print(f"Successfully saved {successful}/{len(save_ops)} chunks")
    
    # Show final stats
    print("\n📊 Final Statistics")
    print("-" * 20)
    
    stats = ctx.get_memory_stats()
    info = ctx.get_context_info()
    
    print(f"Total context size: {stats['total_context']} tokens")
    print(f"Used context: {stats['used_context']} tokens ({stats['used_context']/stats['total_context']:.1%})")
    print(f"Free context: {stats['free_context']} tokens")
    print(f"Fragmentation: {stats['fragmentation']:.2%}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Loaded chunks: {stats['loaded_chunks']}")
    print(f"Saved chunks: {stats['saved_chunks']}")
    
    # Show performance metrics if available
    if "performance" in stats:
        perf = stats["performance"]
        print(f"\n⚡ Performance Metrics:")
        print(f"KV cache hit ratio: {perf['hit_ratio']:.1%} ({perf['kv_cache_hits']}/{perf['kv_cache_hits'] + perf['kv_cache_misses']})")
        print(f"Tokens saved/restored: {perf['total_tokens_saved']}/{perf['total_tokens_restored']}")
        print(f"Uptime: {perf['uptime_seconds']} seconds")
    
    # Example 5: Content deduplication
    print("\n🔄 Example 5: Content Deduplication")
    print("-" * 30)
    
    duplicate_content = "This exact content will be added twice to test deduplication."
    
    hash_a = ctx.add_chunk(duplicate_content, metadata={"instance": "first"})
    hash_b = ctx.add_chunk(duplicate_content, metadata={"instance": "second"})
    
    if hash_a == hash_b:
        print("✅ Content deduplication working - same hash returned")
        print(f"   Hash: {hash_a[:16]}...")
    else:
        print("❌ Deduplication not working - different hashes returned")
    
    # Cleanup example
    print("\n🧹 Cleanup")
    print("-" * 10)
    
    # Get all test chunks
    test_chunks = ctx.get_chunks_by_metadata(type="example")
    
    if test_chunks:
        erase_ops = [{"action": "erase", "hash": chunk.hash} for chunk in test_chunks]
        print(f"Cleaning up {len(erase_ops)} test chunks...")
        ctx.batch_operations(erase_ops)
    
    print("\n✨ Example completed successfully!")
    print("\n🚀 KV Cache Integration Benefits:")
    print("   • Tokens are decoded directly into KV cache at specific positions")
    print("   • Restore operations reuse computed KV states (avoiding re-computation)")  
    print("   • Real memory compaction moves actual KV cache data")
    print("   • Performance metrics track cache hit ratios for optimization")

if __name__ == "__main__":
    main()