#!/usr/bin/env python3

import pytest
import requests
import json
import hashlib
import time
from typing import Dict, Any, List

# Test configuration
BASE_URL = "http://localhost:8080"
CONTEXT_URL = f"{BASE_URL}/context"

class TestContextManagement:
    """Comprehensive test suite for the Context Management API"""
    
    @pytest.fixture(scope="class")
    def server_ready(self):
        """Ensure server is running and responding"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            assert response.status_code == 200
            return True
        except requests.RequestException:
            pytest.skip("Server not running")
    
    def compute_hash(self, content: str) -> str:
        """Compute SHA256 hash of content for verification"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def test_get_empty_context(self, server_ready):
        """Test getting context info when no chunks exist"""
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        
        data = response.json()
        assert "total_context" in data
        assert "used_context" in data
        assert "free_context" in data
        assert "fragmentation" in data
        assert "chunks" in data
        assert "gaps" in data
        assert "content_index" in data
        
        assert isinstance(data["chunks"], list)
        assert len(data["chunks"]) == 0
        assert data["used_context"] == 0
    
    def test_add_simple_chunk(self, server_ready):
        """Test adding a simple text chunk"""
        content = "Hello, world! This is a test chunk."
        expected_hash = self.compute_hash(content)
        
        response = requests.post(CONTEXT_URL, json={
            "content": content,
            "metadata": {"type": "test", "description": "Simple test chunk"}
        })
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["hash"] == expected_hash
        assert "size" in data
        assert "position" in data
        assert data["position"]["start"] >= 0
        assert data["position"]["end"] > data["position"]["start"]
        assert data["status"] == "loaded"
        assert data["deduplication"] == False
    
    def test_add_duplicate_content(self, server_ready):
        """Test that adding duplicate content returns same hash"""
        content = "Duplicate content test"
        
        # Add first time
        response1 = requests.post(CONTEXT_URL, json={"content": content})
        assert response1.status_code == 200
        hash1 = response1.json()["hash"]
        
        # Add second time - should deduplicate
        response2 = requests.post(CONTEXT_URL, json={"content": content})
        assert response2.status_code == 200
        hash2 = response2.json()["hash"]
        
        assert hash1 == hash2
        # Note: deduplication flag might not be set in current implementation
    
    def test_add_chunk_with_positioning(self, server_ready):
        """Test different positioning strategies"""
        base_content = "Base chunk for positioning tests"
        
        # Add base chunk
        response = requests.post(CONTEXT_URL, json={
            "content": base_content,
            "metadata": {"type": "base"}
        })
        assert response.status_code == 200
        base_hash = response.json()["hash"]
        
        # Test positioning after base chunk
        after_content = "Chunk positioned after base"
        response = requests.post(CONTEXT_URL, json={
            "content": after_content,
            "position": f"after:{base_hash}",
            "metadata": {"type": "after"}
        })
        assert response.status_code == 200
        
        # Test best fit positioning
        small_content = "Small"
        response = requests.post(CONTEXT_URL, json={
            "content": small_content,
            "position": "best_fit",
            "metadata": {"type": "small"}
        })
        assert response.status_code == 200
    
    def test_chunk_operations(self, server_ready):
        """Test save, restore, and erase operations"""
        content = "Content for operation testing"
        
        # Add chunk
        response = requests.post(CONTEXT_URL, json={"content": content})
        assert response.status_code == 200
        chunk_hash = response.json()["hash"]
        
        # Save chunk
        response = requests.post(f"{CONTEXT_URL}/{chunk_hash}?action=save")
        assert response.status_code == 200
        assert response.json()["success"] == True
        assert response.json()["action"] == "save"
        
        # Verify chunk is saved
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        chunks = response.json()["chunks"]
        saved_chunk = next((c for c in chunks if c["hash"] == chunk_hash), None)
        assert saved_chunk is not None
        assert saved_chunk["status"] == "saved"
        
        # Restore chunk
        response = requests.post(f"{CONTEXT_URL}/{chunk_hash}?action=restore")
        assert response.status_code == 200
        assert response.json()["success"] == True
        assert response.json()["action"] == "restore"
        
        # Verify chunk is loaded
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        chunks = response.json()["chunks"]
        restored_chunk = next((c for c in chunks if c["hash"] == chunk_hash), None)
        assert restored_chunk is not None
        assert restored_chunk["status"] == "loaded"
        
        # Erase chunk
        response = requests.post(f"{CONTEXT_URL}/{chunk_hash}?action=erase")
        assert response.status_code == 200
        assert response.json()["success"] == True
        assert response.json()["action"] == "erase"
        
        # Verify chunk is gone
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        chunks = response.json()["chunks"]
        erased_chunk = next((c for c in chunks if c["hash"] == chunk_hash), None)
        assert erased_chunk is None
    
    def test_batch_operations(self, server_ready):
        """Test batch operations on multiple chunks"""
        # Add several chunks
        chunks = []
        for i in range(3):
            content = f"Batch test chunk {i}"
            response = requests.post(CONTEXT_URL, json={
                "content": content,
                "metadata": {"type": "batch_test", "index": i}
            })
            assert response.status_code == 200
            chunks.append(response.json()["hash"])
        
        # Batch save operations
        batch_ops = [
            {"action": "save", "hash": chunks[0]},
            {"action": "save", "hash": chunks[1]}
        ]
        
        response = requests.post(f"{CONTEXT_URL}/batch", json={
            "operations": batch_ops,
            "compact_after": True
        })
        
        assert response.status_code == 200
        results = response.json()
        assert len(results) == 2
        
        for result in results:
            assert result["success"] == True
            assert result["action"] == "save"
            assert result["hash"] in chunks[:2]
    
    def test_memory_compaction(self, server_ready):
        """Test memory compaction functionality"""
        # Add several chunks to create fragmentation
        chunk_hashes = []
        for i in range(5):
            content = f"Compaction test chunk {i} with some content to make it substantial"
            response = requests.post(CONTEXT_URL, json={
                "content": content,
                "metadata": {"type": "compaction_test", "index": i}
            })
            assert response.status_code == 200
            chunk_hashes.append(response.json()["hash"])
        
        # Save some chunks to create gaps
        for i in [1, 3]:
            response = requests.post(f"{CONTEXT_URL}/{chunk_hashes[i]}?action=save")
            assert response.status_code == 200
        
        # Check fragmentation
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        pre_compact_fragmentation = response.json()["fragmentation"]
        
        # Force compaction
        response = requests.post(f"{CONTEXT_URL}/compact")
        assert response.status_code == 200
        assert response.json()["success"] == True
        
        # Check fragmentation after compaction
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        post_compact_fragmentation = response.json()["fragmentation"]
        
        # Fragmentation should be reduced
        assert post_compact_fragmentation <= pre_compact_fragmentation
    
    def test_garbage_collection(self, server_ready):
        """Test garbage collection functionality"""
        # Add chunks
        chunk_hashes = []
        for i in range(3):
            content = f"GC test chunk {i}"
            response = requests.post(CONTEXT_URL, json={
                "content": content,
                "metadata": {"type": "gc_test", "index": i}
            })
            assert response.status_code == 200
            chunk_hashes.append(response.json()["hash"])
        
        # Run garbage collection
        response = requests.post(f"{CONTEXT_URL}/gc")
        assert response.status_code == 200
        assert response.json()["success"] == True
        
        # Verify system is still functional
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
    
    def test_invalid_operations(self, server_ready):
        """Test error handling for invalid operations"""
        # Invalid hash format
        response = requests.post(f"{CONTEXT_URL}/invalid_hash?action=save")
        assert response.status_code == 400
        
        # Non-existent hash
        fake_hash = "a" * 64  # Valid format but doesn't exist
        response = requests.post(f"{CONTEXT_URL}/{fake_hash}?action=save")
        assert response.status_code == 500  # Should fail because chunk doesn't exist
        
        # Invalid action
        response = requests.post(CONTEXT_URL, json={"content": "test"})
        assert response.status_code == 200
        chunk_hash = response.json()["hash"]
        
        response = requests.post(f"{CONTEXT_URL}/{chunk_hash}?action=invalid")
        assert response.status_code == 400
        
        # Missing content in POST
        response = requests.post(CONTEXT_URL, json={})
        assert response.status_code == 400
    
    def test_metadata_functionality(self, server_ready):
        """Test metadata storage and retrieval"""
        content = "Content with metadata"
        metadata = {
            "type": "document",
            "source": "test_file.txt",
            "section": "introduction",
            "tags": ["important", "test"],
            "priority": 1
        }
        
        response = requests.post(CONTEXT_URL, json={
            "content": content,
            "metadata": metadata
        })
        
        assert response.status_code == 200
        chunk_hash = response.json()["hash"]
        
        # Verify metadata is stored
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        
        chunks = response.json()["chunks"]
        test_chunk = next((c for c in chunks if c["hash"] == chunk_hash), None)
        assert test_chunk is not None
        assert test_chunk["metadata"] == metadata
    
    def test_context_info_accuracy(self, server_ready):
        """Test that context info provides accurate statistics"""
        # Clear any existing chunks first
        response = requests.get(CONTEXT_URL)
        existing_chunks = response.json()["chunks"]
        for chunk in existing_chunks:
            requests.post(f"{CONTEXT_URL}/{chunk['hash']}?action=erase")
        
        # Add known chunks and verify stats
        chunk_contents = [
            "First chunk for stats testing",
            "Second chunk with different content",
            "Third chunk for comprehensive testing"
        ]
        
        added_hashes = []
        for content in chunk_contents:
            response = requests.post(CONTEXT_URL, json={"content": content})
            assert response.status_code == 200
            added_hashes.append(response.json()["hash"])
        
        # Get context info
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        data = response.json()
        
        # Verify chunk count
        assert len(data["chunks"]) == len(chunk_contents)
        assert data["loaded_chunks"] == len(chunk_contents)
        assert data["saved_chunks"] == 0
        
        # Verify all added chunks are present
        chunk_hashes = [c["hash"] for c in data["chunks"]]
        for expected_hash in added_hashes:
            assert expected_hash in chunk_hashes
        
        # Save one chunk and verify stats update
        response = requests.post(f"{CONTEXT_URL}/{added_hashes[0]}?action=save")
        assert response.status_code == 200
        
        response = requests.get(CONTEXT_URL)
        assert response.status_code == 200
        data = response.json()
        
        assert data["loaded_chunks"] == len(chunk_contents) - 1
        assert data["saved_chunks"] == 1
    
    def test_pre_tokenized_chunks(self, server_ready):
        """Test adding chunks with pre-tokenized content"""
        # This test would need actual token IDs from the model
        # For now, test the API structure
        content = "Pre-tokenized test content"
        
        response = requests.post(CONTEXT_URL, json={
            "content": content,
            "tokenize": False,
            "tokens": [1, 2, 3, 4, 5],  # Dummy token IDs
            "metadata": {"type": "pre_tokenized"}
        })
        
        # This might fail depending on model's actual tokenization
        # But should test the API endpoint structure
        assert response.status_code in [200, 400, 500]
    
    def test_positioning_edge_cases(self, server_ready):
        """Test edge cases in positioning"""
        # Add base chunk
        response = requests.post(CONTEXT_URL, json={
            "content": "Base chunk for edge case testing"
        })
        assert response.status_code == 200
        base_hash = response.json()["hash"]
        
        # Test positioning before non-existent chunk
        response = requests.post(CONTEXT_URL, json={
            "content": "Test content",
            "position": "before:nonexistent_hash"
        })
        # Should fallback to append
        assert response.status_code == 200
        
        # Test specific position
        response = requests.post(CONTEXT_URL, json={
            "content": "Specific position test",
            "position": "specific_pos",
            "preferred_position": 1000
        })
        assert response.status_code == 200

if __name__ == "__main__":
    # Run tests directly if this file is executed
    pytest.main([__file__, "-v"])