#!/usr/bin/env python3
"""
Simple test script to verify Context Management API functionality
Run this after starting the llama-server to test basic operations.
"""

import requests
import json
import time

def test_context_api():
    base_url = "http://localhost:8080"
    context_url = f"{base_url}/context"
    
    print("🧪 Testing Context Management API")
    print("=" * 40)
    
    try:
        # Test 1: Get initial context info
        print("\n1. Getting initial context info...")
        response = requests.get(context_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Context API available")
            print(f"   Total context: {data['total_context']}")
            print(f"   Used context: {data['used_context']}")
            print(f"   Chunks: {len(data['chunks'])}")
        else:
            print(f"❌ Failed to get context info: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ Server not accessible: {e}")
        print("Make sure llama-server is running with: ./llama-server -m <model>")
        return False
    
    try:
        # Test 2: Add a simple chunk
        print("\n2. Adding a simple chunk...")
        test_content = "Hello, this is a test chunk for the context management API."
        response = requests.post(context_url, json={
            "content": test_content,
            "metadata": {"type": "test", "description": "Simple test chunk"}
        })
        
        if response.status_code == 200:
            data = response.json()
            chunk_hash = data["hash"]
            print(f"✅ Chunk added successfully")
            print(f"   Hash: {chunk_hash[:16]}...")
            print(f"   Size: {data['size']} tokens")
            print(f"   Position: {data['position']['start']}-{data['position']['end']}")
        else:
            print(f"❌ Failed to add chunk: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
        
        # Test 3: Save chunk
        print("\n3. Saving chunk to disk...")
        response = requests.post(f"{context_url}/{chunk_hash}?action=save")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Chunk saved successfully")
            else:
                print("❌ Save operation failed")
                return False
        else:
            print(f"❌ Save request failed: {response.status_code}")
            return False
        
        # Test 4: Restore chunk
        print("\n4. Restoring chunk from disk...")
        response = requests.post(f"{context_url}/{chunk_hash}?action=restore")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Chunk restored successfully")
            else:
                print("❌ Restore operation failed")
                return False
        else:
            print(f"❌ Restore request failed: {response.status_code}")
            return False
        
        # Test 5: Check final context state
        print("\n5. Checking final context state...")
        response = requests.get(context_url)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Final state retrieved")
            print(f"   Total chunks: {len(data['chunks'])}")
            print(f"   Loaded chunks: {data['loaded_chunks']}")
            print(f"   Saved chunks: {data['saved_chunks']}")
            print(f"   Fragmentation: {data['fragmentation']:.2%}")
        else:
            print(f"❌ Failed to get final state: {response.status_code}")
            return False
        
        # Test 6: Memory compaction
        print("\n6. Testing memory compaction...")
        response = requests.post(f"{context_url}/compact")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Memory compaction successful")
            else:
                print("❌ Compaction failed")
                return False
        else:
            print(f"❌ Compaction request failed: {response.status_code}")
            return False
        
        # Test 7: Garbage collection
        print("\n7. Testing garbage collection...")
        response = requests.post(f"{context_url}/gc")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Garbage collection successful")
            else:
                print("❌ Garbage collection failed")
                return False
        else:
            print(f"❌ GC request failed: {response.status_code}")
            return False
        
        # Test 8: Cleanup - erase test chunk
        print("\n8. Cleaning up test chunk...")
        response = requests.post(f"{context_url}/{chunk_hash}?action=erase")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Test chunk erased successfully")
            else:
                print("❌ Erase operation failed")
                return False
        else:
            print(f"❌ Erase request failed: {response.status_code}")
            return False
        
        print("\n🎉 All tests passed! Context Management API is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def test_batch_operations():
    """Test batch operations"""
    print("\n🔄 Testing batch operations...")
    
    context_url = "http://localhost:8080/context"
    
    # Add multiple chunks
    chunk_hashes = []
    for i in range(3):
        content = f"Batch test chunk {i}: Content for testing batch operations."
        response = requests.post(context_url, json={
            "content": content,
            "metadata": {"type": "batch_test", "index": i}
        })
        
        if response.status_code == 200:
            chunk_hashes.append(response.json()["hash"])
        else:
            print(f"❌ Failed to add batch chunk {i}")
            return False
    
    print(f"✅ Added {len(chunk_hashes)} chunks for batch testing")
    
    # Test batch save
    batch_ops = [{"action": "save", "hash": h} for h in chunk_hashes[:2]]
    response = requests.post(f"{context_url}/batch", json={
        "operations": batch_ops,
        "compact_after": True
    })
    
    if response.status_code == 200:
        results = response.json()
        successful = sum(1 for r in results if r.get("success"))
        print(f"✅ Batch save: {successful}/{len(batch_ops)} operations successful")
    else:
        print(f"❌ Batch operation failed: {response.status_code}")
        return False
    
    # Cleanup batch test chunks
    erase_ops = [{"action": "erase", "hash": h} for h in chunk_hashes]
    response = requests.post(f"{context_url}/batch", json={
        "operations": erase_ops
    })
    
    if response.status_code == 200:
        print("✅ Batch cleanup completed")
    else:
        print("❌ Batch cleanup failed")
        return False
    
    return True

def main():
    """Run all tests"""
    print("Starting Context Management API Tests")
    print("Make sure llama-server is running first!")
    print()
    
    # Wait a moment to ensure server is ready
    time.sleep(1)
    
    success = True
    
    # Run basic API tests
    if not test_context_api():
        success = False
    
    # Run batch operation tests
    if not test_batch_operations():
        success = False
    
    if success:
        print("\n🎊 All tests completed successfully!")
        print("\nYou can now:")
        print("- Run the comprehensive example: python3 llama_context.py")
        print("- Run the full test suite: python3 -m pytest tools/server/tests/unit/test_context_management.py -v")
        print("- Integrate the API into your own applications")
    else:
        print("\n💥 Some tests failed. Check the server configuration and try again.")

if __name__ == "__main__":
    main()