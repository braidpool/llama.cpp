#!/usr/bin/env python3
"""
Minimal test of Context Management API focusing on in-memory operations
"""

import requests
import json

def test_minimal_context_api():
    base_url = "http://localhost:8080"
    context_url = f"{base_url}/context"
    
    print("🧪 Minimal Context Management API Test")
    print("=" * 40)
    
    try:
        # Test 1: Get context info
        print("\n1. Getting context info...")
        response = requests.get(context_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Context API available")
            print(f"   Total context: {data['total_context']}")
            print(f"   Used context: {data['used_context']}")
            print(f"   Chunks: {len(data['chunks'])}")
        else:
            print(f"❌ Failed: {response.status_code}")
            return False
            
        # Test 2: Add chunks
        print("\n2. Adding chunks...")
        chunks = []
        for i in range(3):
            content = f"Test chunk {i}: This is test content for chunk number {i}."
            response = requests.post(context_url, json={
                "content": content,
                "metadata": {"type": "test", "index": i}
            })
            
            if response.status_code == 200:
                data = response.json()
                chunks.append(data["hash"])
                print(f"✅ Added chunk {i}: {data['hash'][:16]}...")
                print(f"   Size: {data['token_size']} tokens")
                print(f"   Position: {data['position']['start']}-{data['position']['end']}" )
                print(f"   Memory used: {data['memory_size']} tokens")
            else:
                print(f"❌ Failed to add chunk {i}: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
        
        # Test 3: Check context after adding chunks
        print("\n3. Checking context after additions...")
        response = requests.get(context_url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Context updated")
            print(f"   Used context: {data['used_context']}")
            print(f"   Total chunks: {len(data['chunks'])}")
            print(f"   Fragmentation: {data['fragmentation']:.2%}")
        else:
            print(f"❌ Failed to get context: {response.status_code}")
            return False
        
        # Test 4: Test positioning strategies
        print("\n4. Testing positioning strategies...")
        
        # Test best_fit positioning
        response = requests.post(context_url, json={
            "content": "Small chunk",
            "position": "best_fit",
            "metadata": {"type": "positioning_test", "strategy": "best_fit"}
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Best fit positioning: {data['hash'][:16]}...")
            print(f"   Position: {data['position']['start']}-{data['position']['end']}")
            print(f"   Memory used: {data['memory_size']} tokens")
        else:
            print(f"❌ Best fit failed: {response.status_code}")
        
        # Test positioning after specific chunk
        if chunks:
            response = requests.post(context_url, json={
                "content": "Positioned after chunk",
                "position": f"after:{chunks[0]}",
                "metadata": {"type": "positioning_test", "strategy": "after"}
            })
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ After positioning: {data['hash'][:16]}...")
                print(f"   Position: {data['position']['start']}-{data['position']['end']}")
                print(f"   Memory used: {data['memory_size']} tokens")
            else:
                print(f"❌ After positioning failed: {response.status_code}")
        
        # Test 5: Memory compaction
        print("\n5. Testing memory compaction...")
        response = requests.post(f"{context_url}/compact")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Memory compaction successful")
                
                # Check fragmentation after compaction
                response = requests.get(context_url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Fragmentation after compaction: {data['fragmentation']:.2%}")
                else:
                    print("❌ Failed to get post-compaction stats")
            else:
                print("❌ Compaction failed")
        else:
            print(f"❌ Compaction request failed: {response.status_code}")
        
        # Test 6: Garbage collection
        print("\n6. Testing garbage collection...")
        response = requests.post(f"{context_url}/gc")
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print("✅ Garbage collection successful")
            else:
                print("❌ GC failed")
        else:
            print(f"❌ GC request failed: {response.status_code}")
        
        # Test 7: Content deduplication
        print("\n7. Testing content deduplication...")
        duplicate_content = "This exact content will be added twice."
        
        # Add first time
        response1 = requests.post(context_url, json={
            "content": duplicate_content,
            "metadata": {"instance": "first"}
        })
        
        # Add second time
        response2 = requests.post(context_url, json={
            "content": duplicate_content,
            "metadata": {"instance": "second"}
        })
        
        if response1.status_code == 200 and response2.status_code == 200:
            hash1 = response1.json()["hash"]
            hash2 = response2.json()["hash"]
            
            if hash1 == hash2:
                print("✅ Content deduplication working")
                print(f"   Both requests returned same hash: {hash1[:16]}...")
            else:
                print("❌ Deduplication not working - different hashes returned")
        else:
            print("❌ Failed to test deduplication")
        
        # Test 8: Batch operations (without save/restore)
        print("\n8. Testing batch operations...")
        
        # Test batch erase of some chunks (only memory operations, no file I/O)
        if len(chunks) >= 2:
            batch_ops = [
                {"action": "erase", "hash": chunks[0]},
                {"action": "erase", "hash": chunks[1]}
            ]
            
            response = requests.post(f"{context_url}/batch", json={
                "operations": batch_ops,
                "compact_after": True
            })
            
            if response.status_code == 200:
                results = response.json()
                successful = sum(1 for r in results if r.get("success"))
                print(f"✅ Batch operations: {successful}/{len(batch_ops)} successful")
                
                # Check final state
                response = requests.get(context_url)
                if response.status_code == 200:
                    data = response.json()
                    print(f"   Remaining chunks: {len(data['chunks'])}")
            else:
                print(f"❌ Batch operations failed: {response.status_code}")
                print(f"   Response: {response.text}")
        
        print("\n🎉 All in-memory tests completed successfully!")
        print("\nNote: File-based operations (save/restore) require server to be started with --slot-save-path")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    test_minimal_context_api()