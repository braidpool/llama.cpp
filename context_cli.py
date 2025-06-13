#!/usr/bin/env python3
"""
Context Management CLI - Interactive demonstration of the Context Management API

This TUI provides a REPL interface to interact with the context management features
of llama.cpp server, including loading chunks, managing context, and chatting with
the model while maintaining context in the KV cache.
"""

import cmd
import json
import os
import sys
import time
import requests
from typing import Dict, List, Any, Optional

class ContextCLI(cmd.Cmd):
    """Interactive CLI for Context Management API"""

    def __init__(self, base_url: str = "http://localhost:8080"):
        super().__init__()
        self.base_url = base_url
        self.context_url = f"{base_url}/context"
        self.chat_url = f"{base_url}/v1/chat/completions"

        # ANSI color codes
        self.RESET = '\033[0m'
        self.BOLD = '\033[1m'
        self.GREEN = '\033[92m'
        self.RED = '\033[91m'
        self.YELLOW = '\033[93m'
        self.BLUE = '\033[94m'
        self.GRAY = '\033[90m'

        # Context status
        self.context_info = self._get_context_info()
        self._update_prompt()

        # Configure cmd settings
        self.use_rawinput = True
        self.doc_header = "Commands (type help <command>):"
        self.ruler = "-"

        print(f"{self.BOLD}Context Management CLI{self.RESET}")
        print(f"Connected to: {self.base_url}")
        print(f"Type 'help' for available commands or '/context' to see current chunks")
        print(f"{self.YELLOW}Note: Server must be started with --context-manager flag for context to work properly{self.RESET}")
        print(f"{self.YELLOW}      For save/restore features, also add --slot-save-path <directory>{self.RESET}")
        print(f"{self.YELLOW}      LIMITATION: Active/inactive states are tracked but not yet enforced during inference{self.RESET}\n")

    def _get_context_info(self) -> Dict[str, Any]:
        """Get current context information"""
        try:
            response = requests.get(self.context_url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"{self.RED}Failed to get context info: {response.status_code}{self.RESET}")
                return {"chunks": [], "used_context": 0, "total_context": 40960}
        except Exception as e:
            print(f"{self.RED}Error connecting to server: {e}{self.RESET}")
            return {"chunks": [], "used_context": 0, "total_context": 40960}

    def _update_prompt(self):
        """Update the prompt with current context status"""
        info = self.context_info
        chunks = len(info.get("chunks", []))
        used = info.get("used_context", 0)
        total = info.get("total_context", 40960)
        percent = (used / total * 100) if total > 0 else 0

        status_line = f"{self.GRAY}[Context: {chunks} chunks, {used}/{total} tokens, {percent:.0f}% full]{self.RESET}"
        self.prompt = f"{status_line}\n> "

    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """Make HTTP request with error handling"""
        try:
            response = requests.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"{self.RED}Request failed: {e}{self.RESET}")
            return None

    def do_load(self, arg: str):
        """Load a file as a context chunk: /load <filename>"""
        if not arg:
            print(f"{self.RED}Usage: /load <filename>{self.RESET}")
            return

        filename = os.path.expanduser(arg.strip())
        if not os.path.exists(filename):
            print(f"{self.RED}File not found: {filename}{self.RESET}")
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"Loading {filename}...")
            response = self._make_request("POST", self.context_url, json={
                "content": f"<context filename=\"{filename}\">\n" + content + "\n</context>\n",
                "metadata": {"source": filename}
            })

            if response:
                data = response.json()
                print(f"{self.GREEN}✓ Added chunk:{self.RESET}")
                print(f"  Hash: {data['hash']}")
                print(f"  Size: {data.get('token_size', data.get('size', 0))} tokens")
                print(f"  Position: {data['position']['start']}-{data['position']['end']}")

                # Update context info
                self.context_info = self._get_context_info()
                self._update_prompt()

        except Exception as e:
            print(f"{self.RED}Error loading file: {e}{self.RESET}")

    def do_save(self, arg: str):
        """Save a chunk to disk: /save <hash>"""
        if not arg:
            print(f"{self.RED}Usage: /save <hash>{self.RESET}")
            return

        hash_value = arg.strip()
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=save")

        if response:
            print(f"{self.GREEN}✓ Chunk saved: {hash_value}{self.RESET}")
        else:
            print(f"{self.YELLOW}Note: The save feature requires the server to be started with --slot-save-path <directory>{self.RESET}")
            print(f"{self.YELLOW}Example: ./llama-server --context-manager --slot-save-path ./saved_contexts{self.RESET}")

    def do_restore(self, arg: str):
        """Restore a saved chunk: /restore <hash>"""
        if not arg:
            print(f"{self.RED}Usage: /restore <hash>{self.RESET}")
            return

        hash_value = arg.strip()
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=restore")

        if response:
            data = response.json()
            print(f"{self.GREEN}✓ Chunk restored: {hash_value}{self.RESET}")
            print(f"  Size: {data.get('token_size', data.get('size', 0))} tokens")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.YELLOW}Note: The restore feature requires the server to be started with --slot-save-path <directory>{self.RESET}")
            print(f"{self.YELLOW}Example: ./llama-server --context-manager --slot-save-path ./saved_contexts{self.RESET}")

    def do_erase(self, arg: str):
        """Erase a chunk from context: /erase <hash>"""
        if not arg:
            print(f"{self.RED}Usage: /erase <hash>{self.RESET}")
            return

        hash_value = arg.strip()
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=erase")

        if response:
            print(f"{self.GREEN}✓ Chunk erased: {hash_value}{self.RESET}")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()

    def do_compact(self, arg: str):
        """Compact memory to reduce fragmentation"""
        response = self._make_request("POST", f"{self.context_url}/compact")

        if response:
            data = response.json()
            print(f"{self.GREEN}✓ Memory compacted{self.RESET}")
            print(f"  Fragmentation: {data.get('fragmentation_before', 0):.2%} → {data.get('fragmentation_after', 0):.2%}")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()

    def do_gc(self, arg: str):
        """Run garbage collection on saved chunks"""
        response = self._make_request("POST", f"{self.context_url}/gc")

        if response:
            data = response.json()
            print(f"{self.GREEN}✓ Garbage collection completed{self.RESET}")
            print(f"  Freed: {data.get('chunks_freed', 0)} chunks")

    def do_context(self, arg: str):
        """Display current context chunks in tabular format"""
        self.context_info = self._get_context_info()
        info = self.context_info

        print(f"\n{self.BOLD}Context Status:{self.RESET}")
        print(f"  Total: {info.get('total_context', 0)} tokens")
        print(f"  Used: {info.get('used_context', 0)} tokens")
        print(f"  Free: {info.get('free_context', 0)} tokens")
        print(f"  Fragmentation: {info.get('fragmentation', 0):.2%}")

        chunks = info.get('chunks', [])
        if chunks:
            # Separate chunks by status
            active_chunks = [c for c in chunks if c.get('status') == 'active']
            inactive_chunks = [c for c in chunks if c.get('status') == 'inactive']
            system_chunks = [c for c in chunks if c.get('status') == 'system']
            disk_chunks = [c for c in chunks if c.get('status') == 'disk']
            
            # Display active chunks (will be used for next prompt)
            if active_chunks:
                sorted_active = sorted(active_chunks, key=lambda c: c.get('position', {}).get('start', 0))
                
                print(f"\n{self.BOLD}{self.GREEN}Active Chunks (in VRAM, will be used for next prompt):{self.RESET}")
                print("-" * 115)
                print(f"{'#':>3}  {'Hash':<64}  {'Size':>6}  {'Position':>12}  {'Status':<8}  {'Preview':<10}")
                print("-" * 115)

                for idx, chunk in enumerate(sorted_active, 1):
                    hash_val = chunk.get('hash', 'unknown')
                    size = chunk.get('token_size', chunk.get('size', 0))
                    pos_start = chunk.get('position', {}).get('start', 0)
                    pos_end = chunk.get('position', {}).get('end', 0)
                    position = f"{pos_start}-{pos_end}"
                    status = chunk.get('status', 'unknown')
                    preview = self._get_chunk_preview(chunk)

                    print(f"{idx:>3}  {hash_val:<64}  {size:>6}  {position:>12}  {self.GREEN}{status:<8}{self.RESET}  {preview:<10}")

                print("-" * 115)
                print(f"Total active: {len(active_chunks)} chunks")
            
            # Display inactive chunks (in VRAM but excluded from prompts)
            if inactive_chunks:
                sorted_inactive = sorted(inactive_chunks, key=lambda c: c.get('position', {}).get('start', 0))
                
                print(f"\n{self.BOLD}{self.YELLOW}Inactive Chunks (in VRAM but excluded from prompts):{self.RESET}")
                print("-" * 115)
                print(f"{'#':>3}  {'Hash':<64}  {'Size':>6}  {'Position':>12}  {'Status':<8}  {'Preview':<10}")
                print("-" * 115)

                for idx, chunk in enumerate(sorted_inactive, 1):
                    hash_val = chunk.get('hash', 'unknown')
                    size = chunk.get('token_size', chunk.get('size', 0))
                    pos_start = chunk.get('position', {}).get('start', 0)
                    pos_end = chunk.get('position', {}).get('end', 0)
                    position = f"{pos_start}-{pos_end}"
                    status = chunk.get('status', 'unknown')
                    preview = self._get_chunk_preview(chunk)

                    print(f"{idx:>3}  {hash_val:<64}  {size:>6}  {position:>12}  {self.YELLOW}{status:<8}{self.RESET}  {preview:<10}")

                print("-" * 115)
                print(f"Total inactive: {len(inactive_chunks)} chunks")
            
            # Show system RAM chunks if any
            if system_chunks:
                print(f"\n{self.BOLD}{self.BLUE}System RAM Chunks (offloaded from VRAM, use /restore to reload):{self.RESET}")
                print("-" * 90)
                print(f"{'Hash':<64}  {'Size':>6}  {'Status':<8}  {'Preview':<10}")
                print("-" * 90)
                
                for chunk in system_chunks:
                    hash_val = chunk.get('hash', 'unknown')
                    size = chunk.get('token_size', chunk.get('size', 0))
                    status = chunk.get('status', 'unknown')
                    preview = self._get_chunk_preview(chunk)
                    
                    print(f"{hash_val:<64}  {size:>6}  {self.BLUE}{status:<8}{self.RESET}  {preview:<10}")
                
                print("-" * 90)
                print(f"Total in system RAM: {len(system_chunks)} chunks")
            
            # Show disk chunks if any
            if disk_chunks:
                print(f"\n{self.BOLD}Disk Chunks (saved to disk, use /restore to load):{self.RESET}")
                print("-" * 90)
                print(f"{'Hash':<64}  {'Size':>6}  {'Status':<8}  {'Preview':<10}")
                print("-" * 90)
                
                for chunk in disk_chunks:
                    hash_val = chunk.get('hash', 'unknown')
                    size = chunk.get('token_size', chunk.get('size', 0))
                    status = chunk.get('status', 'unknown')
                    preview = self._get_chunk_preview(chunk)
                    
                    print(f"{hash_val:<64}  {size:>6}  {status:<8}  {preview:<10}")
                
                print("-" * 90)
                print(f"Total on disk: {len(disk_chunks)} chunks")
            
        else:
            print(f"\n{self.GRAY}No chunks loaded{self.RESET}")

        print()
    
    def _get_chunk_preview(self, chunk):
        """Helper to get chunk preview text"""
        metadata = chunk.get('metadata', {})
        if 'source' in metadata:
            # Show filename if loaded from a file
            return f"[{os.path.basename(metadata['source'])[:8]}]"
        elif 'content' in chunk:
            # If content is available in chunk data
            content = chunk['content']
            return content[:10].replace('\n', '\\n') if content else '...'
        else:
            return '...'

    def do_clear(self, arg: str):
        """Clear all chunks from context"""
        # Get current chunks
        response = self._make_request("GET", self.context_url)
        if not response:
            return

        data = response.json()
        chunks = data.get("chunks", [])

        if not chunks:
            print(f"{self.GRAY}Context is already empty{self.RESET}")
            return

        # Create erase operations for all chunks
        erase_ops = [{"action": "erase", "hash": chunk["hash"]} for chunk in chunks]

        print(f"Clearing {len(chunks)} chunks...")
        response = self._make_request("POST", f"{self.context_url}/batch", json={"operations": erase_ops})

        if response:
            print(f"{self.GREEN}✓ Context cleared: removed {len(chunks)} chunks{self.RESET}")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()

    def do_activate(self, arg: str):
        """Activate an inactive chunk: /activate <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /activate <hash_or_number>{self.RESET}")
            return
            
        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return
            
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=activate")
        
        if response:
            print(f"{self.GREEN}✓ Chunk activated: {hash_value}{self.RESET}")
            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.RED}Failed to activate chunk (it may already be active or not in VRAM){self.RESET}")
    
    def do_deactivate(self, arg: str):
        """Deactivate an active chunk: /deactivate <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /deactivate <hash_or_number>{self.RESET}")
            return
            
        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return
            
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=deactivate")
        
        if response:
            print(f"{self.GREEN}✓ Chunk deactivated: {hash_value}{self.RESET}")
            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.RED}Failed to deactivate chunk (it may already be inactive or not in VRAM){self.RESET}")
    
    def do_unload(self, arg: str):
        """Unload a chunk to system RAM: /unload <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /unload <hash_or_number>{self.RESET}")
            return
            
        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return
            
        response = self._make_request("POST", f"{self.context_url}/{hash_value}?action=unload")
        
        if response:
            print(f"{self.GREEN}✓ Chunk unloaded to system RAM: {hash_value}{self.RESET}")
            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.RED}Failed to unload chunk (it must be in VRAM to unload){self.RESET}")
    
    def _resolve_chunk_identifier(self, identifier: str) -> str:
        """Resolve a chunk identifier (number or hash) to a hash value"""
        try:
            chunk_num = int(identifier)
            # Get the chunk list and find by index
            chunks = self.context_info.get('chunks', [])
            if chunks:
                # Sort all chunks by position (for consistent numbering)
                all_chunks = sorted(chunks, key=lambda c: (
                    0 if c.get('status') == 'active' else
                    1 if c.get('status') == 'inactive' else
                    2 if c.get('status') == 'system' else
                    3,  # disk
                    c.get('position', {}).get('start', 0)
                ))
                if 1 <= chunk_num <= len(all_chunks):
                    return all_chunks[chunk_num - 1].get('hash')
                else:
                    print(f"{self.RED}Invalid chunk number. Valid range: 1-{len(all_chunks)}{self.RESET}")
                    return None
            else:
                print(f"{self.GRAY}No chunks available{self.RESET}")
                return None
        except ValueError:
            # Not a number, treat as hash
            return identifier

    def do_move(self, arg: str):
        """Move a chunk to a different position: /move <position> <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /move <position> <hash_or_number>{self.RESET}")
            print(f"  Position can be: first, last, or a number (1-based)")
            return
            
        parts = arg.strip().split(None, 1)
        if len(parts) != 2:
            print(f"{self.RED}Usage: /move <position> <hash_or_number>{self.RESET}")
            return
            
        position_arg, identifier = parts
        
        # Get current chunks
        self.context_info = self._get_context_info()
        chunks = self.context_info.get('chunks', [])
        if not chunks:
            print(f"{self.GRAY}No chunks loaded{self.RESET}")
            return
            
        # Sort chunks by position
        sorted_chunks = sorted(chunks, key=lambda c: c.get('position', {}).get('start', 0))
        
        # Resolve hash from identifier
        hash_value = self._resolve_chunk_identifier(identifier)
        if not hash_value:
            return
            
        # Determine target position
        target_position = None
        if position_arg.lower() == 'first':
            target_position = 0
        elif position_arg.lower() == 'last':
            target_position = len(sorted_chunks) - 1
        else:
            try:
                pos = int(position_arg) - 1  # Convert to 0-based
                if 0 <= pos < len(sorted_chunks):
                    target_position = pos
                else:
                    print(f"{self.RED}Invalid position. Valid range: 1-{len(sorted_chunks)} (or 'first'/'last'){self.RESET}")
                    return
            except ValueError:
                print(f"{self.RED}Invalid position. Use a number, 'first', or 'last'{self.RESET}")
                return
                
        # Find current position of the chunk
        current_position = None
        for i, chunk in enumerate(sorted_chunks):
            if chunk.get('hash') == hash_value:
                current_position = i
                break
                
        if current_position is None:
            print(f"{self.RED}Chunk not found: {hash_value}{self.RESET}")
            return
            
        if current_position == target_position:
            print(f"{self.YELLOW}Chunk is already at position {target_position + 1}{self.RESET}")
            return
            
        # Build the new order
        new_order = []
        chunk_to_move = sorted_chunks[current_position]
        
        # Remove chunk from current position
        remaining_chunks = sorted_chunks[:current_position] + sorted_chunks[current_position + 1:]
        
        # Insert at target position
        if target_position >= current_position:
            # Moving forward, adjust for removal
            new_order = remaining_chunks[:target_position] + [chunk_to_move] + remaining_chunks[target_position:]
        else:
            # Moving backward
            new_order = remaining_chunks[:target_position] + [chunk_to_move] + remaining_chunks[target_position:]
            
        # Create batch operations to reorder
        operations = []
        
        # First, erase all chunks
        for chunk in sorted_chunks:
            operations.append({
                "action": "erase",
                "hash": chunk["hash"]
            })
            
        # Then restore in new order
        for chunk in new_order:
            operations.append({
                "action": "restore",
                "hash": chunk["hash"]
            })
            
        print(f"Reordering chunks...")
        response = self._make_request("POST", f"{self.context_url}/batch", json={"operations": operations})
        
        if response:
            result = response.json()
            print(f"{self.GREEN}✓ Chunks reordered successfully{self.RESET}")
            print(f"  Moved chunk from position {current_position + 1} to position {target_position + 1}")
            
            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
            
            # Show new order
            self.do_context("")
        else:
            print(f"{self.RED}Failed to reorder chunks{self.RESET}")

    def do_execute(self, arg: str):
        """Execute inference on the loaded context (no additional prompt)"""
        print(f"{self.YELLOW}Executing inference on loaded context...{self.RESET}")

        try:
            # Use the completion endpoint for pure inference (no chat)
            completion_url = f"{self.base_url}/completion"

            # Empty prompt - just process what's in context
            data = {
                "prompt": "",
                "n_predict": 512,  # Generate up to 512 tokens
                "temperature": 0.7,
                "stream": True
            }

            # Make streaming request
            response = requests.post(
                completion_url,
                json=data,
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=60
            )

            if response.status_code != 200:
                print(f"{self.RED}Execution failed: {response.status_code}{self.RESET}")
                try:
                    error_data = response.json()
                    print(f"{self.RED}Error: {error_data}{self.RESET}")
                except:
                    pass
                return

            print(f"{self.BLUE}Output:{self.RESET} ", end='', flush=True)

            # Process SSE stream
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix

                        try:
                            data = json.loads(data_str)
                            content = data.get('content', '')

                            if content:
                                print(content, end='', flush=True)

                            # Check if we should stop
                            if data.get('stop', False):
                                break

                        except json.JSONDecodeError:
                            continue

            print("\n")  # New line after response

        except Exception as e:
            print(f"\n{self.RED}Error during execution: {e}{self.RESET}")

    def do_show(self, arg: str):
        """Show the details and content of a specific chunk: /show <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /show <hash_or_number>{self.RESET}")
            return

        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return

        # Get chunk details and content using the new GET /context/{hash} endpoint
        response = self._make_request("GET", f"{self.context_url}/{hash_value}")

        if response:
            data = response.json()
            print(f"\n{self.BOLD}Chunk Details:{self.RESET}")
            print(f"  Hash: {data.get('hash', hash_value)}")
            print(f"  Size: {data.get('token_size', data.get('size', 0))} tokens")
            position = data.get('position', {})
            print(f"  Position: {position.get('start', 0)}-{position.get('end', 0)}")

            metadata = data.get('metadata', {})
            if metadata:
                print(f"  Metadata:")
                for key, value in metadata.items():
                    print(f"    {key}: {value}")

            # Show the content
            content = data.get('content', '')
            print(f"\n{self.BOLD}Content:{self.RESET}")
            print("-" * 80)
            if content:
                print(content)
            else:
                print(f"{self.GRAY}(no content available){self.RESET}")
            print("-" * 80)
            print()
        else:
            print(f"{self.RED}Chunk not found: {hash_value}{self.RESET}")

    def _stream_chat_completion(self, message: str):
        """Send a chat completion request and stream the response"""
        try:
            # Prepare the request
            data = {
                "model": "gpt-3.5-turbo",  # Model name doesn't matter for llama.cpp
                "messages": [{"role": "user", "content": message}],
                "stream": True,
                "temperature": 0.7
            }

            # Make streaming request
            response = requests.post(
                self.chat_url,
                json=data,
                stream=True,
                headers={"Accept": "text/event-stream"},
                timeout=60
            )

            if response.status_code != 200:
                print(f"{self.RED}Chat request failed: {response.status_code}{self.RESET}")
                return

            print(f"{self.BLUE}Assistant:{self.RESET} ", end='', flush=True)

            # Process SSE stream
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]  # Remove 'data: ' prefix
                        if data_str == '[DONE]':
                            break

                        try:
                            data = json.loads(data_str)
                            choice = data.get('choices', [{}])[0]
                            delta = choice.get('delta', {})
                            content = delta.get('content', '')

                            if content:
                                print(content, end='', flush=True)

                        except json.JSONDecodeError:
                            continue

            print("\n")  # New line after response

        except Exception as e:
            print(f"\n{self.RED}Error during chat: {e}{self.RESET}")

    def default(self, line: str):
        """Handle regular text input as chat messages"""
        if line.strip():
            # Check if it's a slash command
            if line.startswith('/'):
                # Parse slash command
                parts = line.split(None, 1)
                command = parts[0][1:]  # Remove the leading slash
                args = parts[1] if len(parts) > 1 else ''

                # Map slash commands to their handlers
                command_map = {
                    'load': self.do_load,
                    'save': self.do_save,
                    'restore': self.do_restore,
                    'erase': self.do_erase,
                    'activate': self.do_activate,
                    'deactivate': self.do_deactivate,
                    'unload': self.do_unload,
                    'compact': self.do_compact,
                    'gc': self.do_gc,
                    'context': self.do_context,
                    'clear': self.do_clear,
                    'show': self.do_show,
                    'move': self.do_move,
                    'execute': self.do_execute,
                }

                if command in command_map:
                    command_map[command](args)
                else:
                    print(f"{self.RED}Unknown command: /{command}{self.RESET}")
                    print("Available commands: /load, /save, /restore, /erase, /activate, /deactivate, /unload, /compact, /gc, /context, /clear, /show, /move, /execute")
            else:
                # Send as chat message
                self._stream_chat_completion(line)

    def do_help(self, arg: str):
        """Show help for commands"""
        if not arg:
            print(f"\n{self.BOLD}Available Commands:{self.RESET}")
            print(f"\n{self.BOLD}Basic Operations:{self.RESET}")
            print("  /load <file>         - Load a file as a context chunk (state: ACTIVE)")
            print("  /clear               - Clear all chunks from context")
            print("  /context             - Display current context chunks by state")
            print("  /show <hash|num>     - Show content of chunk (by hash or number)")
            
            print(f"\n{self.BOLD}State Management:{self.RESET}")
            print("  /activate <hash|num>   - Activate chunk (include in prompts)")
            print("  /deactivate <hash|num> - Deactivate chunk (exclude from prompts)")
            print("  /unload <hash|num>     - Move chunk from VRAM to system RAM")
            print("  /restore <hash|num>    - Restore chunk from disk/RAM to VRAM (ACTIVE)")
            
            print(f"\n{self.BOLD}Storage Operations:{self.RESET}")
            print("  /save <hash|num>     - Save chunk to disk (requires --slot-save-path)")
            print("  /erase <hash|num>    - Permanently erase a chunk")
            
            print(f"\n{self.BOLD}Advanced:{self.RESET}")
            print("  /move <pos> <hash|num> - Move chunk to position (first/last/number)")
            print("  /execute             - Execute inference on active chunks only")
            print("  /compact             - Compact memory to reduce fragmentation")
            print("  /gc                  - Run garbage collection on saved chunks")
            
            print(f"\n{self.BOLD}Other:{self.RESET}")
            print("  help                 - Show this help message")
            print("  exit/quit            - Exit the CLI")
            print("\nAnything else will be sent as a chat message to the model.\n")
        else:
            super().do_help(arg)

    def do_exit(self, arg: str):
        """Exit the CLI"""
        print(f"\n{self.GRAY}Goodbye!{self.RESET}")
        return True

    def do_quit(self, arg: str):
        """Exit the CLI"""
        return self.do_exit(arg)

    def do_EOF(self, arg: str):
        """Handle Ctrl+D"""
        print()  # New line after ^D
        return self.do_exit(arg)

    def emptyline(self):
        """Do nothing on empty line"""
        pass

    def cmdloop(self, intro=None):
        """Override cmdloop to handle KeyboardInterrupt"""
        while True:
            try:
                super().cmdloop(intro)
                break
            except KeyboardInterrupt:
                print(f"\n{self.GRAY}(Use 'exit' or 'quit' to leave){self.RESET}")
                self._update_prompt()

def main():
    """Run the Context Management CLI"""
    import argparse

    parser = argparse.ArgumentParser(description="Context Management CLI")
    parser.add_argument(
        "--url",
        default="http://localhost:8080",
        help="Base URL of the llama.cpp server (default: http://localhost:8080)"
    )

    args = parser.parse_args()

    try:
        cli = ContextCLI(base_url=args.url)
        cli.cmdloop()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
