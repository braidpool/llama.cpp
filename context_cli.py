#!/usr/bin/env python3
"""
Context Management CLI - Interactive demonstration of the Context Management API

This TUI provides a REPL interface to interact with the context management features
of llama.cpp server, including loading chunks, managing context, and chatting with
the model while maintaining context in the KV cache.

Features:
- Real-time markdown rendering of LLM responses using Rich library
- Context chunk management (load, save, restore, activate/deactivate)
- Interactive chat with context-aware responses
- Rich formatted output and status displays
"""

import cmd
import json
import os
import sys
import time
import requests
from typing import Dict, List, Any, Optional
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text
from rich.console import Group

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

        # Rich console for markdown rendering
        self.console = Console()

        # Context status
        try:
            import shutil
            self.terminal_width = shutil.get_terminal_size().columns
        except:
            self.terminal_width = 80  # fallback
        self.context_info = self._get_context_info()
        self._update_prompt()

        # Configure cmd settings
        self.use_rawinput = True
        self.doc_header = "Commands (type help <command>):"
        self.ruler = "-"

        # Thinking management
        self.thinking = False           # Track if we're within <think>...</think>
        self.thought_buffer: str = ""   # Buffer for content between <think>

        self.console.print(f"[bold]Context Management CLI[/bold]")
        self.console.print(f"Connected to: {self.base_url}")
        self.console.print(f"Type 'help' for available commands or '/context' to see current chunks")
        self.console.print(f"[yellow]Note: Server must be started with --context-manager flag for context to work properly[/yellow]")
        self.console.print(f"[yellow]      For save/restore features, also add --slot-save-path <directory>[/yellow]")
        self.console.print(f"[green]      ✓ Active/inactive states are now properly enforced during inference[/green]\n")

    def _get_context_info(self) -> Dict[str, Any]:
        """Get current context information"""
        try:
            response = requests.get(self.context_url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                # Map the new API fields to expected field names
                if 'kv_cache_used' in data:
                    data['used_context'] = data['kv_cache_used']
                if 'kv_cache_size' in data:
                    data['total_context'] = data['kv_cache_size']
                return data
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
        n_kv_max = info.get("n_kv_max", total)
        percent = (used / total * 100) if total > 0 else 0

        # Show n_kv_max if it's different from n_ctx
        if n_kv_max != total:
            status_line = f"{self.GRAY}[Context: {chunks} chunks, {used}/{total} tokens ({percent:.0f}%), KV max: {n_kv_max}]{self.RESET}"
        else:
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
        parts = arg.split(None, 1)
        tag = parts[0] if len(parts) > 1 else "context"
        fname = parts[1] if len(parts) > 1 else parts[0]
        """Load a file as a context chunk: /load <filename>"""
        if not arg:
            print(f"{self.RED}Usage: /load <filename>{self.RESET}")
            return

        filename = os.path.expanduser(fname.strip())
        if not os.path.exists(filename):
            print(f"{self.RED}File not found: {filename}{self.RESET}")
            return

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"Loading {filename}...")
            response = self._make_request("POST", f"{self.context_url}/load", json={
                "content": f"<{tag} filename=\"{filename}\">\n" + content + f"\n</{tag}>\n",
                "metadata": {"source": filename}
            })

            if response:
                data = response.json()
                print(f"{self.GREEN}✓ Added chunk:{self.RESET}")
                print(f"  Hash: {data['hash']}")
                print(f"  Size: {data.get('token_size', data.get('token_count', 0))} tokens")
                position = data.get('position', {})
                if position:
                    print(f"  Position: {position.get('start', 0)}-{position.get('end', 0)}")
                else:
                    # Fallback for flattened structure
                    print(f"  Position: {data.get('start_pos', 0)}-{data.get('end_pos', 0)}")

                # Update context info
                self.context_info = self._get_context_info()
                self._update_prompt()

        except Exception as e:
            print(f"{self.RED}Error loading file: {e}{self.RESET}")
        self._update_prompt()

    def do_save(self, arg: str):
        """Save a chunk to disk: /save <hash|num>"""
        if not arg:
            print(f"{self.RED}Usage: /save <hash|num>{self.RESET}")
            return

        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return

        response = self._make_request("POST", f"{self.context_url}/save/{hash_value}")

        if response:
            print(f"{self.GREEN}✓ Chunk saved: {hash_value}{self.RESET}")
        else:
            print(f"{self.YELLOW}Note: The save feature requires the server to be started with --slot-save-path <directory>{self.RESET}")
            print(f"{self.YELLOW}Example: ./llama-server --context-manager --slot-save-path ./saved_contexts{self.RESET}")
        self._update_prompt()

    def do_restore(self, arg: str):
        """Restore a saved chunk: /restore <hash|num>"""
        if not arg:
            print(f"{self.RED}Usage: /restore <hash|num>{self.RESET}")
            return

        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return

        response = self._make_request("POST", f"{self.context_url}/restore/{hash_value}")

        if response:
            data = response.json()
            print(f"{self.GREEN}✓ Chunk restored: {hash_value}{self.RESET}")
            print(f"  Size: {data.get('token_size', data.get('token_count', 0))} tokens")

            # Show restore method if available
            restore_method = data.get('restore_method', 'unknown')
            if restore_method == 'fast_system_cache':
                print(f"  Method: Fast restore from system RAM")
            elif restore_method == 'fast_disk_cache':
                print(f"  Method: Fast restore from disk KV cache")
            elif restore_method == 'slow_inference':
                print(f"  Method: Regenerated via inference")
            else:
                print(f"  Method: Direct KV cache restore")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.YELLOW}Note: The restore feature requires the server to be started with --slot-save-path <directory>{self.RESET}")
            print(f"{self.YELLOW}Example: ./llama-server --context-manager --slot-save-path ./saved_contexts{self.RESET}")
        self._update_prompt()

    def do_erase(self, arg: str):
        """Erase a chunk from context: /erase <hash|num>"""
        if not arg:
            print(f"{self.RED}Usage: /erase <hash|num>{self.RESET}")
            return

        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return

        response = self._make_request("POST", f"{self.context_url}/erase/{hash_value}")

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
        self._update_prompt()

    def do_context(self, arg: str):
        """Display current context chunks in tabular format"""
        self.context_info = self._get_context_info()
        info = self.context_info

        print(f"\n{self.BOLD}Context Status:{self.RESET}")
        print(f"  Total: {info.get('total_context', info.get('kv_cache_size', 0))} tokens (n_ctx)")
        n_kv_max = info.get('n_kv_max', info.get('kv_cache_size', 0))
        if n_kv_max != info.get('total_context', info.get('kv_cache_size', 0)):
            print(f"  KV Max: {n_kv_max} tokens (max capacity)")
        print(f"  Used: {info.get('used_context', info.get('kv_cache_used', 0))} tokens")
        print(f"  Active chunks: {info.get('active_chunks', 0)}")
        print(f"  Inactive chunks: {info.get('inactive_chunks', 0)}")
        if info.get('system_chunks', 0) > 0:
            print(f"  System chunks: {info.get('system_chunks', 0)}")
        # Calculate free tokens
        total = info.get('total_context', info.get('kv_cache_size', 0))
        used = info.get('used_context', info.get('kv_cache_used', 0))
        free = total - used if total > 0 else 0
        print(f"  Free: {free} tokens")
        
        # Show fragmentation if available
        if 'fragmentation' in info:
            print(f"  Fragmentation: {info.get('fragmentation', 0):.2%}")

        # Display gaps if present
        gaps = info.get('gaps', [])
        if gaps:
            print(f"\n  {self.BOLD}Memory Gaps:{self.RESET}")
            for gap in gaps:
                print(f"    Position {gap['start_pos']}: {gap['size_positions']} tokens")

        chunks = info.get('chunks', [])
        if chunks:
            # Sort all chunks by status priority and position for consistent indexing
            # This matches the sorting in _resolve_chunk_identifier
            all_chunks = sorted(chunks, key=lambda c: (
                0 if c.get('status') == 'active' else
                1 if c.get('status') == 'inactive' else
                2 if c.get('status') == 'system' else
                3,  # disk
                c.get('position', {}).get('start', 0)
            ))

            # Get terminal width for dynamic table sizing
            try:
                import shutil
                terminal_width = shutil.get_terminal_size().columns
            except:
                terminal_width = 80  # fallback

            # Calculate column widths
            num_col = 3
            hash_col = 12
            size_col = 6
            mem_col = 8  # MB column
            pos_col = 12
            status_col = 8
            # Preview gets remaining space minus separators
            separators = 6 * 2  # 6 two-space separators
            preview_col = max(20, terminal_width - (num_col + hash_col + size_col + mem_col + pos_col + status_col + separators))

            print(f"\n{self.BOLD}All Chunks:{self.RESET}")
            print("-" * terminal_width)
            print(f"{'#':>{num_col}}  {'Hash':<{hash_col}}  {'Size':>{size_col}}  {'MB':>{mem_col}}  {'Position':>{pos_col}}  {'Status':<{status_col}}  {'Preview':<{preview_col}}")
            print("-" * terminal_width)

            for idx, chunk in enumerate(all_chunks, 1):
                hash_val = chunk.get('hash', 'unknown')
                # Abbreviate hash to first 16 characters + ellipsis
                hash_abbrev = hash_val[:8] + " ..." if len(hash_val) > 8 else hash_val
                size = chunk.get('token_size', chunk.get('size', 0))

                # Calculate memory size in MB
                memory_bytes = chunk.get('memory_size', 0)
                memory_mb = memory_bytes / (1024 * 1024) if memory_bytes > 0 else 0

                # Handle position for chunks in VRAM vs offloaded chunks
                # Check for KV cache positions first (new API format)
                kv_start = chunk.get('kv_start_pos')
                kv_end = chunk.get('kv_end_pos')
                if kv_start is not None and kv_end is not None:
                    position = f"{kv_start}-{kv_end}"
                else:
                    # Fall back to position object
                    position_info = chunk.get('position', {})
                    if position_info and position_info.get('start') is not None:
                        pos_start = position_info.get('start', 0)
                        pos_end = position_info.get('end', 0)
                        position = f"{pos_start}-{pos_end}"
                    else:
                        position = "n/a"

                status = chunk.get('status', 'unknown')
                preview = self._get_chunk_preview(chunk, preview_col)

                # Color code status
                if status == 'active':
                    status_display = f"{self.GREEN}{status:<{status_col}}{self.RESET}"
                elif status == 'inactive':
                    status_display = f"{self.YELLOW}{status:<{status_col}}{self.RESET}"
                elif status == 'system':
                    status_display = f"{self.BLUE}{status:<{status_col}}{self.RESET}"
                elif status == 'disk':
                    status_display = f"{self.GRAY}{status:<{status_col}}{self.RESET}"
                else:
                    status_display = f"{status:<{status_col}}"

                print(f"{idx:>{num_col}}  {hash_abbrev:<{hash_col}}  {size:>{size_col}}  {memory_mb:>{mem_col}.1f}  {position:>{pos_col}}  {status_display}  {preview}")

            print("-" * terminal_width)

            # Summary by status
            status_counts = {}
            for chunk in chunks:
                status = chunk.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1

            summary_parts = []
            for status, count in sorted(status_counts.items()):
                if status == 'active':
                    summary_parts.append(f"{self.GREEN}{count} active{self.RESET}")
                elif status == 'inactive':
                    summary_parts.append(f"{self.YELLOW}{count} inactive{self.RESET}")
                elif status == 'system':
                    summary_parts.append(f"{self.BLUE}{count} in RAM{self.RESET}")
                elif status == 'disk':
                    summary_parts.append(f"{self.GRAY}{count} on disk{self.RESET}")
                else:
                    summary_parts.append(f"{count} {status}")

            print(f"Total: {len(chunks)} chunks ({', '.join(summary_parts)})")
            print(f"\n{self.GRAY}Use /show <number>, /activate <number>, /deactivate <number>, etc. with the # column{self.RESET}")

        else:
            print(f"\n{self.GRAY}No chunks loaded{self.RESET}")

        print()
        self._update_prompt()

    def _get_chunk_preview(self, chunk, max_width=None):
        """Helper to get chunk preview text"""
        metadata = chunk.get('metadata', {})
        
        # Check if this is a user input chunk
        if metadata.get('source') == 'user_input':
            return "[USER INPUT]"
        
        # Check if this is an inference output chunk
        elif metadata.get('type') == 'inference_output':
            n_generated = metadata.get('n_tokens_generated', 0)
            n_prompt = metadata.get('n_prompt_tokens', 0)
            return f"[OUTPUT: {n_generated} tokens from {n_prompt} prompt]"
        
        # Check for source field directly in chunk (flattened structure)
        elif 'source' in chunk:
            # Show full filename if loaded from a file
            filename = os.path.basename(chunk['source'])
            if max_width and len(filename) + 2 > max_width:  # +2 for brackets
                # Truncate if too long for column
                available = max_width - 5  # Reserve space for [...]
                return f"[{filename[:available]}...]"
            else:
                return f"[{filename}]"
        elif 'content' in chunk:
            # If content is available in chunk data
            content = chunk['content']
            if content:
                # Use available width for content preview
                available_width = max_width - 1 if max_width else 20  # -1 for safety
                preview = content[:available_width].replace('\n', '\\n')
                return preview[:available_width]
            else:
                return '...'
        else:
            return '...'

    def do_clear(self, arg: str):
        """Clear all chunks from context"""
        response = self._make_request("POST", f"{self.context_url}/clear")

        if response:
            result = response.json()
            cleared_count = result.get("cleared_chunks", 0)
            print(f"{self.GREEN}✓ Context cleared: removed {cleared_count} chunks{self.RESET}")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()
        else:
            print(f"{self.RED}Failed to clear context{self.RESET}")

    def do_activate(self, arg: str):
        """Activate an inactive chunk: /activate <hash_or_number>"""
        if not arg:
            print(f"{self.RED}Usage: /activate <hash_or_number>{self.RESET}")
            return

        hash_value = self._resolve_chunk_identifier(arg.strip())
        if not hash_value:
            return

        response = self._make_request("POST", f"{self.context_url}/activate/{hash_value}")

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

        response = self._make_request("POST", f"{self.context_url}/deactivate/{hash_value}")

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

        response = self._make_request("POST", f"{self.context_url}/unload/{hash_value}")

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

        # Calculate the target token position based on where we want to insert
        if target_position == 0:
            # Moving to first position
            target_token_position = 0
        elif target_position < current_position:
            # Moving backward - place at the start of the target chunk
            target_chunk = sorted_chunks[target_position]
            target_token_position = target_chunk.get('position', {}).get('start', 0)
        else:
            # Moving forward - place after the chunk that will be at target_position-1 after removal
            if target_position == 1:
                # Special case: moving to second position
                first_chunk = sorted_chunks[0]
                target_token_position = first_chunk.get('position', {}).get('end', 0)
            else:
                # General case: place after the previous chunk
                prev_chunk = sorted_chunks[target_position - 1]
                target_token_position = prev_chunk.get('position', {}).get('end', 0)

        # Try the optimized direct move operation first
        print(f"Moving chunk to position {target_position + 1}...")

        # First, try the rearrange operation with the new order
        # Build the new order of chunk hashes
        new_order_hashes = []
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

        # Extract just the hashes
        for chunk in new_order:
            new_order_hashes.append(chunk['hash'])

        # Try the optimized rearrange operation
        rearrange_success = False
        response = self._make_request("POST", f"{self.context_url}/batch", json={
            "operations": [{
                "action": "rearrange",
                "order": new_order_hashes
            }],
            "compact_after": True
        })

        if response:
            result = response.json()[0]
            if result.get("success", False):
                rearrange_success = True
                details = result.get("details", {})
                method = details.get("method", "unknown")
                moves = details.get("moves_performed", 0)
                print(f"{self.GREEN}✓ Rearranged using {method} method with {moves} moves{self.RESET}")

        # If rearrange failed, try direct move
        direct_move_success = False
        if not rearrange_success:
            response = self._make_request("POST", f"{self.context_url}/batch", json={
                "operations": [{
                    "action": "move",
                    "hash": hash_value,
                    "target_position": target_token_position
                }],
                "compact_after": True
            })

            if response and response.json()[0]["success"]:
                direct_move_success = True
                print(f"{self.GREEN}✓ Direct move successful{self.RESET}")

        # Fall back to erase/restore method if both rearrange and direct move failed
        if not rearrange_success and not direct_move_success:
            print(f"{self.YELLOW}Optimized methods failed, using erase/restore method...{self.RESET}")

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
            # Request compaction after batch operations to minimize fragmentation
            response = self._make_request("POST", f"{self.context_url}/batch", json={
                "operations": operations,
                "compact_after": True
            })

        # Check if any method succeeded
        if rearrange_success or direct_move_success:
            print(f"{self.GREEN}✓ Chunk moved successfully{self.RESET}")
            print(f"  Moved chunk from position {current_position + 1} to position {target_position + 1}")

            # Update context info
            self.context_info = self._get_context_info()
            self._update_prompt()

            # Show new order
            self.do_context("")
        elif response:
            # Erase/restore method
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

            # Use the shared streaming method for completion format
            def stop_check(data):
                return data.get('stop', False)

            self._stream_response(response, "Output", "content", stop_check)

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

        # Get chunk details and content using the new GET /context/show/{hash} endpoint
        response = self._make_request("GET", f"{self.context_url}/show/{hash_value}")

        if response:
            data = response.json()
            metadata = data.get('metadata', {})
            
            print(f"\n{self.BOLD}Chunk Details:{self.RESET}")
            print(f"  Hash: {data.get('hash', hash_value)}")
            
            # Identify the chunk type
            if metadata.get('source') == 'user_input':
                print(f"  Type: {self.BLUE}USER INPUT{self.RESET}")
                print(f"  Timestamp: {metadata.get('timestamp', 'unknown')}")
            elif metadata.get('type') == 'inference_output':
                print(f"  Type: {self.GREEN}INFERENCE OUTPUT{self.RESET}")
                print(f"  Generated tokens: {metadata.get('n_tokens_generated', 0)}")
                print(f"  Prompt tokens: {metadata.get('n_prompt_tokens', 0)}")
                print(f"  Generation time: {metadata.get('generation_time', 0):.1f}ms")
            else:
                print(f"  Type: {self.YELLOW}FILE/CONTEXT{self.RESET}")
                if 'source' in data:
                    print(f"  Source: {data['source']}")
            
            print(f"  Size: {data.get('content_length', 0)} bytes, {data.get('token_size', data.get('token_count', 0))} tokens")
            
            # Show KV cache position for chunks in memory
            kv_start = data.get('kv_start_pos')
            kv_end = data.get('kv_end_pos')
            if kv_start is not None and kv_end is not None:
                print(f"  KV Cache Position: {kv_start}-{kv_end}")
            else:
                position = data.get('position', {})
                if position:
                    print(f"  Position: {position.get('start', 0)}-{position.get('end', 0)}")
                else:
                    # Fallback for flattened structure
                    start_pos = data.get('start_pos')
                    end_pos = data.get('end_pos')
                    if start_pos is not None and end_pos is not None:
                        print(f"  Position: {start_pos}-{end_pos}")
            
            print(f"  Memory size: {data.get('memory_size', 0)} bytes")
            print(f"  Status: {data.get('status', 'unknown')}")

            # Show the content
            content = data.get('content', '')
            print(f"\n{self.BOLD}Content:{self.RESET}")
            print("-" * self.terminal_width)
            if content:
                print(content)
            else:
                print(f"{self.GRAY}(no content available){self.RESET}")
            print("-" * self.terminal_width)
            print()
        else:
            print(f"{self.RED}Chunk not found: {hash_value}{self.RESET}")

    def _stream_response(self, response, header_text: str, content_key: str = 'content', done_check=None):
        """Generic method to stream and render responses with markdown"""
        # Print header
        self.console.print(f"[bold blue]{header_text}:[/bold blue]", end=" ")

        # Accumulate response content
        full_response = ""

        # Use Rich Live for real-time markdown rendering
        with Live(console=self.console, refresh_per_second=1, auto_refresh=False) as live:
            self.thinking = False
            self.thought_buffer = ''
            for line in response.iter_lines():
                if not line:
                    continue
                line = line.decode('utf-8')
                if not line.startswith('data: '):
                    continue
                data_str = line[6:]
                if data_str == '[DONE]':
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if content_key == 'content':
                    content = data.get('content', '')
                else:
                    choice = data.get('choices', [{}])[0]
                    delta = choice.get('delta', {})
                    content = delta.get('content', '')

                if not content:
                    if done_check and done_check(data):
                        break
                    continue

                remaining = content
                while remaining:
                    if not self.thinking:
                        idx = remaining.find('<think>')
                        if idx == -1:
                            full_response += remaining
                            remaining = ''
                            try:
                                live.update(Markdown(full_response))
                            except Exception:
                                live.update(full_response)
                            live.refresh()
                        else:
                            pre = remaining[:idx]
                            full_response += pre
                            remaining = remaining[idx + len('<think>'):]
                            if pre:
                                try:
                                    live.update(Markdown(full_response))
                                except Exception:
                                    live.update(full_response)
                                live.refresh()
                            self.thinking = True
                            self.thought_buffer = ''
                            live.update(
                                Group(
                                    Markdown(full_response),
                                    Text(self.thought_buffer, style='yellow')
                                )
                            )
                            live.refresh()
                    else:
                        idx = remaining.find('</think>')
                        if idx == -1:
                            self.thought_buffer += remaining
                            remaining = ''
                            live.update(
                                Group(
                                    Markdown(full_response),
                                    Text(self.thought_buffer, style='yellow')
                                )
                            )
                            live.refresh()
                        else:
                            part = remaining[:idx]
                            self.thought_buffer += part
                            remaining = remaining[idx + len('</think>'):]
                            live.update(Markdown(full_response))
                            live.refresh()
                            self.thinking = False
                            self.thought_buffer = ''
                    if done_check and done_check(data):
                        remaining = ''
                        break

        # Process XML tags in the final response after thinking is complete
        full_response = self._process_xml_tags(full_response)

        # Final update with processed content
        try:
            live.update(Markdown(full_response))
        except Exception:
            live.update(full_response)
        live.refresh()

        print()

    def _process_xml_tags(self, content: str) -> str:
        """Process XML tags in the response content"""
        import re

        # Handle <response/> tags with or without attributes
        response_pattern = r'<response(?:\s+([^>]*?))?>(.*?)</response>'
        def replace_response(match):
            attributes = match.group(1) or ''
            inner_content = match.group(2)

            # Strip leading/trailing whitespace and newlines from content
            inner_content = inner_content.strip()

            # Parse attributes to get language and filename
            language = ''
            filename = ''

            if attributes:
                # Simple attribute parsing
                lang_match = re.search(r'language\s*=\s*["\']([^"\']*)["\']', attributes)
                if lang_match:
                    language = lang_match.group(1)

                file_match = re.search(r'filename\s*=\s*["\']([^"\']*)["\']', attributes)
                if file_match:
                    filename = file_match.group(1)

            # Create markdown code block
            code_block_header = language
            if filename:
                code_block_header += f' {filename}'

            return f'```{code_block_header}\n{inner_content}\n```'

        content = re.sub(response_pattern, replace_response, content, flags=re.DOTALL)

        # Handle <error/> tags - strip tags and add red color
        error_pattern = r'<error>(.*?)</error>'
        def replace_error(match):
            error_content = match.group(1)
            # Use ANSI red color codes
            return f'{self.RED}{error_content}{self.RESET}'

        content = re.sub(error_pattern, replace_error, content, flags=re.DOTALL)

        return content

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
                    'help': self.do_help,
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
                try:
                    # Prepare the request
                    data = {
                        "messages": [{"role": "user", "content": line}],
                        "stream": True
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

                    # Use the shared streaming method for chat completion format
                    self._stream_response(response, "Assistant", "delta_content")

                except Exception as e:
                    print(f"\n{self.RED}Error during chat: {e}{self.RESET}")

    def do_help(self, arg: str):
        """Show help for commands"""
        if not arg:
            print(f"\n{self.BOLD}Available Commands:{self.RESET}")
            print(f"\n{self.BOLD}Basic Operations:{self.RESET}")
            print("  /load <tag> <file>   - Load a file as a <tag> chunk (state: ACTIVE)")
            print("  /clear               - Clear all chunks from context")
            print("  /context             - Display all chunks with #, hash, size, MB, position, status, and preview")
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
            print("  /help                - Show this help message")
            print("  /exit/quit           - Exit the CLI")
            print(f"\n{self.BOLD}Tips:{self.RESET}")
            print("  - Use /context to see chunk numbers, then reference by number (e.g., /show 1)")
            print("  - You can also use partial hashes instead of numbers")
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
