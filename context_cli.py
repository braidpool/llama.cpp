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
        print(f"Type 'help' for available commands or '/context' to see current chunks\n")
    
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
                "content": content,
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
            print(f"\n{self.BOLD}Chunks:{self.RESET}")
            print("-" * 80)
            print(f"{'Hash':<64}  {'Size':>6}  {'Position':>12}")
            print("-" * 80)
            
            for chunk in chunks:
                hash_val = chunk.get('hash', 'unknown')
                size = chunk.get('token_size', chunk.get('size', 0))
                pos_start = chunk.get('position', {}).get('start', 0)
                pos_end = chunk.get('position', {}).get('end', 0)
                position = f"{pos_start}-{pos_end}"
                
                print(f"{hash_val:<64}  {size:>6}  {position:>12}")
            
            print("-" * 80)
            print(f"Total: {len(chunks)} chunks")
        else:
            print(f"\n{self.GRAY}No chunks loaded{self.RESET}")
        
        print()
    
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
    
    def do_show(self, arg: str):
        """Show the details and content of a specific chunk: /show <hash>"""
        if not arg:
            print(f"{self.RED}Usage: /show <hash>{self.RESET}")
            return
        
        hash_value = arg.strip()
        
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
                    'compact': self.do_compact,
                    'gc': self.do_gc,
                    'context': self.do_context,
                    'clear': self.do_clear,
                    'show': self.do_show,
                }
                
                if command in command_map:
                    command_map[command](args)
                else:
                    print(f"{self.RED}Unknown command: /{command}{self.RESET}")
                    print("Available commands: /load, /save, /restore, /erase, /compact, /gc, /context, /clear, /show")
            else:
                # Send as chat message
                self._stream_chat_completion(line)
    
    def do_help(self, arg: str):
        """Show help for commands"""
        if not arg:
            print(f"\n{self.BOLD}Available Commands:{self.RESET}")
            print("  /load <file>      - Load a file as a context chunk")
            print("  /save <hash>      - Save a chunk to disk")
            print("  /restore <hash>   - Restore a saved chunk")
            print("  /erase <hash>     - Erase a chunk from context")
            print("  /clear            - Clear all chunks from context")
            print("  /show <hash>      - Show the content of a specific chunk")
            print("  /compact          - Compact memory to reduce fragmentation")
            print("  /gc               - Run garbage collection on saved chunks")
            print("  /context          - Display current context chunks")
            print("  help              - Show this help message")
            print("  exit/quit         - Exit the CLI")
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