"""
Context Management System for handling document chunking and context preservation.
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np

class ContextManager:
    """
    Manages document chunking and context preservation across document chunks.
    Implements intelligent chunking strategies and maintains context continuity
    for large documents that exceed LLM context limits.
    """
    def __init__(self):
        pass

    def chunk_document(self, text: str, chunk_size: int = 15000, overlap: int = 1000, context_window: int = None, split_mode: str = None, advanced: bool = True, model_name: str = None, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        Split a document into context-window-aware, structure-preserving chunks.
        If advanced=True, use token-based and metadata-aware chunking.
        Args:
            text: Document text to split
            chunk_size: Maximum characters per chunk (legacy)
            overlap: Overlap size between chunks (in units)
            context_window: (Optional) Max chunk size (tokens)
            split_mode: (Optional) 'line', 'paragraph', or None for auto
            advanced: Use advanced_chunking (token-based, metadata, traceability)
            model_name: (Optional) LLM model name for token counting
            metadata: (Optional) Metadata to attach to each chunk
        Returns:
            List of document chunk texts (advanced: can return list of dicts)
        """
        if advanced:
            try:
                from src.advanced_chunking import smart_chunk_document
                chunks = smart_chunk_document(
                    text,
                    context_window=context_window if context_window else chunk_size,
                    overlap=overlap,
                    split_mode=split_mode,
                    model_name=model_name,
                    metadata=metadata
                )
                # Return just the text for compatibility, but you can return full dicts if needed
                return [c['text'] for c in chunks]
            except Exception as e:
                print(f"[WARN] Falling back to legacy chunking due to error: {e}")
        # Legacy fallback
        # ... (existing legacy logic unchanged)
        max_chunk = context_window if context_window else chunk_size
        if split_mode == 'line' or (split_mode is None and '\t' in text):
            units = text.splitlines()
            joiner = '\n'
        elif split_mode == 'paragraph' or (split_mode is None and '\n\n' in text):
            units = text.split('\n\n')
            joiner = '\n\n'
        else:
            units = text.splitlines()
            joiner = '\n'
        chunks = []
        current_chunk = []
        current_len = 0
        for i, unit in enumerate(units):
            unit_len = len(unit)
            if current_len + unit_len > max_chunk and current_chunk:
                if overlap > 0 and len(current_chunk) > 1:
                    overlap_units = current_chunk[-(overlap//unit_len):] if unit_len else []
                    chunks.append(joiner.join(current_chunk))
                    current_chunk = list(overlap_units)
                    current_len = sum(len(u) for u in current_chunk)
                else:
                    chunks.append(joiner.join(current_chunk))
                    current_chunk = []
                    current_len = 0
            current_chunk.append(unit)
            current_len += unit_len
        if current_chunk:
            chunks.append(joiner.join(current_chunk))
        if len(chunks) <= 1:
            if len(text) <= 1000:
                return [text]
            if len(text) <= chunk_size and len(text) > 1000:
                mid_point = len(text) // 2
                search_area = text[mid_point - min(500, mid_point):mid_point + min(500, len(text) - mid_point)]
                paragraph_break = search_area.find('\n\n')
                if paragraph_break != -1:
                    split_point = mid_point - 500 + paragraph_break + 2
                    return [text[:split_point], text[max(0, split_point - overlap):]]
                else:
                    newline_break = search_area.find('\n')
                    if newline_break != -1:
                        split_point = mid_point - 500 + newline_break + 1
                        return [text[:split_point], text[max(0, split_point - overlap):]]
        return chunks
    
    def add_context_metadata(self, chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        """
        Add metadata context to each chunk to help with context preservation.
        
        Args:
            chunks: List of document chunks
            metadata: Document metadata
            
        Returns:
            Chunks with added context information
        """
        context_header = "Document Metadata:\n"
        
        for key, value in metadata.items():
            context_header += f"{key}: {value}\n"
        
        context_header += "\n---\n\n"
        
        # Add context header to each chunk
        return [context_header + chunk for chunk in chunks]
    
    def add_chunk_position_metadata(self, chunks: List[str]) -> List[str]:
        """
        Add positional metadata to each chunk.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Chunks with added positional information
        """
        total_chunks = len(chunks)
        contextualized_chunks = []
        
        for i, chunk in enumerate(chunks):
            position_info = f"[This is part {i+1} of {total_chunks} of the document]\n\n"
            contextualized_chunks.append(position_info + chunk)
        
        return contextualized_chunks
    
    def find_semantic_boundaries(self, text: str) -> List[int]:
        """
        Find semantic boundaries in text for better chunking.
        
        Args:
            text: Document text
            
        Returns:
            List of character positions for semantic boundaries
        """
        boundaries = []
        
        # Look for section headers (common patterns)
        section_patterns = [
            r'\n#+\s+.+\n',  # Markdown headers
            r'\n[A-Z][A-Za-z\s]+:\n',  # Title followed by colon
            r'\n[IVX]+\.\s+[A-Z]',  # Roman numeral sections
            r'\n\d+\.\d+\s+[A-Z]',  # Numbered sections
            r'\n[A-Z][A-Z\s]+\n',  # ALL CAPS HEADERS
        ]
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text):
                boundaries.append(match.start())
        
        # Add paragraph breaks
        for match in re.finditer(r'\n\s*\n', text):
            boundaries.append(match.start())
        
        # Sort and deduplicate
        boundaries = sorted(set(boundaries))
        
        return boundaries
    
    def chunk_by_semantic_boundaries(self, text: str, max_chunk_size: int = 15000) -> List[str]:
        """
        Chunk document by semantic boundaries.
        
        Args:
            text: Document text
            max_chunk_size: Maximum chunk size
            
        Returns:
            List of document chunks
        """
        boundaries = self.find_semantic_boundaries(text)
        
        # If no boundaries found or text is small, return as single chunk
        if not boundaries or len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        for i, boundary in enumerate(boundaries):
            # If adding this section would exceed max size, create a chunk
            if boundary - start > max_chunk_size:
                # Find a suitable boundary within max_chunk_size
                suitable_boundaries = [b for b in boundaries if start < b < start + max_chunk_size]
                
                if suitable_boundaries:
                    end = suitable_boundaries[-1]  # Take the last boundary within range
                else:
                    # If no suitable boundary, just cut at max_chunk_size
                    end = start + max_chunk_size
                    
                    # Try to find a sentence end within the last 100 chars
                    last_part = text[max(end - 100, start):end]
                    last_sentence = last_part.rfind('. ')
                    if last_sentence != -1:
                        end = max(end - 100, start) + last_sentence + 2
                
                chunks.append(text[start:end])
                start = end
        
        # Add final chunk
        if start < len(text):
            chunks.append(text[start:])
        
        return chunks
    
    def maintain_table_integrity(self, chunks: List[str]) -> List[str]:
        """
        Adjust chunks to maintain table integrity where possible.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Adjusted chunks with improved table handling
        """
        # Simple table detection patterns
        table_start_patterns = [
            r'[\+\-]{3,}[\+\-]+',  # ASCII table borders
            r'^\s*\|.*\|.*\|',      # Markdown tables
            r'^\s*[A-Za-z0-9]+[^\|]+\|', # Simple pipe tables
            r'^\s*[A-Za-z0-9]+(\t[A-Za-z0-9]+)+' # Tab-separated tables
        ]
        
        adjusted_chunks = []
        i = 0
        
        while i < len(chunks):
            chunk = chunks[i]
            is_adjusted = False
            
            # Check if this chunk ends in the middle of a table
            for pattern in table_start_patterns:
                # If chunk contains table start but not complete table
                if re.search(pattern, chunk, re.MULTILINE):
                    # Find all table row matches
                    table_rows = re.findall(pattern, chunk, re.MULTILINE)
                    
                    # If table starts but doesn't have many rows, and there's a next chunk
                    if len(table_rows) < 3 and i < len(chunks) - 1:
                        # Combine with next chunk
                        adjusted_chunks.append(chunk + chunks[i+1])
                        i += 2
                        is_adjusted = True
                        break
            
            if not is_adjusted:
                adjusted_chunks.append(chunk)
                i += 1
        
        return adjusted_chunks 