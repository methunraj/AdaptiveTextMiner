import re
from typing import List, Dict, Any, Optional

# Optional: install tiktoken or fallback to char count
try:
    import tiktoken
    def count_tokens(text, model_name="gpt-3.5-turbo"):
        enc = tiktoken.encoding_for_model(model_name)
        return len(enc.encode(text))
except ImportError:
    def count_tokens(text, model_name=None):
        return len(text) // 4  # rough estimate: 4 chars per token

def smart_chunk_document(
    text: str,
    context_window: int = 4096,
    overlap: int = 0,
    split_mode: str = None,
    model_name: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Advanced chunking with:
    - Token-based splitting (if possible)
    - Structure-aware splitting (lines, paragraphs, auto)
    - Smart overlap (by record/paragraph)
    - Metadata for traceability
    Returns: list of dicts with chunk, chunk_id, start, end, etc.
    """
    # Decide split units
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
    current_tokens = 0
    chunk_start = 0
    chunk_id = 1
    for i, unit in enumerate(units):
        unit_text = unit if i == len(units)-1 else unit + joiner
        unit_tokens = count_tokens(unit_text, model_name)
        if current_tokens + unit_tokens > context_window and current_chunk:
            chunk_text = joiner.join(current_chunk)
            chunks.append({
                "chunk_id": chunk_id,
                "start_unit": chunk_start,
                "end_unit": i-1,
                "text": chunk_text,
                "metadata": metadata or {}
            })
            # Overlap by full units if overlap > 0
            if overlap > 0 and len(current_chunk) > 1:
                overlap_units = current_chunk[-overlap:]
                current_chunk = list(overlap_units)
                chunk_start = i - len(overlap_units)
                current_tokens = sum(count_tokens(u, model_name) for u in current_chunk)
            else:
                current_chunk = []
                chunk_start = i
                current_tokens = 0
            chunk_id += 1
        current_chunk.append(unit)
        current_tokens += unit_tokens
    if current_chunk:
        chunk_text = joiner.join(current_chunk)
        chunks.append({
            "chunk_id": chunk_id,
            "start_unit": chunk_start,
            "end_unit": len(units)-1,
            "text": chunk_text,
            "metadata": metadata or {}
        })
    return chunks

# Optionally, add semantic boundary detection and chunk reconciliation functions here for future expansion.
