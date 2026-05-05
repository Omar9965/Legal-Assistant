"""
Enhanced Semantic Chunking — Improved document segmentation using semantic similarity.

Features:
- Sentence-level chunking with overlap
- Semantic boundary detection
- Sliding window approach
- Context preservation
- Multi-scale chunking (small, medium, large chunks)
"""

import re
from typing import List, Tuple, Optional, Dict
from langchain_core.documents import Document
from dataclasses import dataclass
from enum import Enum


class ChunkSize(Enum):
    """Chunk size categories."""
    SMALL = 500      # For specific queries
    MEDIUM = 1000    # General purpose
    LARGE = 1500     # For complex topics


@dataclass
class ChunkConfig:
    """Configuration for semantic chunking."""
    chunk_size: int
    overlap: int
    min_chunk_size: int = 100
    preserve_sentences: bool = True


class SemanticChunker:
    """
    Semantic chunker that identifies natural breakpoints in legal documents.
    """
    
    def __init__(self, config: ChunkConfig = None):
        self.config = config or ChunkConfig(
            chunk_size=1000,
            overlap=200,
            min_chunk_size=100,
            preserve_sentences=True
        )
        
        # Arabic legal document boundaries
        self._article_pattern = re.compile(
            r"(?:مادة|materia|المادة)\s*[\(（]?\s*(?:\d+|[٠-٩]+)\s*[\)）]?",
            re.IGNORECASE
        )
        
        # Section patterns
        self._section_patterns = [
            re.compile(r"(?:الفصل|الباب| الكتاب |القسم)\s+[^\n]+", re.IGNORECASE),
            re.compile(r"^[أ-ؤء-ي]+\s+:", re.MULTILINE),  # Lists
            re.compile(r"(?:أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً)", re.IGNORECASE),
            re.compile(r"(?:الفرع|الفصل|الباب)\s+", re.IGNORECASE),
        ]
        
        # Sentence ending patterns
        self._sentence_endings = set([".", "،", "؛", "؟", "!", "۔", "📜"])
        
        # Context markers
        self._context_markers = [
            "حيث إن", "بما أن", "لما كان", "بعد الاطلاع",
            "و حيث", "و بناءً", "pursuant to", "given that"
        ]
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict = None,
        chunk_size: ChunkSize = ChunkSize.MEDIUM
    ) -> List[Document]:
        """
        Split text into semantically coherent chunks.
        
        Args:
            text: Input text to chunk.
            metadata: Metadata to attach to each chunk.
            chunk_size: Target chunk size category.
        
        Returns:
            List of Document objects.
        """
        
        # Adjust chunk size based on enum
        size_map = {
            ChunkSize.SMALL: 500,
            ChunkSize.MEDIUM: 1000,
            ChunkSize.LARGE: 1500
        }
        
        target_size = size_map.get(chunk_size, self.config.chunk_size)
        
        # Step 1: Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        # Step 2: Identify semantic boundaries
        boundaries = self._identify_boundaries(text)
        
        # Step 3: Create chunks respecting boundaries
        chunks = self._create_chunks(
            text, 
            paragraphs, 
            boundaries, 
            target_size
        )
        
        # Step 4: Add metadata and create documents
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = (metadata or {}).copy()
            doc_metadata["chunk_index"] = i
            doc_metadata["chunk_size"] = len(chunk)
            doc_metadata["chunk_type"] = "semantic"
            
            # Detect if chunk starts/ends with article
            if self._article_pattern.search(chunk):
                doc_metadata["contains_article"] = True
                article_match = self._article_pattern.search(chunk)
                if article_match:
                    article_num = self._extract_article_number(article_match.group())
                    if article_num:
                        doc_metadata["article_number"] = article_num
            
            documents.append(Document(
                page_content=chunk.strip(),
                metadata=doc_metadata
            ))
        
        return documents
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split on double newlines or single newlines with indentation
        paragraphs = re.split(r'\n\s*\n|\n(?=[^\S\n])', text)
        
        # Clean and filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _identify_boundaries(self, text: str) -> List[int]:
        """
        Identify semantic boundaries in the text.
        Returns list of character positions where boundaries occur.
        """
        boundaries = []
        
        # Add start of text
        boundaries.append(0)
        
        # Find article boundaries
        for match in self._article_pattern.finditer(text):
            boundaries.append(match.start())
        
        # Find section boundaries
        for pattern in self._section_patterns:
            for match in pattern.finditer(text):
                boundaries.append(match.start())
        
        # Add end of text
        boundaries.append(len(text))
        
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        return boundaries
    
    def _create_chunks(
        self, 
        text: str, 
        paragraphs: List[str],
        boundaries: List[int],
        target_size: int
    ) -> List[str]:
        """Create chunks respecting semantic boundaries."""
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed target size
            if len(current_chunk) + len(paragraph) > target_size:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Check if paragraph itself exceeds target size
                if len(paragraph) > target_size:
                    # Split large paragraph
                    sub_chunks = self._split_paragraph(paragraph, target_size)
                    if chunks and current_chunk:
                        # Add overlap with previous chunk
                        chunks[-1] = chunks[-1] + " " + sub_chunks[0]
                        current_chunk = sub_chunks[-1] if len(sub_chunks) > 1 else sub_chunks[0]
                    else:
                        current_chunk = sub_chunks[-1] if len(sub_chunks) > 1 else sub_chunks[0]
                else:
                    current_chunk = paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_paragraph(self, paragraph: str, max_size: int) -> List[str]:
        """Split a large paragraph into smaller chunks at sentence boundaries."""
        
        # Split into sentences
        sentences = self._split_into_sentences(paragraph)
        
        if not sentences:
            # Fallback to character-based split
            return self._fallback_split(paragraph, max_size)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = []
        current = ""
        
        for char in text:
            current += char
            
            # Check if we've reached a sentence ending
            if char in self._sentence_endings:
                # Look ahead to see if next char starts a new sentence
                sentences.append(current.strip())
                current = ""
        
        # Add remaining text
        if current.strip():
            sentences.append(current.strip())
        
        return [s for s in sentences if len(s) > 10]  # Filter very short segments
    
    def _fallback_split(self, text: str, max_size: int) -> List[str]:
        """Fallback split when sentence detection fails."""
        words = text.split()
        chunks = []
        current = ""
        
        for word in words:
            if len(current) + len(word) + 1 > max_size:
                if current:
                    chunks.append(current.strip())
                current = word
            else:
                if current:
                    current += " " + word
                else:
                    current = word
        
        if current:
            chunks.append(current.strip())
        
        return chunks
    
    def _extract_article_number(self, text: str) -> Optional[str]:
        """Extract article number from text."""
        patterns = [
            r"(\d+)",
            r"[٠-٩]+"
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                # Convert Arabic numbers to western if needed
                arabic_nums = "٠١٢٣٤٥٦٧٨٩"
                result = match.group(1)
                if any(c in arabic_nums for c in result):
                    # Convert Arabic to Western
                    arabic_to_western = str.maketrans(arabic_nums, "0123456789")
                    result = result.translate(arabic_to_western)
                return result
        return None


class MultiScaleChunker:
    """
    Creates multiple chunk sizes for different query types.
    """
    
    def __init__(self):
        self.small_chunker = SemanticChunker(ChunkConfig(
            chunk_size=500,
            overlap=100
        ))
        self.medium_chunker = SemanticChunker(ChunkConfig(
            chunk_size=1000,
            overlap=200
        ))
        self.large_chunker = SemanticChunker(ChunkConfig(
            chunk_size=1500,
            overlap=300
        ))
    
    def chunk_all_scales(
        self, 
        text: str, 
        metadata: Dict = None
    ) -> Dict[str, List[Document]]:
        """
        Create chunks at all size scales.
        
        Returns:
            Dictionary with keys 'small', 'medium', 'large' 
            each containing list of Document objects.
        """
        
        return {
            "small": self.small_chunker.chunk_text(
                text, metadata, ChunkSize.SMALL
            ),
            "medium": self.medium_chunker.chunk_text(
                text, metadata, ChunkSize.MEDIUM
            ),
            "large": self.large_chunker.chunk_text(
                text, metadata, ChunkSize.LARGE
            )
        }


def semantic_chunk_text(
    text: str, 
    source: str = "law.pdf",
    chunk_size: int = 1000,
    overlap: int = 200
) -> List[Document]:
    """
    Convenience function for semantic chunking.
    
    Args:
        text: Text to chunk.
        source: Source document name.
        chunk_size: Target chunk size in characters.
        overlap: Overlap between chunks in characters.
    
    Returns:
        List of Document objects.
    """
    
    config = ChunkConfig(
        chunk_size=chunk_size,
        overlap=overlap
    )
    
    chunker = SemanticChunker(config)
    
    return chunker.chunk_text(
        text, 
        metadata={"source": source},
        chunk_size=ChunkSize.MEDIUM
    )