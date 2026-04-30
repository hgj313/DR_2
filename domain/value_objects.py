from typing import NewType

StandardId = NewType("StandardId", str)
DocumentId = NewType("DocumentId", str)
ChunkId = NewType("ChunkId", str)

__all__ = [
    "StandardId",
    "DocumentId",
    "ChunkId"
]