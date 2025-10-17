"""
Vector Database File Storage - single file storage system.
Handles persistence of the vector database to a single .db file.
"""

import logging
import gzip
import json

from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from ...meta import __version__

logger = logging.getLogger(__name__)

class VectorFileStorage:
    """Single-file storage system for Vector Database."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

        self.metadata = {
            "version": __version__,
            "created_at": None,
            "last_modified": None,
            "total_vector_points": 0,
            "total_dimensions": 0
        }

    def save_database(self, database_data: Dict[str, Any]) -> bool:
        """
        Save the complete vector database to file.

        Args:
            database_data: Complete serialized database structure

        Returns:
            bool: True if save succeeded
        """

        try:
            now = datetime.now().isoformat()

            if self.metadata["created_at"] is None:
                self.metadata["created_at"] = now

            self.metadata["last_modified"] = now

            complete_data = {
                "metadata": self.metadata,
                "database": database_data
            }

            json_data = json.dumps(complete_data, indent=2, default=str)
            compressed_data = gzip.compress(json_data.encode('utf-8'))

            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.file_path, 'wb') as f:
                f.write(compressed_data)

            logger.info(f"Vector database saved to {self.file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save vector database: {e}")
            return False

    def load_database(self) -> Optional[Dict[str, Any]]:
        """
        Load the vector database from file.

        Returns:
            Optional[Dict]: Database data if load succeeded, None otherwise
        """

        try:
            if not self.file_path.exists():
                logger.info(f"Database file {self.file_path} does not exist")
                return None

            with open(self.file_path, 'rb') as f:
                compressed_data = f.read()

            json_data = gzip.decompress(compressed_data).decode('utf-8')
            complete_data = json.loads(json_data)

            self.metadata = complete_data.get("metadata", self.metadata)
            database_data = complete_data.get("database", {})

            logger.info(f"Vector database loaded from {self.file_path}")
            return database_data

        except Exception as e:
            logger.error(f"Failed to load vector database: {e}")
            return None

    def exists(self) -> bool:
        """Check if the database file exists."""

        return self.file_path.exists()

    def delete(self) -> bool:
        """Delete the database file."""

        try:
            if self.file_path.exists():
                self.file_path.unlink()
                logger.info(f"Deleted database file {self.file_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to delete database file: {e}")
            return False

    def get_file_size(self) -> int:
        """Get the size of the database file in bytes."""

        if self.file_path.exists():
            return self.file_path.stat().st_size

        return 0

    def get_metadata(self) -> Dict[str, Any]:
        """Get database metadata."""

        return self.metadata.copy()

    def update_metadata(self, updates: Dict[str, Any]):
        """Update database metadata."""

        self.metadata.update(updates)

    def __repr__(self) -> str:
        size = self.get_file_size()
        exists = "exists" if self.exists() else "not found"

        return f"VectorFileStorage(path='{self.file_path}', {exists}, {size} bytes)"
