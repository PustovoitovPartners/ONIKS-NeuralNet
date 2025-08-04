"""Checkpoint management system for graph execution state persistence.

This module provides checkpoint functionality to save and restore graph execution states
during long-running processes, enabling recovery from failures and resumption of
computations from specific points.
"""

import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from oniks.core.state import State


class CheckpointSaver(ABC):
    """Abstract base class for checkpoint saving and loading functionality.
    
    This class defines the interface for checkpoint management systems that can
    persist and restore graph execution states. Concrete implementations should
    handle the specific storage mechanisms (database, file system, etc.).
    """
    
    @abstractmethod
    def save(self, thread_id: str, state: State) -> None:
        """Save the current state for a specific thread/task.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            state: The current State object to be saved.
            
        Raises:
            NotImplementedError: If not implemented by concrete subclass.
        """
        pass
    
    @abstractmethod
    def load(self, thread_id: str) -> Optional[State]:
        """Load the last saved state for a specific thread/task.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            
        Returns:
            The last saved State object for the thread, or None if no checkpoint exists.
            
        Raises:
            NotImplementedError: If not implemented by concrete subclass.
        """
        pass


class SQLiteCheckpointSaver(CheckpointSaver):
    """SQLite-based implementation of checkpoint saving functionality.
    
    This class provides persistent checkpoint storage using SQLite database,
    allowing for reliable state preservation across application restarts and
    system failures.
    
    Attributes:
        db_path: Path to the SQLite database file.
    
    Example:
        >>> checkpointer = SQLiteCheckpointSaver("checkpoints.db")
        >>> state = State()
        >>> state.data["progress"] = 50
        >>> checkpointer.save("task_123", state)
        >>> loaded_state = checkpointer.load("task_123")
        >>> print(loaded_state.data["progress"])
        50
    """
    
    def __init__(self, db_path: str) -> None:
        """Initialize SQLite checkpoint saver with database path.
        
        Args:
            db_path: Path to the SQLite database file. Will be created if it doesn't exist.
            
        Raises:
            sqlite3.Error: If database initialization fails.
        """
        self.db_path = Path(db_path)
        self._initialize_database()
    
    def _initialize_database(self) -> None:
        """Initialize the SQLite database and create the checkpoints table if needed.
        
        Creates the checkpoints table with the following schema:
        - thread_id: TEXT PRIMARY KEY - unique identifier for the thread
        - state_data: TEXT - JSON serialized state data
        - created_at: TIMESTAMP - when the checkpoint was created
        - updated_at: TIMESTAMP - when the checkpoint was last updated
        
        Raises:
            sqlite3.Error: If database initialization fails.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS checkpoints (
                        thread_id TEXT PRIMARY KEY,
                        state_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create trigger to update the updated_at timestamp
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS update_checkpoint_timestamp
                    AFTER UPDATE ON checkpoints
                    FOR EACH ROW
                    BEGIN
                        UPDATE checkpoints SET updated_at = CURRENT_TIMESTAMP
                        WHERE thread_id = NEW.thread_id;
                    END
                """)
                
                conn.commit()
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to initialize checkpoint database: {e}")
    
    def save(self, thread_id: str, state: State) -> None:
        """Save the current state for a specific thread to SQLite database.
        
        Serializes the State object to JSON and stores it in the database.
        If a checkpoint already exists for the thread_id, it will be updated.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            state: The current State object to be saved.
            
        Raises:
            ValueError: If thread_id is empty or None.
            sqlite3.Error: If database save operation fails.
            json.JSONEncodeError: If state serialization fails.
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty or None")
        
        try:
            # Serialize the state to JSON
            state_json = state.model_dump_json()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints (thread_id, state_data)
                    VALUES (?, ?)
                """, (thread_id, state_json))
                conn.commit()
                
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize state for thread '{thread_id}': {e}")
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to save checkpoint for thread '{thread_id}': {e}")
    
    def load(self, thread_id: str) -> Optional[State]:
        """Load the last saved state for a specific thread from SQLite database.
        
        Retrieves the JSON data from the database and deserializes it back
        into a State object.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            
        Returns:
            The last saved State object for the thread, or None if no checkpoint exists.
            
        Raises:
            ValueError: If thread_id is empty or None.
            sqlite3.Error: If database load operation fails.
            json.JSONDecodeError: If state deserialization fails.
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty or None")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT state_data FROM checkpoints
                    WHERE thread_id = ?
                """, (thread_id,))
                
                result = cursor.fetchone()
                
                if result is None:
                    return None
                
                # Deserialize the JSON back to State object
                state_data = json.loads(result[0])
                return State(**state_data)
                
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Failed to deserialize state for thread '{thread_id}': {e}", "", 0)
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to load checkpoint for thread '{thread_id}': {e}")
    
    def delete_checkpoint(self, thread_id: str) -> bool:
        """Delete a checkpoint for a specific thread.
        
        Args:
            thread_id: Unique identifier for the thread or execution context.
            
        Returns:
            True if a checkpoint was deleted, False if no checkpoint existed.
            
        Raises:
            ValueError: If thread_id is empty or None.
            sqlite3.Error: If database delete operation fails.
        """
        if not thread_id:
            raise ValueError("Thread ID cannot be empty or None")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM checkpoints WHERE thread_id = ?
                """, (thread_id,))
                conn.commit()
                
                return cursor.rowcount > 0
                
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to delete checkpoint for thread '{thread_id}': {e}")
    
    def list_checkpoints(self) -> list[str]:
        """List all thread IDs that have saved checkpoints.
        
        Returns:
            List of thread IDs that have saved checkpoints.
            
        Raises:
            sqlite3.Error: If database query fails.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT thread_id FROM checkpoints ORDER BY updated_at DESC")
                return [row[0] for row in cursor.fetchall()]
                
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to list checkpoints: {e}")
    
    def cleanup_old_checkpoints(self, keep_count: int = 100) -> int:
        """Clean up old checkpoints, keeping only the most recent ones.
        
        Args:
            keep_count: Number of most recent checkpoints to keep.
            
        Returns:
            Number of checkpoints that were deleted.
            
        Raises:
            ValueError: If keep_count is less than 1.
            sqlite3.Error: If database cleanup operation fails.
        """
        if keep_count < 1:
            raise ValueError("Keep count must be at least 1")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM checkpoints 
                    WHERE thread_id NOT IN (
                        SELECT thread_id FROM checkpoints 
                        ORDER BY updated_at DESC 
                        LIMIT ?
                    )
                """, (keep_count,))
                conn.commit()
                
                return cursor.rowcount
                
        except sqlite3.Error as e:
            raise sqlite3.Error(f"Failed to cleanup old checkpoints: {e}")