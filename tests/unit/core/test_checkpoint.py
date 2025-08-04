"""Unit tests for the CheckpointSaver and SQLiteCheckpointSaver classes."""

import pytest
import sqlite3
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from oniks.core.checkpoint import CheckpointSaver, SQLiteCheckpointSaver
from oniks.core.state import State


class TestCheckpointSaverAbstract:
    """Test the abstract CheckpointSaver class."""
    
    def test_checkpoint_saver_is_abstract(self):
        """Test that CheckpointSaver cannot be instantiated directly."""
        with pytest.raises(TypeError):
            CheckpointSaver()
    
    def test_checkpoint_saver_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        class IncompleteCheckpointer(CheckpointSaver):
            pass
        
        with pytest.raises(TypeError):
            IncompleteCheckpointer()


class TestSQLiteCheckpointSaver:
    """Test the SQLiteCheckpointSaver class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path for testing."""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            db_path = tmp.name
        
        yield db_path
        
        # Cleanup
        path = Path(db_path)
        if path.exists():
            path.unlink()
    
    @pytest.fixture
    def checkpointer(self, temp_db_path):
        """Create a SQLiteCheckpointSaver instance with temporary database."""
        return SQLiteCheckpointSaver(temp_db_path)
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample state for testing."""
        state = State()
        state.data["test_key"] = "test_value"
        state.data["number"] = 42
        state.add_message("Test message 1")
        state.add_message("Test message 2")
        state.tool_outputs["tool1"] = "output1"
        return state
    
    def test_sqlite_checkpointer_initialization(self, temp_db_path):
        """Test SQLiteCheckpointSaver initialization."""
        checkpointer = SQLiteCheckpointSaver(temp_db_path)
        
        assert checkpointer.db_path == Path(temp_db_path)
        assert Path(temp_db_path).exists()
        
        # Verify database structure
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='checkpoints'
            """)
            assert cursor.fetchone() is not None
    
    def test_sqlite_checkpointer_database_schema(self, checkpointer, temp_db_path):
        """Test that the database schema is created correctly."""
        with sqlite3.connect(temp_db_path) as conn:
            # Check table exists
            cursor = conn.execute("""
                SELECT sql FROM sqlite_master 
                WHERE type='table' AND name='checkpoints'
            """)
            schema = cursor.fetchone()[0]
            
            assert "thread_id TEXT PRIMARY KEY" in schema
            assert "state_data TEXT NOT NULL" in schema
            assert "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in schema
            assert "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP" in schema
            
            # Check trigger exists
            cursor = conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='trigger' AND name='update_checkpoint_timestamp'
            """)
            assert cursor.fetchone() is not None
    
    def test_sqlite_checkpointer_save_success(self, checkpointer, sample_state):
        """Test successful checkpoint save."""
        thread_id = "test_thread_123"
        
        checkpointer.save(thread_id, sample_state)
        
        # Verify data was saved
        with sqlite3.connect(checkpointer.db_path) as conn:
            cursor = conn.execute("""
                SELECT thread_id, state_data FROM checkpoints 
                WHERE thread_id = ?
            """, (thread_id,))
            result = cursor.fetchone()
            
            assert result is not None
            assert result[0] == thread_id
            
            # Verify state data can be parsed
            saved_data = json.loads(result[1])
            assert saved_data["data"]["test_key"] == "test_value"
            assert saved_data["data"]["number"] == 42
            assert len(saved_data["message_history"]) == 2
            assert saved_data["tool_outputs"]["tool1"] == "output1"
    
    def test_sqlite_checkpointer_save_empty_thread_id_raises_error(self, checkpointer, sample_state):
        """Test that saving with empty thread_id raises ValueError."""
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.save("", sample_state)
        
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.save(None, sample_state)
    
    def test_sqlite_checkpointer_save_overwrite_existing(self, checkpointer, sample_state):
        """Test that saving overwrites existing checkpoint."""
        thread_id = "test_thread_overwrite"
        
        # Save initial state
        checkpointer.save(thread_id, sample_state)
        
        # Modify state and save again
        modified_state = sample_state.model_copy(deep=True)
        modified_state.data["new_key"] = "new_value"
        modified_state.add_message("New message")
        
        checkpointer.save(thread_id, modified_state)
        
        # Verify only the new state exists
        with sqlite3.connect(checkpointer.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*), state_data FROM checkpoints 
                WHERE thread_id = ?
                GROUP BY thread_id
            """, (thread_id,))
            result = cursor.fetchone()
            
            assert result[0] == 1  # Only one record
            saved_data = json.loads(result[1])
            assert "new_key" in saved_data["data"]
            assert len(saved_data["message_history"]) == 3
    
    def test_sqlite_checkpointer_load_success(self, checkpointer, sample_state):
        """Test successful checkpoint load."""
        thread_id = "test_thread_load"
        
        # Save state first
        checkpointer.save(thread_id, sample_state)
        
        # Load state
        loaded_state = checkpointer.load(thread_id)
        
        assert loaded_state is not None
        assert isinstance(loaded_state, State)
        assert loaded_state.data["test_key"] == sample_state.data["test_key"]
        assert loaded_state.data["number"] == sample_state.data["number"]
        assert loaded_state.message_history == sample_state.message_history
        assert loaded_state.tool_outputs == sample_state.tool_outputs
    
    def test_sqlite_checkpointer_load_nonexistent_returns_none(self, checkpointer):
        """Test that loading nonexistent checkpoint returns None."""
        result = checkpointer.load("nonexistent_thread")
        
        assert result is None
    
    def test_sqlite_checkpointer_load_empty_thread_id_raises_error(self, checkpointer):
        """Test that loading with empty thread_id raises ValueError."""
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.load("")
        
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.load(None)
    
    def test_sqlite_checkpointer_delete_checkpoint_success(self, checkpointer, sample_state):
        """Test successful checkpoint deletion."""
        thread_id = "test_thread_delete"
        
        # Save state first
        checkpointer.save(thread_id, sample_state)
        
        # Verify it exists
        assert checkpointer.load(thread_id) is not None
        
        # Delete checkpoint
        result = checkpointer.delete_checkpoint(thread_id)
        
        assert result is True
        assert checkpointer.load(thread_id) is None
    
    def test_sqlite_checkpointer_delete_nonexistent_returns_false(self, checkpointer):
        """Test that deleting nonexistent checkpoint returns False."""
        result = checkpointer.delete_checkpoint("nonexistent_thread")
        
        assert result is False
    
    def test_sqlite_checkpointer_delete_empty_thread_id_raises_error(self, checkpointer):
        """Test that deleting with empty thread_id raises ValueError."""
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.delete_checkpoint("")
        
        with pytest.raises(ValueError, match="Thread ID cannot be empty"):
            checkpointer.delete_checkpoint(None)
    
    def test_sqlite_checkpointer_list_checkpoints_empty(self, checkpointer):
        """Test listing checkpoints when database is empty."""
        result = checkpointer.list_checkpoints()
        
        assert result == []
    
    def test_sqlite_checkpointer_list_checkpoints_with_data(self, checkpointer, sample_state):
        """Test listing checkpoints with data."""
        thread_ids = ["thread1", "thread2", "thread3"]
        
        # Save multiple checkpoints
        for thread_id in thread_ids:
            checkpointer.save(thread_id, sample_state)
        
        result = checkpointer.list_checkpoints()
        
        assert len(result) == 3
        assert set(result) == set(thread_ids)
    
    def test_sqlite_checkpointer_cleanup_old_checkpoints(self, checkpointer, sample_state):
        """Test cleanup of old checkpoints."""
        # Create more checkpoints than keep_count
        for i in range(10):
            checkpointer.save(f"thread_{i}", sample_state)
        
        # Verify all exist
        assert len(checkpointer.list_checkpoints()) == 10
        
        # Cleanup, keeping only 5
        deleted_count = checkpointer.cleanup_old_checkpoints(keep_count=5)
        
        assert deleted_count == 5
        assert len(checkpointer.list_checkpoints()) == 5
    
    def test_sqlite_checkpointer_cleanup_keep_count_validation(self, checkpointer):
        """Test cleanup validation of keep_count parameter."""
        with pytest.raises(ValueError, match="Keep count must be at least 1"):
            checkpointer.cleanup_old_checkpoints(keep_count=0)
        
        with pytest.raises(ValueError, match="Keep count must be at least 1"):
            checkpointer.cleanup_old_checkpoints(keep_count=-1)
    
    def test_sqlite_checkpointer_cleanup_when_less_than_keep_count(self, checkpointer, sample_state):
        """Test cleanup when fewer checkpoints exist than keep_count."""
        # Create only 3 checkpoints
        for i in range(3):
            checkpointer.save(f"thread_{i}", sample_state)
        
        # Try to keep 5 (more than exist)
        deleted_count = checkpointer.cleanup_old_checkpoints(keep_count=5)
        
        assert deleted_count == 0
        assert len(checkpointer.list_checkpoints()) == 3
    
    def test_sqlite_checkpointer_complex_state_serialization(self, checkpointer):
        """Test serialization of complex state objects."""
        complex_state = State()
        complex_state.data.update({
            "string": "test",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "none_value": None,
            "list": [1, "two", 3.0, True, None],
            "nested_dict": {
                "inner": {
                    "deep": {
                        "value": "nested"
                    }
                }
            },
            "unicode": "Hello 世界 🌍",
            "empty_structures": {
                "empty_list": [],
                "empty_dict": {}
            }
        })
        
        complex_state.message_history.extend([
            "Message with unicode: 🚀",
            "Empty message: ",
            "Message with special chars: !@#$%^&*()",
            "Multi\nline\nmessage"
        ])
        
        complex_state.tool_outputs.update({
            "tool_complex": {
                "result": "success",
                "data": [1, 2, 3],
                "metadata": {
                    "timestamp": "2023-01-01T00:00:00Z"
                }
            },
            "tool_unicode": "Result with unicode: 你好世界"
        })
        
        thread_id = "complex_thread"
        
        # Save and load complex state
        checkpointer.save(thread_id, complex_state)
        loaded_state = checkpointer.load(thread_id)
        
        # Verify all data is preserved
        assert loaded_state.data == complex_state.data
        assert loaded_state.message_history == complex_state.message_history
        assert loaded_state.tool_outputs == complex_state.tool_outputs
    
    @patch('sqlite3.connect')
    def test_sqlite_checkpointer_database_error_handling_save(self, mock_connect, checkpointer, sample_state):
        """Test error handling during save operation."""
        mock_connect.side_effect = sqlite3.Error("Database error")
        
        with pytest.raises(sqlite3.Error, match="Failed to save checkpoint"):
            checkpointer.save("test_thread", sample_state)
    
    @patch('sqlite3.connect')
    def test_sqlite_checkpointer_database_error_handling_load(self, mock_connect, checkpointer):
        """Test error handling during load operation."""
        mock_connect.side_effect = sqlite3.Error("Database error")
        
        with pytest.raises(sqlite3.Error, match="Failed to load checkpoint"):
            checkpointer.load("test_thread")
    
    def test_sqlite_checkpointer_corrupted_data_handling(self, checkpointer, temp_db_path):
        """Test handling of corrupted data in database."""
        thread_id = "corrupted_thread"
        
        # Manually insert corrupted JSON data
        with sqlite3.connect(temp_db_path) as conn:
            conn.execute("""
                INSERT INTO checkpoints (thread_id, state_data)
                VALUES (?, ?)
            """, (thread_id, "invalid json data"))
            conn.commit()
        
        # Attempt to load should raise JSONDecodeError
        with pytest.raises(json.JSONDecodeError, match="Failed to deserialize state"):
            checkpointer.load(thread_id)
    
    def test_sqlite_checkpointer_concurrent_access(self, checkpointer, sample_state):
        """Test concurrent access to the database."""
        import threading
        import time
        
        results = []
        errors = []
        
        def save_worker(thread_num):
            try:
                for i in range(10):
                    state = sample_state.model_copy(deep=True)
                    state.data["thread"] = thread_num
                    state.data["iteration"] = i
                    checkpointer.save(f"thread_{thread_num}_{i}", state)
                    time.sleep(0.001)  # Small delay
                results.append(f"Thread {thread_num} completed")
            except Exception as e:
                errors.append(f"Thread {thread_num} error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_worker, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 5
        
        # Verify all data was saved
        checkpoints = checkpointer.list_checkpoints()
        assert len(checkpoints) == 50  # 5 threads * 10 iterations each
    
    def test_sqlite_checkpointer_database_initialization_error_handling(self):
        """Test error handling during database initialization."""
        # Try to create database in a non-existent directory
        invalid_path = "/nonexistent/directory/test.db"
        
        with pytest.raises(sqlite3.Error, match="Failed to initialize checkpoint database"):
            SQLiteCheckpointSaver(invalid_path)
    
    def test_sqlite_checkpointer_state_modification_independence(self, checkpointer, sample_state):
        """Test that saved states are independent of original state modifications."""
        thread_id = "independence_test"
        
        # Save state
        checkpointer.save(thread_id, sample_state)
        
        # Modify original state
        sample_state.data["modified"] = "after_save"
        sample_state.add_message("Modified after save")
        
        # Load state should not have modifications
        loaded_state = checkpointer.load(thread_id)
        
        assert "modified" not in loaded_state.data
        assert "Modified after save" not in loaded_state.message_history
        assert len(loaded_state.message_history) == 2  # Original count
    
    def test_sqlite_checkpointer_large_state_handling(self, checkpointer):
        """Test handling of large state objects."""
        large_state = State()
        
        # Create large data structures
        large_state.data["large_list"] = list(range(10000))
        large_state.data["large_string"] = "x" * 100000
        large_state.data["large_dict"] = {f"key_{i}": f"value_{i}" for i in range(1000)}
        
        for i in range(1000):
            large_state.add_message(f"Message {i}")
        
        for i in range(100):
            large_state.tool_outputs[f"tool_{i}"] = f"output_{i}" * 100
        
        thread_id = "large_state_test"
        
        # Save and load large state
        checkpointer.save(thread_id, large_state)
        loaded_state = checkpointer.load(thread_id)
        
        # Verify data integrity
        assert len(loaded_state.data["large_list"]) == 10000
        assert len(loaded_state.data["large_string"]) == 100000
        assert len(loaded_state.data["large_dict"]) == 1000
        assert len(loaded_state.message_history) == 1000
        assert len(loaded_state.tool_outputs) == 100
        
        # Spot check some values
        assert loaded_state.data["large_list"][5000] == 5000
        assert loaded_state.data["large_dict"]["key_500"] == "value_500"
        assert loaded_state.message_history[500] == "Message 500"
        assert "output_50" * 100 in loaded_state.tool_outputs["tool_50"]