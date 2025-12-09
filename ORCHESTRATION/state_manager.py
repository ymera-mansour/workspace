"""
State Manager
=============

Manages workflow state persistence for checkpointing and recovery.
Enables pause/resume functionality for long-running workflows.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class StateManager:
    """Manages workflow state persistence"""
    
    def __init__(self, state_dir: str = ".ymera_state", config: Dict[str, Any] = None):
        """
        Initialize state manager
        
        Args:
            state_dir: Directory for state files
            config: Platform configuration
        """
        self.state_dir = Path(state_dir)
        self.config = config or {}
        self.current_state = {}
        self.state_history = []
        self.max_history = self.config.get("state_manager", {}).get("max_history", 50)
        
        logger.info(f"State manager initialized with dir: {self.state_dir}")
    
    async def initialize(self):
        """Initialize state manager"""
        try:
            # Create state directory
            self.state_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.state_dir / "checkpoints").mkdir(exist_ok=True)
            (self.state_dir / "history").mkdir(exist_ok=True)
            (self.state_dir / "snapshots").mkdir(exist_ok=True)
            
            logger.info("✅ State manager initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize state manager: {e}")
            raise
    
    async def save_state(
        self,
        state: Dict[str, Any],
        checkpoint: bool = True,
        state_id: Optional[str] = None
    ) -> str:
        """
        Save workflow state
        
        Args:
            state: State data to save
            checkpoint: Whether this is a checkpoint
            state_id: Optional state ID (generated if not provided)
        
        Returns:
            str: State ID
        """
        try:
            # Generate state ID if not provided
            if state_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                state_hash = self._generate_hash(state)[:8]
                state_id = f"state_{timestamp}_{state_hash}"
            
            # Add metadata
            state["_metadata"] = {
                "state_id": state_id,
                "timestamp": datetime.now().isoformat(),
                "checkpoint": checkpoint
            }
            
            # Save to disk
            if checkpoint:
                state_file = self.state_dir / "checkpoints" / f"{state_id}.json"
            else:
                state_file = self.state_dir / "history" / f"{state_id}.json"
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Update current state
            self.current_state = state
            
            # Add to history
            self.state_history.append({
                "state_id": state_id,
                "timestamp": datetime.now().isoformat(),
                "checkpoint": checkpoint
            })
            
            # Trim history if needed
            if len(self.state_history) > self.max_history:
                self.state_history = self.state_history[-self.max_history:]
            
            logger.info(f"💾 State saved: {state_id}")
            return state_id
            
        except Exception as e:
            logger.error(f"❌ Failed to save state: {e}")
            raise
    
    async def load_state(self, state_id: str) -> Dict[str, Any]:
        """
        Load workflow state
        
        Args:
            state_id: State ID to load
        
        Returns:
            dict: State data
        """
        try:
            # Try checkpoint first
            state_file = self.state_dir / "checkpoints" / f"{state_id}.json"
            
            if not state_file.exists():
                # Try history
                state_file = self.state_dir / "history" / f"{state_id}.json"
            
            if not state_file.exists():
                raise FileNotFoundError(f"State not found: {state_id}")
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.current_state = state
            logger.info(f"📂 State loaded: {state_id}")
            return state
            
        except Exception as e:
            logger.error(f"❌ Failed to load state: {e}")
            raise
    
    async def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List available checkpoints
        
        Returns:
            list: List of checkpoint metadata
        """
        try:
            checkpoints = []
            checkpoint_dir = self.state_dir / "checkpoints"
            
            if checkpoint_dir.exists():
                for checkpoint_file in sorted(checkpoint_dir.glob("*.json")):
                    with open(checkpoint_file, 'r') as f:
                        state = json.load(f)
                        metadata = state.get("_metadata", {})
                        metadata["file"] = checkpoint_file.name
                        checkpoints.append(metadata)
            
            return checkpoints
            
        except Exception as e:
            logger.error(f"❌ Failed to list checkpoints: {e}")
            return []
    
    async def delete_checkpoint(self, state_id: str):
        """
        Delete a checkpoint
        
        Args:
            state_id: State ID to delete
        """
        try:
            state_file = self.state_dir / "checkpoints" / f"{state_id}.json"
            
            if state_file.exists():
                state_file.unlink()
                logger.info(f"🗑️  Checkpoint deleted: {state_id}")
            else:
                logger.warning(f"Checkpoint not found: {state_id}")
                
        except Exception as e:
            logger.error(f"❌ Failed to delete checkpoint: {e}")
            raise
    
    async def create_snapshot(
        self,
        name: str,
        description: str = ""
    ) -> str:
        """
        Create a named snapshot of current state
        
        Args:
            name: Snapshot name
            description: Optional description
        
        Returns:
            str: Snapshot ID
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"snapshot_{name}_{timestamp}"
            
            snapshot = {
                **self.current_state,
                "_snapshot_metadata": {
                    "snapshot_id": snapshot_id,
                    "name": name,
                    "description": description,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            snapshot_file = self.state_dir / "snapshots" / f"{snapshot_id}.json"
            
            with open(snapshot_file, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"📸 Snapshot created: {snapshot_id}")
            return snapshot_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create snapshot: {e}")
            raise
    
    async def restore_snapshot(self, snapshot_id: str):
        """
        Restore from snapshot
        
        Args:
            snapshot_id: Snapshot ID to restore
        """
        try:
            snapshot_file = self.state_dir / "snapshots" / f"{snapshot_id}.json"
            
            if not snapshot_file.exists():
                raise FileNotFoundError(f"Snapshot not found: {snapshot_id}")
            
            with open(snapshot_file, 'r') as f:
                snapshot = json.load(f)
            
            # Remove snapshot metadata
            snapshot.pop("_snapshot_metadata", None)
            
            self.current_state = snapshot
            logger.info(f"♻️  Snapshot restored: {snapshot_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to restore snapshot: {e}")
            raise
    
    async def get_current_state(self) -> Dict[str, Any]:
        """
        Get current state
        
        Returns:
            dict: Current state
        """
        return self.current_state.copy()
    
    async def update_state(self, updates: Dict[str, Any]):
        """
        Update current state
        
        Args:
            updates: State updates to apply
        """
        self.current_state.update(updates)
        logger.debug("State updated")
    
    async def get_state_history(self) -> List[Dict[str, Any]]:
        """
        Get state history
        
        Returns:
            list: State history
        """
        return self.state_history.copy()
    
    async def cleanup_old_states(self, days: int = 7):
        """
        Cleanup old state files
        
        Args:
            days: Number of days to keep
        """
        try:
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cleaned_count = 0
            
            # Cleanup history
            history_dir = self.state_dir / "history"
            if history_dir.exists():
                for state_file in history_dir.glob("*.json"):
                    file_time = datetime.fromtimestamp(state_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        state_file.unlink()
                        cleaned_count += 1
            
            logger.info(f"🧹 Cleaned up {cleaned_count} old state files")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old states: {e}")
    
    def _generate_hash(self, data: Dict[str, Any]) -> str:
        """Generate hash for state data"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def close(self):
        """Close state manager and cleanup"""
        try:
            # Save final state
            if self.current_state:
                await self.save_state(self.current_state, checkpoint=True)
            
            logger.info("State manager closed")
            
        except Exception as e:
            logger.error(f"Error closing state manager: {e}")
