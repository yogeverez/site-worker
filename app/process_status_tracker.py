"""
Process Status Tracker - Tracks and updates the status of each phase in Firestore
"""
import time
import logging
from typing import Dict, Any, Optional, List
from google.cloud import firestore
from app.database import get_db
from datetime import datetime

logger = logging.getLogger(__name__)

class ProcessStatusTracker:
    """
    Tracks the status of the site generation process across all phases.
    Updates status in Firestore for real-time client monitoring.
    """
    
    PHASES = {
        "initialization": "Initializing process",
        "user_data_collection": "Collecting and processing user data",
        "research": "Conducting research",
        "content_generation": "Generating content",
        "translation": "Translating content",
        "completion": "Process completed"
    }
    
    def __init__(self, uid: str, session_id: int, parent_logger: Optional[logging.Logger] = None):
        self.uid = uid
        self.session_id = session_id
        self.logger = parent_logger or logger
        self.db = get_db()
        self.status_ref = self.db.collection("siteGenerationStatus").document(uid)
        self.start_time = time.time()
        
        # Initialize status document
        self._initialize_status()
    
    def _initialize_status(self):
        """Initialize the status document in Firestore."""
        initial_status = {
            "uid": self.uid,
            "session_id": self.session_id,
            "status": "initializing",
            "current_phase": "initialization",
            "phases_completed": [],
            "phases_status": {
                phase: "pending" for phase in self.PHASES.keys()
            },
            "start_time": self.start_time,
            "last_updated": self.start_time,
            "created_at": firestore.SERVER_TIMESTAMP,
            "progress_percentage": 0,
            "estimated_time_remaining": None,
            "errors": []
        }
        
        self.status_ref.set(initial_status)
        self.logger.info(f"Initialized process status for UID: {self.uid}")
    
    def update_phase(self, phase: str, status: str = "in_progress", details: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        """
        Update the status of a specific phase.
        
        Args:
            phase: The phase name (must be one of PHASES keys)
            status: Status of the phase (pending, in_progress, completed, error)
            details: Additional details about the phase
        """
        if phase not in self.PHASES:
            self.logger.warning(f"Unknown phase: {phase}")
            return
        
        update_data = {
            "current_phase": phase,
            "status": f"{phase}_{status}",
            f"phases_status.{phase}": status,
            "last_updated": time.time(),
            "last_phase_update": {
                "phase": phase,
                "status": status,
                "timestamp": time.time(),
                "details": details or {}
            }
        }
        
        # Add metadata if provided
        if metadata:
            update_data["last_phase_update"]["metadata"] = metadata
            
        # Add error if provided
        if error:
            update_data["last_phase_update"]["error"] = error
        
        # Update phase description
        update_data["current_phase_description"] = self.PHASES[phase]
        
        # Calculate progress
        completed_phases = []
        if status == "completed":
            # Get current completed phases
            current_doc = self.status_ref.get()
            if current_doc.exists:
                completed_phases = current_doc.to_dict().get("phases_completed", [])
                if phase not in completed_phases:
                    completed_phases.append(phase)
                    update_data["phases_completed"] = completed_phases
        
        # Calculate progress percentage
        progress = (len(completed_phases) / len(self.PHASES)) * 100
        update_data["progress_percentage"] = round(progress, 1)
        
        # Estimate time remaining (simple linear estimation)
        if progress > 0:
            elapsed = time.time() - self.start_time
            estimated_total = (elapsed / progress) * 100
            estimated_remaining = estimated_total - elapsed
            update_data["estimated_time_remaining_seconds"] = max(0, round(estimated_remaining))
        
        self.status_ref.update(update_data)
        self.logger.info(f"Updated phase '{phase}' to status '{status}' for UID: {self.uid}")
    
    def add_phase_metric(self, phase: str, metric_name: str, value: Any):
        """
        Add a metric for a specific phase.
        
        Args:
            phase: The phase name
            metric_name: Name of the metric
            value: Value of the metric
        """
        metric_key = f"phase_metrics.{phase}.{metric_name}"
        self.status_ref.update({
            metric_key: value,
            "last_updated": time.time()
        })
    
    def add_error(self, phase: str, error_message: str, error_type: str = "general"):
        """
        Add an error to the status.
        
        Args:
            phase: The phase where the error occurred
            error_message: The error message
            error_type: Type of error (general, api_error, validation_error, etc.)
        """
        error_entry = {
            "phase": phase,
            "type": error_type,
            "message": error_message,
            "timestamp": time.time()
        }
        
        self.status_ref.update({
            "errors": firestore.ArrayUnion([error_entry]),
            "has_errors": True,
            "last_error": error_entry,
            "last_updated": time.time()
        })
        
        self.logger.error(f"Error in phase '{phase}': {error_message}")
    
    def complete_process(self, final_status: str = "completed", summary: Optional[Dict[str, Any]] = None):
        """
        Mark the entire process as completed.
        
        Args:
            final_status: Final status (completed, failed, partial)
            summary: Summary of the entire process
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        update_data = {
            "status": final_status,
            "current_phase": "completion",
            "end_time": end_time,
            "duration_seconds": round(duration, 2),
            "last_updated": end_time,
            "completed_at": firestore.SERVER_TIMESTAMP
        }
        
        if summary:
            update_data["summary"] = summary
        
        # Add final metrics
        update_data["final_metrics"] = {
            "total_duration_seconds": round(duration, 2),
            "phases_completed": len([p for p, s in self._get_phases_status().items() if s == "completed"]),
            "total_phases": len(self.PHASES),
            "success": final_status == "completed"
        }
        
        self.status_ref.update(update_data)
        self.logger.info(f"Process completed with status '{final_status}' for UID: {self.uid}")
    
    def _get_phases_status(self) -> Dict[str, str]:
        """Get current status of all phases."""
        doc = self.status_ref.get()
        if doc.exists:
            return doc.to_dict().get("phases_status", {})
        return {}
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get the current status document."""
        doc = self.status_ref.get()
        if doc.exists:
            return doc.to_dict()
        return {}

class PhaseTimer:
    """Context manager for timing phases."""
    
    def __init__(self, tracker: ProcessStatusTracker, phase: str):
        self.tracker = tracker
        self.phase = phase
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.tracker.update_phase(self.phase, "in_progress")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.tracker.update_phase(self.phase, "completed")
            self.tracker.add_phase_metric(self.phase, "duration_seconds", round(duration, 2))
        else:
            self.tracker.update_phase(self.phase, "error")
            self.tracker.add_error(self.phase, str(exc_val), error_type=exc_type.__name__)
            self.tracker.add_phase_metric(self.phase, "duration_seconds", round(duration, 2))
        
        return False  # Don't suppress exceptions

def create_status_tracker(uid: str, session_id: int, parent_logger: Optional[logging.Logger] = None) -> ProcessStatusTracker:
    """
    Create a new process status tracker.
    
    Args:
        uid: User ID
        session_id: Session ID
        parent_logger: Parent logger
    
    Returns:
        ProcessStatusTracker instance
    """
    return ProcessStatusTracker(uid, session_id, parent_logger)
