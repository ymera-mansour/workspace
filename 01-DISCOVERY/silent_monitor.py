"""Silent Monitor - External observer for model performance tracking"""
import time
from typing import Dict, Any, List

class SilentMonitor:
    """Monitors model performance without interfering with workflow"""
    
    def __init__(self):
        self.events = []
        self.running = False
        
    async def start_monitoring(self):
        """Start silent monitoring"""
        self.running = True
        print("📊 Silent Monitor: Started")
        
    def record_event(self, event: Dict[str, Any]):
        """Record a model invocation event"""
        event["timestamp"] = time.time()
        self.events.append(event)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        return {
            "total_events": len(self.events),
            "events": self.events
        }
        
    async def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        print("📊 Silent Monitor: Stopped")
