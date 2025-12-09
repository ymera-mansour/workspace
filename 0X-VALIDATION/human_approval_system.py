"""
Human Approval System for Phase X
Manages human approval workflows
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json


class HumanApprovalSystem:
    """Manages human approval workflows"""
    
    def __init__(self):
        self.pending_approvals = []
        self.approved = []
        self.rejected = []
        self.timeout_minutes = 60
        
    def request_approval(self, approval_request: Dict[str, Any]) -> str:
        """Request human approval"""
        print(f"\n{'=' * 60}")
        print("PHASE X: Human Approval Request")
        print(f"{'=' * 60}")
        
        request_id = f"approval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        request = {
            'id': request_id,
            'timestamp': datetime.now().isoformat(),
            'phase': approval_request.get('phase', 'Unknown'),
            'reason': approval_request.get('reason', 'Quality check'),
            'quality_score': approval_request.get('quality_score', 0.0),
            'issues': approval_request.get('issues', []),
            'recommendations': approval_request.get('recommendations', []),
            'status': 'PENDING',
            'timeout_at': (datetime.now() + timedelta(minutes=self.timeout_minutes)).isoformat()
        }
        
        self.pending_approvals.append(request)
        
        self._display_approval_request(request)
        
        return request_id
    
    def _display_approval_request(self, request: Dict[str, Any]):
        """Display approval request"""
        print(f"\nRequest ID: {request['id']}")
        print(f"Phase: {request['phase']}")
        print(f"Quality Score: {request['quality_score']:.1f}/10.0")
        print(f"Reason: {request['reason']}")
        
        if request['issues']:
            print(f"\nIssues ({len(request['issues'])}):")
            for issue in request['issues']:
                print(f"  ❌ {issue}")
        
        if request['recommendations']:
            print(f"\nRecommendations ({len(request['recommendations'])}):")
            for rec in request['recommendations']:
                print(f"  💡 {rec}")
        
        print(f"\nTimeout: {request['timeout_at']}")
        print(f"\n{'=' * 60}")
        print("WAITING FOR HUMAN APPROVAL...")
        print("Options: APPROVE, REJECT, RETRY")
        print(f"{'=' * 60}")
    
    def check_approval_status(self, request_id: str) -> str:
        """Check status of approval request"""
        for request in self.pending_approvals:
            if request['id'] == request_id:
                # Check timeout
                timeout_at = datetime.fromisoformat(request['timeout_at'])
                if datetime.now() > timeout_at:
                    request['status'] = 'TIMEOUT'
                    return 'TIMEOUT'
                return 'PENDING'
        
        for request in self.approved:
            if request['id'] == request_id:
                return 'APPROVED'
        
        for request in self.rejected:
            if request['id'] == request_id:
                return 'REJECTED'
        
        return 'NOT_FOUND'
    
    def approve(self, request_id: str, approver: str = "Human", notes: str = ""):
        """Approve request"""
        for i, request in enumerate(self.pending_approvals):
            if request['id'] == request_id:
                request['status'] = 'APPROVED'
                request['approver'] = approver
                request['approval_notes'] = notes
                request['approved_at'] = datetime.now().isoformat()
                
                self.approved.append(request)
                self.pending_approvals.pop(i)
                
                print(f"\n✅ Request {request_id} APPROVED by {approver}")
                if notes:
                    print(f"   Notes: {notes}")
                return
        
        print(f"\n❌ Request {request_id} not found")
    
    def reject(self, request_id: str, rejector: str = "Human", reason: str = ""):
        """Reject request"""
        for i, request in enumerate(self.pending_approvals):
            if request['id'] == request_id:
                request['status'] = 'REJECTED'
                request['rejector'] = rejector
                request['rejection_reason'] = reason
                request['rejected_at'] = datetime.now().isoformat()
                
                self.rejected.append(request)
                self.pending_approvals.pop(i)
                
                print(f"\n❌ Request {request_id} REJECTED by {rejector}")
                if reason:
                    print(f"   Reason: {reason}")
                return
        
        print(f"\n❌ Request {request_id} not found")
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail of all approvals"""
        return self.pending_approvals + self.approved + self.rejected


if __name__ == "__main__":
    system = HumanApprovalSystem()
    
    # Test approval request
    approval_request = {
        'phase': 'Phase 1: Discovery',
        'reason': 'Quality score below threshold',
        'quality_score': 6.5,
        'issues': ['Data inconsistency', 'Missing files'],
        'recommendations': ['Re-scan files', 'Validate data']
    }
    
    request_id = system.request_approval(approval_request)
    
    # Simulate approval
    print("\n\n[Simulating approval after 5 seconds...]")
    import time
    time.sleep(2)
    
    system.approve(request_id, "Test User", "Approved for testing")
    
    status = system.check_approval_status(request_id)
    print(f"\nFinal Status: {status}")
