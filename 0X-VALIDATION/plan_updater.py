"""
Plan Updater for Phase X
Updates execution plan based on real results
"""

from typing import Dict, List, Any
import json
from datetime import datetime


class PlanUpdater:
    """Updates execution plan dynamically"""
    
    def __init__(self):
        self.updates = []
        self.original_plan = {}
        self.updated_plan = {}
        
    def update_plan(self, current_plan: Dict[str, Any], phase_results: Dict[str, Any], quality_score: float) -> Dict[str, Any]:
        """Update plan based on real results"""
        print(f"\n{'=' * 60}")
        print("PHASE X: Plan Update")
        print(f"{'=' * 60}")
        
        self.original_plan = current_plan.copy()
        self.updated_plan = current_plan.copy()
        
        # 1. Adjust timelines
        self._adjust_timelines(phase_results, quality_score)
        
        # 2. Update resource allocation
        self._update_resources(phase_results, quality_score)
        
        # 3. Modify next phase parameters
        self._modify_next_phase(phase_results, quality_score)
        
        # 4. Update model selection
        self._update_model_strategy(phase_results, quality_score)
        
        # 5. Revise success criteria
        self._revise_criteria(phase_results, quality_score)
        
        # 6. Document changes
        self._document_changes()
        
        return self.updated_plan
    
    def _adjust_timelines(self, results: Dict[str, Any], quality_score: float):
        """Adjust timelines based on actual performance"""
        print("\n[1/6] Adjusting timelines...")
        
        actual_time = results.get('execution_time', 0)
        estimated_time = self.original_plan.get('estimated_time', actual_time)
        
        if actual_time > estimated_time * 1.2:
            # Taking longer than expected
            adjustment_factor = actual_time / estimated_time
            self.updated_plan['estimated_time'] = estimated_time * adjustment_factor
            self.updates.append(f"Timeline extended by {(adjustment_factor-1)*100:.1f}%")
            print(f"  ⚠️  Timeline extended: {estimated_time:.1f}s → {self.updated_plan['estimated_time']:.1f}s")
        elif actual_time < estimated_time * 0.8:
            # Faster than expected
            adjustment_factor = actual_time / estimated_time
            self.updated_plan['estimated_time'] = estimated_time * adjustment_factor
            self.updates.append(f"Timeline reduced by {(1-adjustment_factor)*100:.1f}%")
            print(f"  ✅ Timeline reduced: {estimated_time:.1f}s → {self.updated_plan['estimated_time']:.1f}s")
        else:
            print(f"  ✅ Timeline on track: {estimated_time:.1f}s")
    
    def _update_resources(self, results: Dict[str, Any], quality_score: float):
        """Update resource allocation"""
        print("\n[2/6] Updating resource allocation...")
        
        if quality_score < 7.0:
            # Need more resources
            current_resources = self.original_plan.get('resources', {})
            self.updated_plan['resources'] = {
                'cpu': current_resources.get('cpu', 1) * 1.5,
                'memory': current_resources.get('memory', 1) * 1.5,
                'models': current_resources.get('models', ['basic']) + ['advanced']
            }
            self.updates.append("Increased resource allocation")
            print(f"  ⚠️  Resources increased (quality: {quality_score:.1f}/10.0)")
        else:
            print(f"  ✅ Resources adequate (quality: {quality_score:.1f}/10.0)")
    
    def _modify_next_phase(self, results: Dict[str, Any], quality_score: float):
        """Modify next phase parameters"""
        print("\n[3/6] Modifying next phase parameters...")
        
        next_phase = self.original_plan.get('next_phase', {})
        
        if quality_score < 7.0:
            # Add more validation
            next_phase['validation_level'] = 'strict'
            next_phase['review_required'] = True
            self.updates.append("Increased validation for next phase")
            print(f"  ⚠️  Validation increased for next phase")
        elif quality_score >= 9.0:
            # Can proceed with confidence
            next_phase['validation_level'] = 'standard'
            next_phase['review_required'] = False
            self.updates.append("Standard validation for next phase")
            print(f"  ✅ Standard validation for next phase")
        
        self.updated_plan['next_phase'] = next_phase
    
    def _update_model_strategy(self, results: Dict[str, Any], quality_score: float):
        """Update model selection strategy"""
        print("\n[4/6] Updating model selection strategy...")
        
        model_performance = results.get('model_performance', {})
        
        if model_performance:
            # Sort models by performance
            sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
            best_models = [m[0] for m in sorted_models[:3]]
            
            self.updated_plan['preferred_models'] = best_models
            self.updates.append(f"Updated model preferences: {', '.join(best_models)}")
            print(f"  ✅ Preferred models: {', '.join(best_models)}")
        else:
            print(f"  ⏭️  No model performance data")
    
    def _revise_criteria(self, results: Dict[str, Any], quality_score: float):
        """Revise success criteria"""
        print("\n[5/6] Revising success criteria...")
        
        if quality_score < 7.0:
            # Make criteria more lenient temporarily
            self.updated_plan['success_threshold'] = 6.5
            self.updates.append("Temporarily reduced success threshold")
            print(f"  ⚠️  Success threshold: 7.0 → 6.5")
        elif quality_score >= 9.0:
            # Can raise the bar
            self.updated_plan['success_threshold'] = 7.5
            self.updates.append("Increased success threshold")
            print(f"  ✅ Success threshold: 7.0 → 7.5")
        else:
            print(f"  ✅ Success threshold unchanged: 7.0")
    
    def _document_changes(self):
        """Document all changes"""
        print("\n[6/6] Documenting changes...")
        
        self.updated_plan['plan_updates'] = {
            'timestamp': datetime.now().isoformat(),
            'changes': self.updates,
            'change_count': len(self.updates)
        }
        
        print(f"  ✅ Documented {len(self.updates)} changes")


if __name__ == "__main__":
    updater = PlanUpdater()
    
    current_plan = {
        'estimated_time': 100,
        'resources': {'cpu': 1, 'memory': 1},
        'next_phase': {},
        'success_threshold': 7.0
    }
    
    phase_results = {
        'execution_time': 150,
        'model_performance': {'gemini': 9.0, 'mistral': 8.5}
    }
    
    updated = updater.update_plan(current_plan, phase_results, 8.5)
    print(f"\n\nPlan Updates: {len(updater.updates)}")
    for update in updater.updates:
        print(f"  - {update}")
