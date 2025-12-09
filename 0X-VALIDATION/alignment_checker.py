"""
Alignment Checker for Phase X
Checks alignment with project goals
"""

from typing import Dict, List, Any


class AlignmentChecker:
    """Checks alignment with goals"""
    
    def __init__(self, project_goals: Dict[str, Any] = None):
        self.project_goals = project_goals or self._default_goals()
        self.alignment_score = 0.0
        self.issues = []
        
    def check_alignment(self, phase_results: Dict[str, Any], phase_name: str) -> Dict[str, Any]:
        """Check alignment with project goals"""
        print(f"\n{'=' * 60}")
        print(f"PHASE X: Alignment Check - {phase_name}")
        print(f"{'=' * 60}")
        
        # 1. Check against requirements
        requirements_score = self._check_requirements(phase_results)
        
        # 2. Check scope adherence
        scope_score = self._check_scope(phase_results)
        
        # 3. Identify scope creep
        scope_creep = self._identify_scope_creep(phase_results)
        
        # 4. Validate deliverable relevance
        relevance_score = self._check_relevance(phase_results)
        
        # 5. Track goal achievement
        achievement_score = self._track_achievement(phase_results)
        
        # Calculate overall alignment
        self.alignment_score = (
            requirements_score * 0.3 +
            scope_score * 0.25 +
            relevance_score * 0.25 +
            achievement_score * 0.20
        )
        
        return {
            'alignment_score': self.alignment_score,
            'requirements': requirements_score,
            'scope': scope_score,
            'scope_creep': scope_creep,
            'relevance': relevance_score,
            'achievement': achievement_score,
            'aligned': self.alignment_score >= 7.0,
            'issues': self.issues
        }
    
    def _default_goals(self) -> Dict[str, Any]:
        """Default project goals"""
        return {
            'primary_objective': 'Platform consolidation',
            'success_criteria': ['Reduce duplicates', 'Improve quality', 'Maintain functionality'],
            'scope': ['Code consolidation', 'Testing', 'Documentation'],
            'constraints': ['No breaking changes', 'Maintain backwards compatibility']
        }
    
    def _check_requirements(self, results: Dict[str, Any]) -> float:
        """Check against requirements"""
        print("\n[1/5] Checking requirements...")
        
        required_deliverables = results.get('required_deliverables', [])
        actual_deliverables = results.get('deliverables', [])
        
        if not required_deliverables:
            print("  ⏭️  No requirements defined")
            return 8.0
        
        met_requirements = sum(1 for req in required_deliverables if req in actual_deliverables)
        score = (met_requirements / len(required_deliverables)) * 10.0
        
        print(f"  Requirements met: {met_requirements}/{len(required_deliverables)}")
        print(f"  Score: {score:.1f}/10.0")
        
        if score < 7.0:
            self.issues.append(f"Unmet requirements: {len(required_deliverables) - met_requirements}")
        
        return score
    
    def _check_scope(self, results: Dict[str, Any]) -> float:
        """Check scope adherence"""
        print("\n[2/5] Checking scope adherence...")
        
        defined_scope = self.project_goals.get('scope', [])
        actual_work = results.get('work_completed', [])
        
        if not defined_scope:
            print("  ⏭️  No scope defined")
            return 8.0
        
        in_scope = sum(1 for work in actual_work if any(s in work for s in defined_scope))
        score = (in_scope / len(actual_work)) * 10.0 if actual_work else 8.0
        
        print(f"  In-scope work: {in_scope}/{len(actual_work)}")
        print(f"  Score: {score:.1f}/10.0")
        
        return score
    
    def _identify_scope_creep(self, results: Dict[str, Any]) -> List[str]:
        """Identify scope creep"""
        print("\n[3/5] Identifying scope creep...")
        
        defined_scope = self.project_goals.get('scope', [])
        actual_work = results.get('work_completed', [])
        
        scope_creep = []
        for work in actual_work:
            if not any(s in work for s in defined_scope):
                scope_creep.append(work)
        
        if scope_creep:
            print(f"  ⚠️  Scope creep detected: {len(scope_creep)} items")
            self.issues.append(f"Scope creep: {len(scope_creep)} out-of-scope items")
        else:
            print(f"  ✅ No scope creep detected")
        
        return scope_creep
    
    def _check_relevance(self, results: Dict[str, Any]) -> float:
        """Check deliverable relevance"""
        print("\n[4/5] Checking deliverable relevance...")
        
        primary_objective = self.project_goals.get('primary_objective', '')
        deliverables = results.get('deliverables', [])
        
        # Simple relevance check based on keyword matching
        relevant_count = sum(1 for d in deliverables if primary_objective.lower() in str(d).lower())
        score = (relevant_count / len(deliverables)) * 10.0 if deliverables else 8.0
        
        print(f"  Relevant deliverables: {relevant_count}/{len(deliverables)}")
        print(f"  Score: {score:.1f}/10.0")
        
        return score
    
    def _track_achievement(self, results: Dict[str, Any]) -> float:
        """Track goal achievement"""
        print("\n[5/5] Tracking goal achievement...")
        
        success_criteria = self.project_goals.get('success_criteria', [])
        achievements = results.get('achievements', [])
        
        if not success_criteria:
            print("  ⏭️  No success criteria defined")
            return 8.0
        
        met_criteria = sum(1 for crit in success_criteria if any(crit.lower() in str(a).lower() for a in achievements))
        score = (met_criteria / len(success_criteria)) * 10.0
        
        print(f"  Criteria met: {met_criteria}/{len(success_criteria)}")
        print(f"  Score: {score:.1f}/10.0")
        
        return score


if __name__ == "__main__":
    checker = AlignmentChecker()
    
    phase_results = {
        'required_deliverables': ['consolidation', 'testing'],
        'deliverables': ['consolidation', 'testing'],
        'work_completed': ['code merge', 'test generation'],
        'achievements': ['reduced duplicates', 'improved quality']
    }
    
    result = checker.check_alignment(phase_results, 'Phase 3')
    print(f"\n\nAlignment Score: {result['alignment_score']:.1f}/10.0")
    print(f"Aligned: {result['aligned']}")
