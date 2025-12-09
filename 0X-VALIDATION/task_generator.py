"""
Task Generator for Phase X
Generates additional tasks when needed
"""

from typing import Dict, List, Any
from datetime import datetime
import uuid


class TaskGenerator:
    """Generates additional tasks"""
    
    def __init__(self):
        self.generated_tasks = []
        
    def generate_tasks(self, phase_results: Dict[str, Any], quality_score: float, issues: List[str]) -> List[Dict[str, Any]]:
        """Generate additional tasks based on results"""
        print(f"\n{'=' * 60}")
        print("PHASE X: Task Generation")
        print(f"{'=' * 60}")
        
        self.generated_tasks = []
        
        # 1. Generate remediation tasks
        if quality_score < 7.0:
            self._generate_remediation_tasks(phase_results, issues)
        
        # 2. Generate validation tasks
        if quality_score < 8.0:
            self._generate_validation_tasks(phase_results)
        
        # 3. Generate optimization tasks
        if quality_score >= 7.0 and quality_score < 9.0:
            self._generate_optimization_tasks(phase_results)
        
        # 4. Prioritize tasks
        self._prioritize_tasks()
        
        # 5. Add dependencies
        self._add_dependencies()
        
        print(f"\n{'=' * 60}")
        print(f"Generated {len(self.generated_tasks)} tasks")
        print(f"{'=' * 60}")
        
        return self.generated_tasks
    
    def _generate_remediation_tasks(self, results: Dict[str, Any], issues: List[str]):
        """Generate tasks to fix issues"""
        print("\n[1/5] Generating remediation tasks...")
        
        for issue in issues:
            task = {
                'id': str(uuid.uuid4())[:8],
                'type': 'remediation',
                'title': f"Fix: {issue}",
                'description': f"Remediate issue: {issue}",
                'priority': 'HIGH',
                'estimated_time': 30,
                'dependencies': [],
                'created_at': datetime.now().isoformat()
            }
            self.generated_tasks.append(task)
            print(f"  ✅ Created: {task['title']}")
    
    def _generate_validation_tasks(self, results: Dict[str, Any]):
        """Generate additional validation tasks"""
        print("\n[2/5] Generating validation tasks...")
        
        validation_tasks = [
            "Validate data integrity",
            "Cross-check results with source",
            "Verify output completeness",
            "Check edge cases"
        ]
        
        for title in validation_tasks:
            task = {
                'id': str(uuid.uuid4())[:8],
                'type': 'validation',
                'title': title,
                'description': f"Additional validation: {title}",
                'priority': 'MEDIUM',
                'estimated_time': 15,
                'dependencies': [],
                'created_at': datetime.now().isoformat()
            }
            self.generated_tasks.append(task)
            print(f"  ✅ Created: {task['title']}")
    
    def _generate_optimization_tasks(self, results: Dict[str, Any]):
        """Generate optimization tasks"""
        print("\n[3/5] Generating optimization tasks...")
        
        optimization_tasks = [
            "Optimize performance",
            "Improve code quality",
            "Enhance error handling"
        ]
        
        for title in optimization_tasks:
            task = {
                'id': str(uuid.uuid4())[:8],
                'type': 'optimization',
                'title': title,
                'description': f"Optimization task: {title}",
                'priority': 'LOW',
                'estimated_time': 20,
                'dependencies': [],
                'created_at': datetime.now().isoformat()
            }
            self.generated_tasks.append(task)
            print(f"  ✅ Created: {task['title']}")
    
    def _prioritize_tasks(self):
        """Prioritize tasks"""
        print("\n[4/5] Prioritizing tasks...")
        
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        self.generated_tasks.sort(key=lambda t: priority_order.get(t['priority'], 3))
        
        print(f"  ✅ Sorted {len(self.generated_tasks)} tasks by priority")
    
    def _add_dependencies(self):
        """Add task dependencies"""
        print("\n[5/5] Adding dependencies...")
        
        # Remediation tasks must complete before validation
        remediation_ids = [t['id'] for t in self.generated_tasks if t['type'] == 'remediation']
        
        for task in self.generated_tasks:
            if task['type'] == 'validation' and remediation_ids:
                task['dependencies'] = remediation_ids
        
        print(f"  ✅ Added dependencies")


if __name__ == "__main__":
    generator = TaskGenerator()
    
    phase_results = {'execution_time': 150}
    issues = ['Data inconsistency', 'Missing files']
    
    tasks = generator.generate_tasks(phase_results, 6.5, issues)
    
    print("\n\nGenerated Tasks:")
    for task in tasks:
        print(f"  [{task['priority']}] {task['title']}")
        print(f"    Type: {task['type']}, Time: {task['estimated_time']}min")
