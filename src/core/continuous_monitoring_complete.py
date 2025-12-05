# YMERA Refactoring Project - Continuous Monitoring (Continuation)
# Part 2: Complete Monitoring & Orchestration

from typing import Dict, Any, List
import asyncio
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Continuation of ContinuousMonitor class...

    async def _check_security_status(self):
        """Check security event rates"""
        if len(self.security_events) == 0:
            return
        
        # Check events in last hour
        one_hour_ago = datetime.now().timestamp() - 3600
        recent_events = [
            e for e in self.security_events
            if e["timestamp"] > one_hour_ago
        ]
        
        if len(recent_events) > self.alert_thresholds["max_security_events_per_hour"]:
            await self._raise_alert(
                "high_security_events",
                f"{len(recent_events)} security events in last hour (threshold: {self.alert_thresholds['max_security_events_per_hour']})"
            )
    
    async def _check_performance(self):
        """Check system performance metrics"""
        from core_services.ai_mcp.monitoring.metrics_collector import get_metrics_collector
        
        metrics = get_metrics_collector()
        dashboard_data = metrics.get_dashboard_data()
        
        error_rate = dashboard_data["overview"]["error_rate"]
        if error_rate > self.alert_thresholds["max_error_rate"]:
            await self._raise_alert(
                "high_error_rate",
                f"Error rate ({error_rate:.1%}) exceeds threshold ({self.alert_thresholds['max_error_rate']:.1%})"
            )
    
    async def _generate_alerts(self):
        """Generate alerts based on monitoring data"""
        # Check for anomalies
        anomalies = await self._detect_anomalies()
        
        for anomaly in anomalies:
            await self._raise_alert(
                "anomaly_detected",
                anomaly["description"]
            )
    
    async def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in metrics"""
        anomalies = []
        
        if len(self.quality_history) < 20:
            return anomalies
        
        recent = self.quality_history[-20:]
        
        # Calculate moving average
        window = 10
        for i in range(len(recent) - window):
            window_data = recent[i:i+window]
            avg = sum(item["quality_score"] for item in window_data) / window
            
            # Check current vs average
            current = recent[i+window]["quality_score"]
            if abs(current - avg) > 0.3:  # 30% deviation
                anomalies.append({
                    "type": "quality_deviation",
                    "description": f"Quality score deviated by {abs(current - avg):.2f}",
                    "severity": "medium"
                })
        
        return anomalies
    
    async def _raise_alert(self, alert_type: str, message: str):
        """Raise an alert"""
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "message": message,
            "severity": self._get_alert_severity(alert_type)
        }
        
        logger.warning(f"ALERT: {alert_type} - {message}")
        
        # Save alert to file
        await self._save_alert(alert)
    
    def _get_alert_severity(self, alert_type: str) -> str:
        """Determine alert severity"""
        high_severity = ["high_security_events", "quality_degradation"]
        if alert_type in high_severity:
            return "high"
        return "medium"
    
    async def _save_alert(self, alert: Dict[str, Any]):
        """Save alert to file"""
        import os
        
        alert_dir = "_data/monitoring/alerts"
        os.makedirs(alert_dir, exist_ok=True)
        
        date_str = datetime.now().strftime("%Y-%m-%d")
        alert_file = os.path.join(alert_dir, f"alerts_{date_str}.jsonl")
        
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
    
    def record_quality_metric(
        self,
        task_id: str,
        quality_score: float,
        task_type: str
    ):
        """Record quality metric for monitoring"""
        self.quality_history.append({
            "timestamp": datetime.now().timestamp(),
            "task_id": task_id,
            "quality_score": quality_score,
            "task_type": task_type
        })
        
        # Keep only last 1000 records
        if len(self.quality_history) > 1000:
            self.quality_history = self.quality_history[-1000:]
    
    def record_security_event(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Record security event for monitoring"""
        self.security_events.append({
            "timestamp": datetime.now().timestamp(),
            "event_type": event_type,
            "severity": severity,
            "details": details
        })
        
        # Keep only last 1000 events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]
    
    def get_monitoring_report(self) -> Dict[str, Any]:
        """Get comprehensive monitoring report"""
        # Calculate statistics
        if self.quality_history:
            recent_quality = self.quality_history[-100:]
            avg_quality = sum(item["quality_score"] for item in recent_quality) / len(recent_quality)
            min_quality = min(item["quality_score"] for item in recent_quality)
            max_quality = max(item["quality_score"] for item in recent_quality)
        else:
            avg_quality = min_quality = max_quality = 0.0
        
        # Security statistics
        one_hour_ago = datetime.now().timestamp() - 3600
        recent_security = [
            e for e in self.security_events
            if e["timestamp"] > one_hour_ago
        ]
        
        return {
            "monitoring_status": "active" if self.is_monitoring else "inactive",
            "quality_metrics": {
                "average": avg_quality,
                "minimum": min_quality,
                "maximum": max_quality,
                "total_samples": len(self.quality_history)
            },
            "security_metrics": {
                "events_last_hour": len(recent_security),
                "total_events": len(self.security_events),
                "events_by_type": self._count_by_type(recent_security)
            },
            "alerts": {
                "active_alerts": self._get_active_alerts()
            }
        }
    
    def _count_by_type(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count events by type"""
        counts = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts
    
    def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts from last 24 hours"""
        import os
        
        alert_dir = "_data/monitoring/alerts"
        if not os.path.exists(alert_dir):
            return []
        
        # Read today's alerts
        date_str = datetime.now().strftime("%Y-%m-%d")
        alert_file = os.path.join(alert_dir, f"alerts_{date_str}.jsonl")
        
        if not os.path.exists(alert_file):
            return []
        
        alerts = []
        with open(alert_file, 'r') as f:
            for line in f:
                try:
                    alerts.append(json.loads(line))
                except:
                    pass
        
        return alerts

# ============================================================================
# WORKFLOW ORCHESTRATOR (Main Integration)
# ============================================================================

class WorkflowOrchestrator:
    """
    Main orchestrator integrating all systems
    
    Workflow:
    1. Validate workflow
    2. Distribute tasks
    3. Execute with quality monitoring
    4. Validate outcomes
    5. Continuous monitoring
    """
    
    def __init__(self):
        self.validator = WorkflowValidator()
        self.distributor = TaskDistributor()
        self.quality_assessor = QualityAssessor()
        self.outcome_validator = OutcomeValidator()
        self.monitor = ContinuousMonitor()
        
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def execute_workflow(
        self,
        workflow_id: str,
        tasks: List[Task],
        enable_monitoring: bool = True
    ) -> Dict[str, Any]:
        """
        Execute complete workflow with validation and monitoring
        
        Returns:
            Complete workflow result with all validation data
        """
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        # Start monitoring if enabled
        if enable_monitoring and not self.monitor.is_monitoring:
            await self.monitor.start_monitoring()
        
        try:
            # STEP 1: Validate workflow
            logger.info("Step 1: Validating workflow...")
            validation = await self.validator.validate_workflow(workflow_id, tasks)
            
            if not validation.valid:
                return {
                    "workflow_id": workflow_id,
                    "status": "failed",
                    "error": "Validation failed",
                    "validation": validation.__dict__,
                    "tasks_completed": 0
                }
            
            # STEP 2: Distribute tasks
            logger.info("Step 2: Distributing tasks...")
            distribution = await self.distributor.distribute_workflow(workflow_id, tasks)
            
            # STEP 3: Execute tasks
            logger.info("Step 3: Executing tasks...")
            execution_result = await self._execute_distributed_tasks(
                workflow_id,
                tasks,
                distribution
            )
            
            # STEP 4: Validate outcomes
            logger.info("Step 4: Validating outcomes...")
            validation_results = await self._validate_all_outcomes(
                execution_result["task_outcomes"]
            )
            
            # Calculate overall quality
            overall_quality = self._calculate_overall_quality(validation_results)
            
            # Record for monitoring
            if enable_monitoring:
                self.monitor.record_quality_metric(
                    workflow_id,
                    overall_quality,
                    "workflow"
                )
            
            return {
                "workflow_id": workflow_id,
                "status": "completed",
                "validation": validation.__dict__,
                "distribution": distribution,
                "execution": execution_result,
                "validation_results": validation_results,
                "overall_quality": overall_quality,
                "quality_level": self.quality_assessor.get_quality_level(overall_quality).value
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "workflow_id": workflow_id,
                "status": "failed",
                "error": str(e),
                "tasks_completed": 0
            }
    
    async def _execute_distributed_tasks(
        self,
        workflow_id: str,
        tasks: List[Task],
        distribution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute tasks according to distribution plan"""
        from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
        
        executor = get_multi_model_executor()
        task_map = {task.task_id: task for task in tasks}
        
        completed_tasks = {}
        task_outcomes = []
        
        # Execute in order (respecting dependencies)
        for task_id in distribution["execution_order"]:
            task = task_map[task_id]
            
            logger.info(f"Executing task {task_id}: {task.description[:50]}...")
            
            # Execute task
            start_time = datetime.now().timestamp()
            
            try:
                result = await executor.execute_with_multi_model(
                    agent_name=task.agent_name,
                    task_description=task.description,
                    task_parameters=task.parameters
                )
                
                execution_time = datetime.now().timestamp() - start_time
                
                # Validate outcome
                outcome = await self.outcome_validator.validate_outcome(
                    task=task,
                    result=result.final_result,
                    model_used=",".join(result.models_used),
                    execution_time=execution_time
                )
                
                task_outcomes.append(outcome)
                completed_tasks[task_id] = outcome
                
                # Record quality metric
                self.monitor.record_quality_metric(
                    task_id,
                    outcome.quality_metrics.overall_score,
                    task.task_type
                )
                
                # Check if outcome is acceptable
                if not outcome.validation_passed:
                    logger.warning(f"Task {task_id} failed validation")
                    
                    # Retry if possible
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        logger.info(f"Retrying task {task_id} (attempt {task.retry_count})")
                        # Add back to queue
                        continue
                
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                task.error = str(e)
        
        return {
            "tasks_attempted": len(distribution["execution_order"]),
            "tasks_completed": len(completed_tasks),
            "task_outcomes": task_outcomes
        }
    
    async def _validate_all_outcomes(
        self,
        outcomes: List[TaskOutcome]
    ) -> Dict[str, Any]:
        """Validate all task outcomes"""
        validation_summary = {
            "total_tasks": len(outcomes),
            "passed_validation": 0,
            "passed_security": 0,
            "quality_scores": [],
            "failed_tasks": []
        }
        
        for outcome in outcomes:
            if outcome.validation_passed:
                validation_summary["passed_validation"] += 1
            else:
                validation_summary["failed_tasks"].append({
                    "task_id": outcome.task_id,
                    "reason": "validation_failed",
                    "details": outcome.validation_details
                })
            
            if outcome.security_passed:
                validation_summary["passed_security"] += 1
            else:
                validation_summary["failed_tasks"].append({
                    "task_id": outcome.task_id,
                    "reason": "security_failed",
                    "details": outcome.security_details
                })
            
            validation_summary["quality_scores"].append(
                outcome.quality_metrics.overall_score
            )
        
        return validation_summary
    
    def _calculate_overall_quality(
        self,
        validation_results: Dict[str, Any]
    ) -> float:
        """Calculate overall workflow quality"""
        if not validation_results["quality_scores"]:
            return 0.0
        
        return sum(validation_results["quality_scores"]) / len(validation_results["quality_scores"])
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get current status of a workflow"""
        if workflow_id not in self.active_workflows:
            return {
                "workflow_id": workflow_id,
                "status": "not_found"
            }
        
        return self.active_workflows[workflow_id]
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        return {
            "system_status": {
                "monitoring_active": self.monitor.is_monitoring,
                "active_workflows": len(self.active_workflows)
            },
            "monitoring_report": self.monitor.get_monitoring_report(),
            "quality_benchmarks": self.quality_assessor.quality_benchmarks,
            "recent_alerts": self.monitor._get_active_alerts()
        }


# Singletons
_orchestrator_instance = None
_monitor_instance = None

def get_workflow_orchestrator() -> WorkflowOrchestrator:
    """Get singleton workflow orchestrator"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = WorkflowOrchestrator()
    return _orchestrator_instance

def get_continuous_monitor() -> ContinuousMonitor:
    """Get singleton continuous monitor"""
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = ContinuousMonitor()
    return _monitor_instance
