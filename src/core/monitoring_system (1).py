# ============================================================================
# PRODUCTION MONITORING & ALERTING SYSTEM
# Real-time monitoring, alerting, and observability
# ============================================================================

import asyncio
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import statistics

# ============================================================================
# 1. METRICS COLLECTOR
# ============================================================================

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class Metric:
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_prometheus(self) -> str:
        """Convert to Prometheus format"""
        labels_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        return f'{self.name}{{{labels_str}}} {self.value} {int(self.timestamp.timestamp() * 1000)}'

class MetricsCollector:
    """
    Collects and exposes metrics in Prometheus format
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = {}
        self.histograms: Dict[str, deque] = {}
        
    def counter(self, name: str, value: float = 1.0, **labels):
        """Increment a counter"""
        metric = Metric(
            name=name,
            type=MetricType.COUNTER,
            value=value,
            labels=labels
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        # Sum with existing counter
        existing = [m for m in self.metrics[name] if m.labels == labels]
        if existing:
            existing[0].value += value
        else:
            self.metrics[name].append(metric)
    
    def gauge(self, name: str, value: float, **labels):
        """Set a gauge value"""
        metric = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels
        )
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        # Replace existing gauge
        self.metrics[name] = [m for m in self.metrics[name] if m.labels != labels]
        self.metrics[name].append(metric)
    
    def histogram(self, name: str, value: float, **labels):
        """Record a histogram observation"""
        key = f"{name}_{json.dumps(labels, sort_keys=True)}"
        
        if key not in self.histograms:
            self.histograms[key] = deque(maxlen=1000)  # Keep last 1000
        
        self.histograms[key].append(value)
        
        # Calculate percentiles
        values = sorted(self.histograms[key])
        
        if values:
            p50 = values[len(values) // 2]
            p95 = values[int(len(values) * 0.95)]
            p99 = values[int(len(values) * 0.99)]
            
            self.gauge(f"{name}_p50", p50, **labels)
            self.gauge(f"{name}_p95", p95, **labels)
            self.gauge(f"{name}_p99", p99, **labels)
    
    def get_metrics(self) -> str:
        """Export all metrics in Prometheus format"""
        lines = []
        
        for name, metrics_list in self.metrics.items():
            for metric in metrics_list:
                lines.append(metric.to_prometheus())
        
        return "\n".join(lines)
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.histograms.clear()


# ============================================================================
# 2. REAL-TIME DASHBOARD DATA
# ============================================================================

@dataclass
class DashboardSnapshot:
    """Real-time dashboard data"""
    timestamp: datetime
    
    # Request metrics
    total_requests: int = 0
    requests_per_second: float = 0.0
    active_requests: int = 0
    
    # Performance metrics
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Success/Error metrics
    success_rate: float = 100.0
    error_count: int = 0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    
    # Cost metrics
    total_cost: float = 0.0
    cost_per_request: float = 0.0
    
    # Provider metrics
    provider_health: Dict[str, str] = field(default_factory=dict)
    provider_usage: Dict[str, int] = field(default_factory=dict)
    
    # Agent metrics
    agent_usage: Dict[str, int] = field(default_factory=dict)
    agent_success_rates: Dict[str, float] = field(default_factory=dict)

class DashboardCollector:
    """
    Collects data for real-time dashboard
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.snapshots: deque = deque(maxlen=1000)  # Last 1000 snapshots
        self.collection_interval = 5  # seconds
        self.running = False
        
    async def start_collection(self):
        """Start collecting dashboard data"""
        self.running = True
        
        while self.running:
            snapshot = await self._collect_snapshot()
            self.snapshots.append(snapshot)
            
            await asyncio.sleep(self.collection_interval)
    
    def stop_collection(self):
        """Stop collecting data"""
        self.running = False
    
    async def _collect_snapshot(self) -> DashboardSnapshot:
        """Collect current system state"""
        
        # Get base stats
        status = await self.orchestrator.get_system_status()
        
        # Calculate derived metrics
        llm_stats = status.get("llm_usage", {})
        cache_stats = status.get("cache", {})
        
        # Calculate RPS from recent snapshots
        rps = self._calculate_rps()
        
        snapshot = DashboardSnapshot(
            timestamp=datetime.now(),
            total_requests=llm_stats.get("total_calls", 0),
            requests_per_second=rps,
            avg_latency_ms=0.0,  # Would need to track
            p95_latency_ms=0.0,
            success_rate=100.0,
            cache_hit_rate=cache_stats.get("hit_rate", 0),
            total_cost=llm_stats.get("total_cost", 0),
            cost_per_request=llm_stats.get("avg_cost_per_call", 0),
            provider_usage=llm_stats.get("calls_by_provider", {}),
            agent_usage={}
        )
        
        return snapshot
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second from recent snapshots"""
        if len(self.snapshots) < 2:
            return 0.0
        
        recent = list(self.snapshots)[-10:]  # Last 10 snapshots
        
        if len(recent) < 2:
            return 0.0
        
        time_diff = (recent[-1].timestamp - recent[0].timestamp).seconds
        request_diff = recent[-1].total_requests - recent[0].total_requests
        
        return request_diff / max(time_diff, 1)
    
    def get_latest_snapshot(self) -> Optional[DashboardSnapshot]:
        """Get most recent snapshot"""
        return self.snapshots[-1] if self.snapshots else None
    
    def get_time_series(self, metric: str, duration_minutes: int = 60) -> List[tuple]:
        """Get time series data for a metric"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        
        data = []
        for snapshot in self.snapshots:
            if snapshot.timestamp >= cutoff:
                value = getattr(snapshot, metric, None)
                if value is not None:
                    data.append((snapshot.timestamp, value))
        
        return data


# ============================================================================
# 3. ALERTING SYSTEM
# ============================================================================

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "resolved": self.resolved
        }

@dataclass
class AlertRule:
    name: str
    condition: Callable[[DashboardSnapshot], bool]
    severity: AlertSeverity
    message_template: str
    cooldown_seconds: int = 300  # Don't alert more than once per 5 min
    last_alerted: Optional[datetime] = None

class AlertManager:
    """
    Manages alerts and notifications
    """
    
    def __init__(self):
        self.rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_history: deque = deque(maxlen=1000)
        self.notification_handlers: List[Callable] = []
        
        # Register default rules
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register default alert rules"""
        
        # High error rate
        self.rules.append(AlertRule(
            name="high_error_rate",
            condition=lambda s: s.success_rate < 95.0,
            severity=AlertSeverity.ERROR,
            message_template="Success rate dropped to {success_rate:.1f}%"
        ))
        
        # High latency
        self.rules.append(AlertRule(
            name="high_latency",
            condition=lambda s: s.p95_latency_ms > 5000,
            severity=AlertSeverity.WARNING,
            message_template="P95 latency is {p95_latency_ms:.0f}ms"
        ))
        
        # Cost overrun
        self.rules.append(AlertRule(
            name="cost_overrun",
            condition=lambda s: s.total_cost > 10.0,
            severity=AlertSeverity.CRITICAL,
            message_template="Total cost exceeded $10 (${total_cost:.2f})"
        ))
        
        # Low cache hit rate
        self.rules.append(AlertRule(
            name="low_cache_hit_rate",
            condition=lambda s: s.cache_hit_rate < 50.0 and s.total_requests > 100,
            severity=AlertSeverity.WARNING,
            message_template="Cache hit rate only {cache_hit_rate:.1f}%"
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        self.rules.append(rule)
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add notification handler (e.g., email, Slack)"""
        self.notification_handlers.append(handler)
    
    async def evaluate_rules(self, snapshot: DashboardSnapshot):
        """Evaluate all rules against current snapshot"""
        
        for rule in self.rules:
            # Check if in cooldown
            if rule.last_alerted:
                elapsed = (datetime.now() - rule.last_alerted).seconds
                if elapsed < rule.cooldown_seconds:
                    continue
            
            # Evaluate condition
            try:
                if rule.condition(snapshot):
                    await self._trigger_alert(rule, snapshot)
            except Exception as e:
                print(f"Error evaluating rule {rule.name}: {e}")
    
    async def _trigger_alert(self, rule: AlertRule, snapshot: DashboardSnapshot):
        """Trigger an alert"""
        
        # Format message
        message = rule.message_template.format(**snapshot.__dict__)
        
        alert = Alert(
            name=rule.name,
            severity=rule.severity,
            message=message,
            metadata={"snapshot": snapshot.__dict__}
        )
        
        # Add to active alerts
        self.active_alerts.append(alert)
        self.alert_history.append(alert)
        
        # Update last alerted time
        rule.last_alerted = datetime.now()
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Notification handler failed: {e}")
        
        print(f"🚨 ALERT [{alert.severity.value.upper()}] {alert.message}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return [a for a in self.active_alerts if not a.resolved]
    
    def resolve_alert(self, alert_name: str):
        """Mark alert as resolved"""
        for alert in self.active_alerts:
            if alert.name == alert_name and not alert.resolved:
                alert.resolved = True
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        active = self.get_active_alerts()
        
        return {
            "active_count": len(active),
            "by_severity": {
                "critical": len([a for a in active if a.severity == AlertSeverity.CRITICAL]),
                "error": len([a for a in active if a.severity == AlertSeverity.ERROR]),
                "warning": len([a for a in active if a.severity == AlertSeverity.WARNING]),
                "info": len([a for a in active if a.severity == AlertSeverity.INFO])
            },
            "recent_alerts": [a.to_dict() for a in list(self.alert_history)[-10:]]
        }


# ============================================================================
# 4. NOTIFICATION HANDLERS
# ============================================================================

class SlackNotifier:
    """Send alerts to Slack"""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    async def send(self, alert: Alert):
        """Send alert to Slack"""
        import aiohttp
        
        color_map = {
            AlertSeverity.CRITICAL: "#ff0000",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.INFO: "#0099ff"
        }
        
        payload = {
            "attachments": [{
                "color": color_map[alert.severity],
                "title": f"🚨 {alert.severity.value.upper()}: {alert.name}",
                "text": alert.message,
                "footer": "Agent Platform Monitor",
                "ts": int(alert.timestamp.timestamp())
            }]
        }
        
        async with aiohttp.ClientSession() as session:
            await session.post(self.webhook_url, json=payload)

class EmailNotifier:
    """Send alerts via email"""
    
    def __init__(self, smtp_config: Dict[str, Any]):
        self.smtp_config = smtp_config
    
    async def send(self, alert: Alert):
        """Send alert via email"""
        import smtplib
        from email.mime.text import MIMEText
        
        msg = MIMEText(f"""
Alert: {alert.name}
Severity: {alert.severity.value}
Time: {alert.timestamp}

{alert.message}

Metadata:
{json.dumps(alert.metadata, indent=2)}
        """)
        
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.name}"
        msg['From'] = self.smtp_config['from']
        msg['To'] = self.smtp_config['to']
        
        # Send email (async wrapper)
        await asyncio.to_thread(self._send_smtp, msg)
    
    def _send_smtp(self, msg):
        """Send via SMTP"""
        with smtplib.SMTP(self.smtp_config['host'], self.smtp_config['port']) as server:
            if self.smtp_config.get('use_tls'):
                server.starttls()
            if self.smtp_config.get('username'):
                server.login(
                    self.smtp_config['username'],
                    self.smtp_config['password']
                )
            server.send_message(msg)


# ============================================================================
# 5. WEB DASHBOARD API
# ============================================================================

class DashboardAPI:
    """
    FastAPI endpoints for dashboard
    """
    
    def __init__(self, dashboard_collector: DashboardCollector, 
                 alert_manager: AlertManager,
                 metrics_collector: MetricsCollector):
        self.dashboard = dashboard_collector
        self.alerts = alert_manager
        self.metrics = metrics_collector
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        snapshot = self.dashboard.get_latest_snapshot()
        
        if not snapshot:
            return {"error": "No data available"}
        
        return {
            "current": snapshot.__dict__,
            "alerts": self.alerts.get_alert_summary(),
            "charts": {
                "requests_per_second": self.dashboard.get_time_series("requests_per_second", 60),
                "latency": self.dashboard.get_time_series("avg_latency_ms", 60),
                "cost": self.dashboard.get_time_series("total_cost", 60),
                "cache_hit_rate": self.dashboard.get_time_series("cache_hit_rate", 60)
            }
        }
    
    def get_metrics_prometheus(self) -> str:
        """Get Prometheus metrics"""
        return self.metrics.get_metrics()


# ============================================================================
# 6. COMPLETE MONITORING SYSTEM
# ============================================================================

class MonitoringSystem:
    """
    Complete monitoring system orchestrator
    """
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        
        # Components
        self.metrics = MetricsCollector()
        self.dashboard = DashboardCollector(orchestrator)
        self.alerts = AlertManager()
        self.api = DashboardAPI(self.dashboard, self.alerts, self.metrics)
        
        self.running = False
    
    async def start(self):
        """Start monitoring system"""
        self.running = True
        
        print("🚀 Starting monitoring system...")
        
        # Start dashboard collection
        dashboard_task = asyncio.create_task(self.dashboard.start_collection())
        
        # Start alert evaluation loop
        alert_task = asyncio.create_task(self._alert_loop())
        
        print("✅ Monitoring system started")
        
        await asyncio.gather(dashboard_task, alert_task)
    
    async def _alert_loop(self):
        """Continuously evaluate alerts"""
        while self.running:
            snapshot = self.dashboard.get_latest_snapshot()
            
            if snapshot:
                await self.alerts.evaluate_rules(snapshot)
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    def stop(self):
        """Stop monitoring system"""
        self.running = False
        self.dashboard.stop_collection()
        print("✅ Monitoring system stopped")
    
    def add_slack_notifications(self, webhook_url: str):
        """Enable Slack notifications"""
        notifier = SlackNotifier(webhook_url)
        self.alerts.add_notification_handler(
            lambda alert: asyncio.create_task(notifier.send(alert))
        )
    
    def add_email_notifications(self, smtp_config: Dict[str, Any]):
        """Enable email notifications"""
        notifier = EmailNotifier(smtp_config)
        self.alerts.add_notification_handler(
            lambda alert: asyncio.create_task(notifier.send(alert))
        )


# ============================================================================
# 7. FASTAPI INTEGRATION
# ============================================================================

def add_monitoring_endpoints(app, monitoring_system: MonitoringSystem):
    """
    Add monitoring endpoints to FastAPI app
    """
    
    @app.get("/monitoring/dashboard")
    async def get_dashboard():
        """Get dashboard data"""
        return monitoring_system.api.get_dashboard_data()
    
    @app.get("/monitoring/metrics")
    async def get_metrics():
        """Get Prometheus metrics"""
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            monitoring_system.api.get_metrics_prometheus(),
            media_type="text/plain"
        )
    
    @app.get("/monitoring/alerts")
    async def get_alerts():
        """Get active alerts"""
        return {
            "active": [a.to_dict() for a in monitoring_system.alerts.get_active_alerts()],
            "summary": monitoring_system.alerts.get_alert_summary()
        }
    
    @app.post("/monitoring/alerts/{alert_name}/resolve")
    async def resolve_alert(alert_name: str):
        """Resolve an alert"""
        monitoring_system.alerts.resolve_alert(alert_name)
        return {"status": "resolved"}


# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

async def demo_monitoring():
    """Demonstrate monitoring system"""
    
    from agent_platform import ProductionOrchestrator
    
    # Initialize orchestrator
    orch = ProductionOrchestrator()
    await orch.initialize()
    
    # Initialize monitoring
    monitoring = MonitoringSystem(orch)
    
    # Add notifications (optional)
    # monitoring.add_slack_notifications("https://hooks.slack.com/...")
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitoring.start())
    
    # Simulate some requests
    print("\nSimulating requests...")
    for i in range(10):
        await orch.process_request(
            user_id="test_user",
            prompt=f"Test request {i}"
        )
        await asyncio.sleep(1)
    
    # Wait a bit for metrics
    await asyncio.sleep(15)
    
    # Get dashboard data
    dashboard_data = monitoring.api.get_dashboard_data()
    print("\n" + "="*60)
    print("DASHBOARD DATA")
    print("="*60)
    print(json.dumps(dashboard_data, indent=2, default=str))
    
    # Stop monitoring
    monitoring.stop()
    monitoring_task.cancel()
    
    await orch.close()

if __name__ == "__main__":
    asyncio.run(demo_monitoring())