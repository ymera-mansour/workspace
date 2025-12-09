"""
Usage Analytics Dashboard - Prometheus + Grafana Integration
Real-time metrics and monitoring for AI model usage
"""

from prometheus_client import Counter, Histogram, Gauge, start_http_server, Info
from typing import Dict, Any, Optional
import time


class AnalyticsDashboard:
    """
    Analytics dashboard with Prometheus metrics
    
    Metrics tracked:
    - Request count by provider/model
    - Response latency
    - Token usage
    - Cost tracking
    - Error rates
    - Cache hit rates
    - Circuit breaker states
    
    Grafana Integration:
    - Pre-built dashboards
    - Real-time visualization
    - Alerting rules
    """
    
    def __init__(self, port: int = 8000):
        self.port = port
        
        # Define metrics
        self.request_counter = Counter(
            'ai_model_requests_total',
            'Total AI model requests',
            ['provider', 'model', 'status']
        )
        
        self.latency_histogram = Histogram(
            'ai_model_latency_seconds',
            'AI model response latency',
            ['provider', 'model']
        )
        
        self.tokens_counter = Counter(
            'ai_model_tokens_total',
            'Total tokens used',
            ['provider', 'model', 'type']  # type: input/output
        )
        
        self.cost_counter = Counter(
            'ai_model_cost_dollars',
            'Total cost in dollars',
            ['provider', 'model']
        )
        
        self.cache_hit_rate = Gauge(
            'ai_cache_hit_rate',
            'Cache hit rate percentage'
        )
        
        self.circuit_breaker_state = Gauge(
            'circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=open, 2=half-open)',
            ['name']
        )
        
        self.active_requests = Gauge(
            'ai_model_active_requests',
            'Number of active requests',
            ['provider', 'model']
        )
        
        self.system_info = Info(
            'ai_system',
            'AI system information'
        )
        
        # Set system info
        self.system_info.info({
            'version': '1.0.0',
            'providers': '9',
            'models': '53',
            'cost': '$0/month'
        })
    
    def start(self):
        """Start Prometheus metrics server"""
        start_http_server(self.port)
        print(f"📊 Analytics dashboard started on http://localhost:{self.port}/metrics")
        print(f"📈 Grafana dashboard: http://localhost:3000 (if Grafana is running)")
    
    def record_request(
        self,
        provider: str,
        model: str,
        latency_seconds: float,
        tokens_input: int,
        tokens_output: int,
        cost: float,
        status: str = "success"
    ):
        """Record a model request"""
        self.request_counter.labels(
            provider=provider,
            model=model,
            status=status
        ).inc()
        
        self.latency_histogram.labels(
            provider=provider,
            model=model
        ).observe(latency_seconds)
        
        self.tokens_counter.labels(
            provider=provider,
            model=model,
            type="input"
        ).inc(tokens_input)
        
        self.tokens_counter.labels(
            provider=provider,
            model=model,
            type="output"
        ).inc(tokens_output)
        
        self.cost_counter.labels(
            provider=provider,
            model=model
        ).inc(cost)
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Update cache hit rate"""
        self.cache_hit_rate.set(hit_rate * 100)
    
    def update_circuit_breaker_state(self, name: str, state: str):
        """Update circuit breaker state"""
        state_map = {'closed': 0, 'open': 1, 'half_open': 2}
        self.circuit_breaker_state.labels(name=name).set(
            state_map.get(state, 0)
        )
    
    def track_active_request(self, provider: str, model: str, increment: bool = True):
        """Track active requests"""
        if increment:
            self.active_requests.labels(provider=provider, model=model).inc()
        else:
            self.active_requests.labels(provider=provider, model=model).dec()


# Example Grafana dashboard JSON (save as grafana_dashboard.json)
GRAFANA_DASHBOARD_JSON = """
{
  "dashboard": {
    "title": "AI Model Analytics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [
          {
            "expr": "rate(ai_model_requests_total[5m])",
            "legendFormat": "{{provider}} - {{model}}"
          }
        ]
      },
      {
        "title": "Response Latency",
        "targets": [
          {
            "expr": "ai_model_latency_seconds",
            "legendFormat": "{{provider}} - {{model}}"
          }
        ]
      },
      {
        "title": "Token Usage",
        "targets": [
          {
            "expr": "rate(ai_model_tokens_total[5m])",
            "legendFormat": "{{type}}"
          }
        ]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [
          {
            "expr": "ai_cache_hit_rate",
            "legendFormat": "Hit Rate %"
          }
        ]
      }
    ]
  }
}
"""


if __name__ == "__main__":
    # Start dashboard
    dashboard = AnalyticsDashboard(port=8000)
    dashboard.start()
    
    # Simulate some metrics
    dashboard.record_request(
        provider="gemini",
        model="gemini-1.5-flash",
        latency_seconds=0.5,
        tokens_input=100,
        tokens_output=200,
        cost=0.001
    )
    
    dashboard.update_cache_hit_rate(0.75)
    
    print("\nMetrics are being collected...")
    print("Visit http://localhost:8000/metrics to see Prometheus metrics")
    print("\nPress Ctrl+C to stop")
    
    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped")
