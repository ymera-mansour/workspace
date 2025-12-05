# Performance Benchmarking Plan for the Unified YMERA Agent System

This plan outlines the steps, tools, and methodology required to perform comprehensive performance benchmarking on the newly refactored YMERA agent system. The goal is to establish baseline performance metrics, identify bottlenecks, and ensure the system can handle expected production load.

## 1. Tool Selection and Rationale

Based on the system's Python-based microservices architecture (FastAPI/API Gateway), the following tools are recommended:

| Category | Recommended Tool | Rationale |
| :--- | :--- | :--- |
| **Load Generation** | **Locust** | Python-based, allowing test scripts to be written in the same language as the application. Excellent for API load testing and easily integrates with the existing Python environment and `shared/` utilities. |
| **Monitoring & Profiling** | **Prometheus & Grafana** | Standard open-source stack for time-series monitoring. Prometheus will scrape metrics (CPU, memory, request counts) from the `core_services` and `unified_agents` components, and Grafana will visualize the results in real-time. |
| **Code Profiling** | **cProfile/py-spy** | Essential for deep-diving into the Python code during a load test to pinpoint exact functions causing high latency or CPU usage. |

## 2. Benchmarking Methodology and Metrics

The benchmarking process will combine standard load testing with agent-specific functional metrics.

### Key Performance Indicators (KPIs)

| Metric | Description | Target Component |
| :--- | :--- | :--- |
| **Latency (Response Time)** | Time taken for the API Gateway to return a final response (P95 and P99). | `unified_agents` API Gateway |
| **Throughput (RPS)** | Number of successful requests processed per second. | `core_services` & `unified_agents` |
| **Agent Success Rate** | Percentage of requests that result in a successful agent outcome (i.e., not just a successful HTTP 200, but a successful task completion). | `core_services` (Agent Manager) |
| **Resource Utilization** | CPU and Memory usage of the `core_services` and `unified_agents` containers/processes. | All components (via Prometheus) |
| **Time-to-Completion (TTC)** | Total time from task submission to final result for long-running, asynchronous tasks. | `core_services` (Engines) |

### Test Scenarios (Locust Test Cases)

The tests should focus on the most critical and resource-intensive end-to-end (E2E) flows identified in the `master_test_plan.md`.

1.  **Scenario A: High-Volume, Low-Latency (e.g., Status Check)**: A simple, fast API call that hits the Agent Manager for a status update. Used to measure raw throughput and API overhead.
2.  **Scenario B: Standard Agent Execution (e.g., Code Generation)**: A typical synchronous agent request that involves orchestration, one engine call, and a final response. Used to measure average latency under load.
3.  **Scenario C: Stress Agent Execution (e.g., Deep Research)**: A long-running, asynchronous request that involves multiple engine calls and external AI-MCP interaction. Used to measure Time-to-Completion and resource saturation.

## 3. Phased Execution Plan

The benchmarking will be executed in four distinct phases to systematically identify performance limits.

### Phase 1: Smoke Test and Baseline Establishment

*   **Objective:** Verify the testing environment is correctly configured and establish a baseline for a minimal load.
*   **Load:** 10 concurrent users for 5 minutes.
*   **Action:** Run **Scenario A** and **Scenario B**.
*   **Outcome:** Confirm all test cases run without errors, and collect initial metrics (Latency, Throughput) to serve as the "ideal" baseline.

### Phase 2: Load Test (Expected Production Peak)

*   **Objective:** Determine if the system can handle the expected peak production load (e.g., 80% of capacity).
*   **Load:** Gradually ramp up to the expected peak number of concurrent users (e.g., 100 users) over 15 minutes.
*   **Action:** Run **Scenario B** and **Scenario C**.
*   **Outcome:** Measure Latency and Throughput stability. Identify the point where P95 latency begins to degrade significantly.

### Phase 3: Stress Test (Capacity Limit)

*   **Objective:** Find the system's breaking point and identify the bottleneck (CPU, Memory, Database, or AI-MCP rate limits).
*   **Load:** Continuously increase the number of concurrent users until the system's throughput plateaus or the error rate exceeds 5%.
*   **Action:** Run **Scenario C** (most resource-intensive).
*   **Outcome:** Determine the maximum sustainable throughput (RPS) and the resource that saturates first (e.g., "CPU hits 100% at 250 RPS").

### Phase 4: Bottleneck Analysis and Optimization

*   **Objective:** Use profiling tools to pinpoint the exact code location causing the bottleneck identified in Phase 3.
*   **Action:** Rerun the stress test (Phase 3) while simultaneously running **cProfile** or **py-spy** on the saturated `core_services` process.
*   **Outcome:** Identify the top 5 most time-consuming functions. Optimize these functions and repeat the stress test to confirm performance improvement.

## 4. Reporting

The final report will be a comparison of the baseline (Phase 1) versus the optimized peak performance (Phase 4). It will include:

1.  **Summary of Test Results:** Maximum sustainable throughput and the P95 latency at that load.
2.  **Resource Graphs:** Grafana screenshots showing CPU/Memory utilization during the stress test.
3.  **Bottleneck Findings:** Detailed report from the code profiler showing the functions that were optimized.
4.  **Recommendations:** Suggestions for further scaling (e.g., horizontal scaling of the `core_services` or optimizing database queries).
