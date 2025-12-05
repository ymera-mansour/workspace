# YMERA Refactoring Project
# Phase: 2E Enhanced | Agent: qoder | Created: 2024-12-05
# Multi-Model System Integration Example

"""
This example demonstrates how to use the intelligent agent-model matching
and multi-model execution system together.
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_1_coding_agent_simple():
    """Example 1: Simple code generation with automatic model selection"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Simple Code Generation")
    print("="*80)
    
    result = await executor.execute_with_multi_model(
        agent_name="coding_agent",
        task_description="Create a Python function to calculate fibonacci numbers",
        task_parameters={
            "language": "python",
            "include_docstring": True,
            "include_tests": True
        }
    )
    
    print(f"\nStrategy Used: {result.strategy_type}")
    print(f"Total Phases: {result.total_phases}")
    print(f"Successful Phases: {result.successful_phases}")
    print(f"Total Execution Time: {result.total_execution_time:.2f}s")
    print(f"Total Tokens Used: {result.total_tokens_used}")
    print(f"Models Used: {', '.join(result.models_used)}")
    
    print("\n--- Phase Results ---")
    for phase_result in result.phase_results:
        status = "✅" if phase_result.success else "❌"
        print(f"{status} {phase_result.phase.upper()}: "
              f"{phase_result.provider_used}:{phase_result.model_used} "
              f"({phase_result.execution_time:.2f}s)")
    
    print("\n--- Final Output ---")
    print(result.final_result)


async def example_2_coding_agent_complex():
    """Example 2: Complex code generation with multiple phases"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Complex Code Generation (Multi-Phase)")
    print("="*80)
    
    result = await executor.execute_with_multi_model(
        agent_name="coding_agent",
        task_description="""Create a comprehensive FastAPI REST API with:
        - User authentication (JWT)
        - CRUD operations for products
        - Database integration (SQLAlchemy)
        - Input validation (Pydantic)
        - Error handling
        - API documentation
        """,
        task_parameters={
            "language": "python",
            "framework": "fastapi",
            "database": "postgresql",
            "authentication": "jwt"
        }
    )
    
    print(f"\nStrategy: {result.strategy_type}")
    print(f"Phases: {result.total_phases}")
    print(f"Success Rate: {result.successful_phases}/{result.total_phases}")
    
    print("\n--- Execution Flow ---")
    for i, phase_result in enumerate(result.phase_results, 1):
        status = "✅" if phase_result.success else "❌"
        print(f"{i}. {status} {phase_result.phase.upper()}")
        print(f"   Model: {phase_result.provider_used}:{phase_result.model_used}")
        print(f"   Time: {phase_result.execution_time:.2f}s")
        print(f"   Tokens: {phase_result.tokens_used}")
        if not phase_result.success:
            print(f"   Error: {phase_result.error}")
    
    if result.final_result:
        print("\n--- Final Code Generated ---")
        final = result.final_result
        if isinstance(final, dict) and 'final_output' in final:
            print(final['final_output'])
        else:
            print(final)


async def example_3_database_agent():
    """Example 3: Database operations with optimized model selection"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Database Operations")
    print("="*80)
    
    result = await executor.execute_with_multi_model(
        agent_name="database_agent",
        task_description="Create SQL queries for e-commerce analytics dashboard",
        task_parameters={
            "database_type": "postgresql",
            "requirements": [
                "Total sales by month",
                "Top 10 products by revenue",
                "Customer retention rate",
                "Average order value",
                "Inventory turnover rate"
            ]
        }
    )
    
    print(f"\nStrategy: {result.strategy_type}")
    print(f"Models: {', '.join(result.models_used)}")
    print(f"Execution Time: {result.total_execution_time:.2f}s")
    
    print("\n--- Generated Queries ---")
    print(result.final_result)


async def example_4_analysis_agent():
    """Example 4: Data analysis with high-quality models"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 4: Data Analysis")
    print("="*80)
    
    result = await executor.execute_with_multi_model(
        agent_name="analysis_agent",
        task_description="""Analyze the following sales data and provide insights:
        Q1 Sales: $1.2M (↑15% YoY)
        Q2 Sales: $1.5M (↑22% YoY)
        Q3 Sales: $1.1M (↓8% YoY)
        Q4 Sales: $1.8M (↑30% YoY)
        
        Provide:
        1. Trend analysis
        2. Seasonal patterns
        3. Growth projections
        4. Recommendations
        """,
        task_parameters={
            "analysis_type": "business_intelligence",
            "include_visualizations": True,
            "confidence_level": 0.95
        }
    )
    
    print(f"\nMulti-Model Analysis Phases:")
    for phase_result in result.phase_results:
        if phase_result.success:
            print(f"✅ {phase_result.phase}: {phase_result.model_used}")
    
    print("\n--- Analysis Report ---")
    print(result.final_result)


async def example_5_custom_phase_selection():
    """Example 5: Custom phase selection for specific workflows"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Phase Selection")
    print("="*80)
    
    # Only run planning and generation phases (skip review/refinement)
    result = await executor.execute_with_multi_model(
        agent_name="coding_agent",
        task_description="Create a simple REST API endpoint",
        task_parameters={"language": "python"},
        enable_phases=["planning", "generation"]  # Only these phases
    )
    
    print(f"\nEnabled Phases: planning, generation")
    print(f"Executed Phases: {len(result.phase_results)}")
    print(f"Total Time: {result.total_execution_time:.2f}s")
    
    for phase_result in result.phase_results:
        print(f"- {phase_result.phase}: {phase_result.model_used} "
              f"({phase_result.execution_time:.2f}s)")


async def example_6_model_matching_inspection():
    """Example 6: Inspect model matching strategy without execution"""
    
    from core_services.ai_mcp.agent_model_matcher import get_agent_model_matcher
    
    matcher = get_agent_model_matcher()
    
    print("\n" + "="*80)
    print("EXAMPLE 6: Model Matching Strategy Inspection")
    print("="*80)
    
    # Get strategy without executing
    strategy = await matcher.match_agent_to_models(
        agent_name="coding_agent",
        task_description="Create a complex microservices architecture",
        task_parameters={"complexity": "high"}
    )
    
    print(f"\nStrategy Type: {strategy['strategy_type']}")
    print(f"Agent: {strategy['agent_name']}")
    
    if strategy['strategy_type'] == 'multi_model':
        print(f"\nPlanned Phases: {len(strategy['phases'])}")
        print("\n--- Model Selection per Phase ---")
        for phase in strategy['phases']:
            print(f"\nPhase: {phase['phase'].upper()}")
            print(f"Primary Model: {phase['primary_model']['provider']}:"
                  f"{phase['primary_model']['model']}")
            print(f"Score: {phase['primary_model']['score']}")
            print(f"Reason: {phase['reason']}")
            print(f"Fallback Models: {len(phase['fallback_models'])}")
    else:
        print(f"\nSingle Model: {strategy['model']['provider']}:"
              f"{strategy['model']['model']}")
        print(f"Score: {strategy['model']['score']}")


async def example_7_error_handling():
    """Example 7: Error handling and fallback mechanisms"""
    
    from core_services.ai_mcp.multi_model_executor import get_multi_model_executor
    
    executor = get_multi_model_executor()
    
    print("\n" + "="*80)
    print("EXAMPLE 7: Error Handling and Fallbacks")
    print("="*80)
    
    # Simulate a task that might fail in some phases
    result = await executor.execute_with_multi_model(
        agent_name="coding_agent",
        task_description="Create an extremely complex quantum computing algorithm",
        task_parameters={"complexity": "impossible"}
    )
    
    print(f"\nTotal Phases Attempted: {result.total_phases}")
    print(f"Successful Phases: {result.successful_phases}")
    print(f"Failed Phases: {result.total_phases - result.successful_phases}")
    
    print("\n--- Phase-by-Phase Status ---")
    for phase_result in result.phase_results:
        if phase_result.success:
            print(f"✅ {phase_result.phase}: SUCCESS")
        else:
            print(f"❌ {phase_result.phase}: FAILED - {phase_result.error}")
    
    if result.final_result:
        print("\n✅ Final result generated despite some failures")
    else:
        print("\n❌ Task could not be completed")


async def main():
    """Run all examples"""
    
    print("\n" + "="*80)
    print("YMERA MULTI-MODEL EXECUTION SYSTEM - EXAMPLES")
    print("="*80)
    
    examples = [
        ("Simple Code Generation", example_1_coding_agent_simple),
        ("Complex Multi-Phase Generation", example_2_coding_agent_complex),
        ("Database Operations", example_3_database_agent),
        ("Data Analysis", example_4_analysis_agent),
        ("Custom Phase Selection", example_5_custom_phase_selection),
        ("Model Matching Inspection", example_6_model_matching_inspection),
        ("Error Handling", example_7_error_handling),
    ]
    
    for i, (name, example_func) in enumerate(examples, 1):
        try:
            print(f"\n\n{'='*80}")
            print(f"Running Example {i}: {name}")
            print(f"{'='*80}")
            await example_func()
        except Exception as e:
            logger.error(f"Example {i} failed: {e}", exc_info=True)
            print(f"\n❌ Example {i} failed: {e}")
    
    print("\n\n" + "="*80)
    print("ALL EXAMPLES COMPLETED")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
