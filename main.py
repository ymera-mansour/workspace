#!/usr/bin/env python3
"""
YMERA Platform - Main Entry Point
==================================

This is the main entry point for the YMERA multi-layer workflow platform.
It orchestrates the entire workflow from Phase 0 (Setup) through Phase 5 (Integration),
with Phase X inter-phase validation running between each phase.

Usage:
    python main.py --repo-path /path/to/workspace --phases all
    python main.py --repo-path /path/to/workspace --phases discovery,analysis
    python main.py --config config.yaml --enable-monitoring
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime

# Add project directories to path
sys.path.insert(0, str(Path(__file__).parent / "00-FOUNDATION"))
sys.path.insert(0, str(Path(__file__).parent / "ORCHESTRATION"))
sys.path.insert(0, str(Path(__file__).parent / "0X-VALIDATION"))
sys.path.insert(0, str(Path(__file__).parent / "OPTIMIZATIONS"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ymera_platform.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YMERAPlatform:
    """Main YMERA Platform orchestrator"""
    
    def __init__(self, config_path: str = "00-FOUNDATION/config.yaml"):
        """
        Initialize YMERA Platform
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
        self.workflow_orchestrator = None
        self.phase_x_validator = None
        self.state_manager = None
        self.monitoring_enabled = False
        
        logger.info("Initializing YMERA Platform...")
        
    async def initialize(self):
        """Initialize all platform components"""
        try:
            # Import components (will be created in subsequent steps)
            from config_loader import ConfigLoader
            from workflow_orchestrator import WorkflowOrchestrator
            from phase_x_validator import PhaseXValidator
            from state_manager import StateManager
            
            # Load configuration
            logger.info("Loading configuration...")
            config_loader = ConfigLoader(self.config_path)
            self.config = config_loader.load()
            
            # Validate configuration
            logger.info("Validating configuration...")
            from config_validator import ConfigValidator
            validator = ConfigValidator(self.config)
            if not validator.validate():
                raise ValueError("Configuration validation failed")
            
            # Run preflight checks
            logger.info("Running preflight checks...")
            from preflight_checker import PreflightChecker
            preflight = PreflightChecker(self.config)
            if not await preflight.check():
                raise RuntimeError("Preflight checks failed")
            
            # Initialize state manager
            logger.info("Initializing state manager...")
            self.state_manager = StateManager(
                state_dir=".ymera_state",
                config=self.config
            )
            await self.state_manager.initialize()
            
            # Initialize Phase X validator
            logger.info("Initializing Phase X validator...")
            self.phase_x_validator = PhaseXValidator(
                config=self.config,
                state_manager=self.state_manager
            )
            
            # Initialize workflow orchestrator
            logger.info("Initializing workflow orchestrator...")
            self.workflow_orchestrator = WorkflowOrchestrator(
                config=self.config,
                state_manager=self.state_manager,
                phase_x_validator=self.phase_x_validator
            )
            
            logger.info("✅ YMERA Platform initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize YMERA Platform: {e}")
            raise
    
    async def run_workflow(
        self,
        repo_path: str,
        phases: List[str] = None,
        enable_monitoring: bool = True,
        enable_phase_x: bool = True
    ):
        """
        Run the complete workflow
        
        Args:
            repo_path: Path to repository to process
            phases: List of phases to run (default: all)
            enable_monitoring: Enable silent monitoring
            enable_phase_x: Enable inter-phase validation
        """
        try:
            if phases is None or phases == ["all"]:
                phases = [
                    "Phase 0: Pre-Flight & Setup",
                    "Phase 1: Discovery",
                    "Phase 2: Analysis",
                    "Phase 3: Consolidation",
                    "Phase 4: Testing",
                    "Phase 5: Integration"
                ]
            
            logger.info(f"🚀 Starting YMERA workflow for: {repo_path}")
            logger.info(f"📋 Phases to execute: {', '.join(phases)}")
            logger.info(f"🔍 Monitoring enabled: {enable_monitoring}")
            logger.info(f"✅ Phase X validation enabled: {enable_phase_x}")
            
            # Start monitoring if enabled
            if enable_monitoring:
                logger.info("Starting silent monitoring system...")
                # Will be implemented with monitoring system
            
            # Execute workflow
            results = await self.workflow_orchestrator.execute_workflow(
                repo_path=repo_path,
                phases=phases,
                enable_phase_x=enable_phase_x
            )
            
            # Save final results
            await self.save_results(results)
            
            logger.info("✅ Workflow completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ Workflow execution failed: {e}")
            raise
    
    async def save_results(self, results: dict):
        """Save workflow results"""
        try:
            results_dir = Path(".ymera_results")
            results_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = results_dir / f"workflow_results_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📄 Results saved to: {results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            logger.info("Cleaning up resources...")
            if self.state_manager:
                await self.state_manager.close()
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YMERA Platform - Multi-Layer Workflow System"
    )
    parser.add_argument(
        "--repo-path",
        type=str,
        required=True,
        help="Path to repository to process"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="00-FOUNDATION/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="all",
        help="Comma-separated list of phases to run (default: all)"
    )
    parser.add_argument(
        "--enable-monitoring",
        action="store_true",
        default=True,
        help="Enable silent monitoring system"
    )
    parser.add_argument(
        "--disable-phase-x",
        action="store_true",
        help="Disable Phase X inter-phase validation"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from saved state"
    )
    
    args = parser.parse_args()
    
    # Parse phases
    if args.phases == "all":
        phases = ["all"]
    else:
        phases = [p.strip() for p in args.phases.split(",")]
    
    # Initialize platform
    platform = YMERAPlatform(config_path=args.config)
    
    try:
        # Initialize all components
        await platform.initialize()
        
        # Run workflow
        results = await platform.run_workflow(
            repo_path=args.repo_path,
            phases=phases,
            enable_monitoring=args.enable_monitoring,
            enable_phase_x=not args.disable_phase_x
        )
        
        # Print summary
        print("\n" + "="*80)
        print("🎉 YMERA Workflow Completed Successfully!")
        print("="*80)
        print(f"\nRepository: {args.repo_path}")
        print(f"Phases executed: {len(results.get('phases_completed', []))}")
        print(f"Total duration: {results.get('total_duration', 'N/A')}")
        print(f"Overall quality score: {results.get('overall_quality_score', 'N/A')}")
        print("\n" + "="*80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Workflow interrupted by user")
        return 130
        
    except Exception as e:
        logger.error(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        await platform.cleanup()


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        sys.exit(1)
