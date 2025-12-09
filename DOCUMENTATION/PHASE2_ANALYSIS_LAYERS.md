# Phase 2: Analysis - Multi-Layer Workflow

**Purpose**: Analyze discovered files, identify consolidation opportunities, create strategic plan

**Total Layers**: 4 Processing + 2 Validation = 6 layers  
**Timeline**: 4-6 hours (2-3 hours with parallelization)

---

## Overview

Phase 2 analyzes the repository discovered in Phase 1, identifying duplicates, architectural patterns, and consolidation opportunities. It culminates in a strategic consolidation plan.

### Layer Summary

| Layer | Type | Models | Speed | Purpose |
|-------|------|--------|-------|---------|
| 1 | Processing | Ministral-8B, Phi-3-mini | Fast | Duplicate detection |
| 2 | Processing | Qwen2.5-Coder-32B, DeepSeek-Coder-33B | Moderate | Code similarity |
| 3 | Processing | Gemini 1.5 Pro, Qwen2-72B | Moderate | Architecture analysis |
| 4 | Processing | Hermes-3-405B, DeepSeek-Chat-v3 | Slow | Strategic planning |
| V1 | Validation | Automated | Fast | Strategy cross-check |
| V2 | Validation | Hermes-3-405B + Human | Moderate | Expert approval |

---

## Layer 1: Basic Duplicate Detection

**Models**: Ministral-8b-latest, Phi-3-mini-128k  
**Purpose**: Quick duplicate identification  
**Speed**: Fast (10-15 min)  
**Input**: Discovery Phase output

### Tasks
- Hash-based duplicate detection
- File size comparison  
- Name similarity analysis
- Initial duplicate candidates
- Group similar files

---

## Layer 2: Code Similarity Analysis

**Models**: Qwen2.5-Coder-32B-Instruct, DeepSeek-Coder-33B  
**Purpose**: Deep code similarity detection  
**Speed**: Moderate (30-45 min)

### Tasks
- Abstract Syntax Tree (AST) comparison
- Semantic code similarity
- Function-level duplicate detection
- Multi-version file identification
- Refactoring opportunity analysis

---

## Layer 3: Architecture Analysis

**Models**: Gemini 1.5 Pro, Qwen2-72B-Instruct  
**Purpose**: System architecture understanding  
**Speed**: Moderate (45-60 min)

### Tasks
- Component identification
- Dependency analysis
- Module boundary detection
- Architecture pattern recognition
- Refactoring opportunity identification

---

## Layer 4: Strategic Consolidation Planning

**Models**: Hermes-3-405B, DeepSeek-Chat-v3  
**Purpose**: Expert consolidation strategy  
**Speed**: Slow (1-2 hours)

### Tasks
- Prioritize consolidation tasks
- Risk assessment
- Migration path planning
- Resource estimation
- Strategic recommendations

---

## Validation Layers

### Validation Layer 1: Strategy Cross-Check
**Purpose**: Validate consolidation strategy consistency

### Validation Layer 2: Expert Approval + Human Review
**Models**: Hermes-3-405B  
**Purpose**: Final strategic validation

---

**Phase 2 Timeline**: 4-6 hours (2-3 hours parallelized)

**Next**: [Phase 3: Consolidation](./PHASE3_CONSOLIDATION_LAYERS.md)

---

**Document Status**: Complete ✅  
**Last Updated**: 2025-12-09
