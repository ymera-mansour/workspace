# Phase 1: Discovery - Multi-Layer Workflow

**Purpose**: Scan repository, classify files, build initial understanding

**Total Layers**: 5 Processing + 3 Validation = 8 layers  
**Timeline**: 3-5 hours (2-3 hours with parallelization)

---

## Overview

Phase 1 progressively scans and analyzes the entire repository through 5 processing layers, each adding deeper understanding. Three validation layers ensure quality before proceeding to Phase 2.

### Layer Summary

| Layer | Type | Models | Speed | Purpose |
|-------|------|--------|-------|---------|
| 1 | Processing | Ministral-3B, Gemini Flash-8B | Very Fast | Basic file scanning |
| 2 | Processing | Ministral-8B, Phi-3-mini | Fast | Initial classification |
| 3 | Processing | Qwen-3-32B, Mixtral-8x7B, Cohere | Moderate | Semantic analysis |
| 4 | Processing | Gemini 1.5 Pro, Qwen2-72B | Slow | Pattern recognition |
| 5 | Processing | Hermes-3-405B, DeepSeek-Chat-v3 | Slowest | Expert integration |
| V1 | Validation | Automated | Fast | Cross-validation |
| V2 | Validation | Mistral-Large | Moderate | Quality checks |
| V3 | Validation | Hermes-3-405B + Human | Slow | Expert review |

---

## Layer 1: Basic File Scanning

**Models**: Ministral-3B-latest, Gemini Flash-8B  
**Purpose**: Fast initial file enumeration  
**Speed**: Very Fast (<1 min for 10,000 files)  
**Cost**: Very Low

### Tasks
- List all files in repository
- Detect file types (extension-based)
- Calculate file sizes
- Generate initial metadata
- Create basic inventory

### Implementation

```python
class Layer1BasicScanning:
    """Fast file enumeration and basic metadata"""
    
    def __init__(self):
        self.scanner_a = ChatMistral(model="ministral-3b-latest")
        self.scanner_b = ChatGoogleGenerativeAI(model="gemini-1.5-flash-8b")
    
    async def scan_repository(self, repo_path):
        """Fast enumeration of all files"""
        files = []
        
        for root, dirs, filenames in os.walk(repo_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in filenames:
                # Skip hidden files
                if filename.startswith('.'):
                    continue
                
                filepath = os.path.join(root, filename)
                file_info = {
                    'path': filepath,
                    'name': filename,
                    'extension': os.path.splitext(filename)[1],
                    'size': os.path.getsize(filepath),
                    'modified': os.path.getmtime(filepath),
                    'relative_path': os.path.relpath(filepath, repo_path)
                }
                files.append(file_info)
        
        return files
    
    async def generate_metadata(self, files):
        """Generate summary metadata"""
        metadata = {
            'total_files': len(files),
            'total_size': sum(f['size'] for f in files),
            'extensions': Counter(f['extension'] for f in files),
            'directories': len(set(os.path.dirname(f['path']) for f in files))
        }
        return metadata
```

### Output Example

```python
{
    'files': [
        {
            'path': '/workspace/src/main.py',
            'name': 'main.py',
            'extension': '.py',
            'size': 15234,
            'modified': 1702140800,
            'relative_path': 'src/main.py'
        },
        # ... more files
    ],
    'metadata': {
        'total_files': 10542,
        'total_size': 524288000,
        'extensions': {'.py': 1250, '.md': 85, '.yaml': 12, ...},
        'directories': 156
    }
}
```

---

## Layer 2: Initial Classification

**Models**: Ministral-8b-latest, Phi-3-mini-128k  
**Purpose**: Quick file classification  
**Speed**: Fast (5-10 min for 10,000 files)  
**Cost**: Low

### Tasks
- Classify files by type (code, doc, config, data, test)
- Identify programming languages
- Detect framework/library usage
- Mark potential duplicates
- Extract file purpose

### Implementation

```python
class Layer2Classification:
    """Classify files by type and purpose"""
    
    def __init__(self):
        self.classifier_a = ChatMistral(model="ministral-8b-latest")
        self.classifier_b = ChatOpenRouter(model="microsoft/phi-3-mini-128k")
        self.cache = {}
    
    async def classify_files(self, files):
        """Classify each file"""
        classified = []
        
        for file in files:
            # Read file sample
            content_sample = await self.read_file_sample(file['path'], max_lines=50)
            
            # Classify using AI
            classification = await self.classify_file(file, content_sample)
            
            file['classification'] = classification
            classified.append(file)
        
        return classified
    
    async def classify_file(self, file, content_sample):
        """Classify a single file"""
        # Quick classification based on extension
        extension = file['extension'].lower()
        
        quick_classification = {
            '.py': 'code', '.js': 'code', '.ts': 'code', '.java': 'code',
            '.md': 'documentation', '.txt': 'documentation',
            '.yaml': 'config', '.yml': 'config', '.json': 'config',
            '.csv': 'data', '.xlsx': 'data',
            '_test.py': 'test', 'test_*.py': 'test'
        }
        
        file_type = quick_classification.get(extension, 'unknown')
        
        # For unknown or code files, use AI classification
        if file_type in ['unknown', 'code']:
            ai_classification = await self.classifier_a.classify({
                'filename': file['name'],
                'extension': extension,
                'content_sample': content_sample,
                'size': file['size']
            })
            return ai_classification
        
        return {
            'type': file_type,
            'language': self.detect_language(extension),
            'purpose': 'inferred',
            'framework': None
        }
    
    async def read_file_sample(self, filepath, max_lines=50):
        """Read first N lines of file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                lines = [f.readline() for _ in range(max_lines)]
                return ''.join(lines)
        except Exception as e:
            return ""
```

### Output Example

```python
{
    'path': '/workspace/src/main.py',
    'name': 'main.py',
    'classification': {
        'type': 'code',
        'language': 'python',
        'purpose': 'entry_point',
        'framework': 'fastapi',
        'dependencies': ['fastapi', 'uvicorn', 'pydantic']
    }
}
```

---

## Layer 3: Semantic Analysis

**Models**: Qwen-3-32b, Mixtral-8x7B-Instruct, Cohere embed-v3.0  
**Purpose**: Deeper semantic understanding  
**Speed**: Moderate (15-30 min for 10,000 files)  
**Cost**: Moderate

### Tasks
- Extract key concepts from files
- Generate embeddings for semantic search
- Identify relationships between files
- Build initial dependency graph
- Detect code patterns

### Implementation

```python
class Layer3SemanticAnalysis:
    """Semantic analysis and embedding generation"""
    
    def __init__(self):
        self.analyzer_a = ChatGroq(model="qwen/qwen3-32b")
        self.analyzer_b = ChatTogether(model="mistralai/Mixtral-8x7B-Instruct")
        self.embeddings = CohereEmbeddings(model="embed-english-v3.0")
        self.vectorstore = FAISS.from_documents([], self.embeddings)
    
    async def analyze_semantics(self, classified_files):
        """Extract concepts and relationships"""
        analyzed = []
        documents = []
        
        for file in classified_files:
            if file['classification']['type'] not in ['code', 'documentation']:
                analyzed.append(file)
                continue
            
            # Read full content
            content = await self.read_file_content(file['path'])
            
            # Extract key concepts
            concepts = await self.analyzer_a.extract_concepts({
                'filename': file['name'],
                'type': file['classification']['type'],
                'content': content
            })
            
            # Analyze relationships
            relationships = await self.find_relationships(file, classified_files)
            
            # Create document for embedding
            doc_content = f"{file['relative_path']}\n{concepts['summary']}\n{concepts['key_concepts']}"
            doc = Document(
                page_content=doc_content,
                metadata={
                    'file': file['path'],
                    'type': file['classification']['type'],
                    'concepts': concepts['key_concepts'],
                    'importance': concepts.get('importance', 0.5)
                }
            )
            documents.append(doc)
            
            # Add to file info
            file['concepts'] = concepts
            file['relationships'] = relationships
            analyzed.append(file)
        
        # Build vector index
        if documents:
            self.vectorstore.add_documents(documents)
        
        return analyzed, self.vectorstore
    
    async def find_relationships(self, file, all_files):
        """Identify relationships with other files"""
        relationships = {
            'imports': [],
            'imported_by': [],
            'similar_to': [],
            'depends_on': []
        }
        
        # Parse imports (for code files)
        if file['classification']['type'] == 'code':
            content = await self.read_file_content(file['path'])
            imports = self.parse_imports(content, file['classification']['language'])
            relationships['imports'] = imports
        
        return relationships
    
    def parse_imports(self, content, language):
        """Parse import statements"""
        imports = []
        
        if language == 'python':
            # Simple regex for Python imports
            import_pattern = r'^\s*(?:from\s+(\S+)\s+)?import\s+(.+)$'
            for line in content.split('\n'):
                match = re.match(import_pattern, line)
                if match:
                    module = match.group(1) or match.group(2).split(',')[0].strip()
                    imports.append(module)
        
        return imports
```

### Output Example

```python
{
    'path': '/workspace/src/main.py',
    'concepts': {
        'summary': 'Main entry point for the application, sets up FastAPI server',
        'key_concepts': ['fastapi', 'api server', 'entry point', 'initialization'],
        'importance': 0.9,
        'complexity': 'medium'
    },
    'relationships': {
        'imports': ['fastapi', 'uvicorn', 'src.routes', 'src.database'],
        'imported_by': [],
        'similar_to': [],
        'depends_on': ['src/routes.py', 'src/database.py']
    }
}
```

---

## Layer 4: Advanced Pattern Recognition

**Models**: Gemini 1.5 Pro, Qwen2-72B-Instruct  
**Purpose**: Complex pattern detection  
**Speed**: Slow (30-60 min for 10,000 files)  
**Cost**: Higher

### Tasks
- Detect design patterns
- Identify code smells
- Find architectural patterns
- Detect duplicate/similar code
- Analyze code quality

### Implementation

```python
class Layer4PatternRecognition:
    """Detect complex patterns and anti-patterns"""
    
    def __init__(self):
        self.pattern_expert_a = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        self.pattern_expert_b = ChatTogether(model="Qwen/Qwen2-72B-Instruct")
    
    async def recognize_patterns(self, analyzed_files, vectorstore):
        """Detect complex patterns"""
        patterns = {
            'design_patterns': [],
            'code_smells': [],
            'architectural_patterns': [],
            'duplicates': [],
            'quality_issues': []
        }
        
        # Group files by type
        code_files = [f for f in analyzed_files 
                     if f['classification']['type'] == 'code']
        
        # Analyze code files
        for file in code_files:
            # Detect design patterns
            design = await self.detect_design_patterns(file)
            if design:
                patterns['design_patterns'].extend(design)
            
            # Find code smells
            smells = await self.detect_code_smells(file)
            if smells:
                patterns['code_smells'].extend(smells)
            
            # Find duplicates using vector search
            similar = await self.find_similar_files(file, vectorstore)
            if similar:
                patterns['duplicates'].append({
                    'file': file['path'],
                    'similar_to': similar
                })
        
        # Detect architectural patterns
        arch_patterns = await self.detect_architectural_patterns(code_files)
        patterns['architectural_patterns'] = arch_patterns
        
        return patterns
    
    async def detect_design_patterns(self, file):
        """Detect design patterns in code"""
        content = await self.read_file_content(file['path'])
        
        result = await self.pattern_expert_a.analyze({
            'task': 'detect_design_patterns',
            'filename': file['name'],
            'content': content[:5000],  # First 5000 chars
            'language': file['classification']['language']
        })
        
        return result.get('patterns', [])
    
    async def detect_code_smells(self, file):
        """Detect code smells"""
        content = await self.read_file_content(file['path'])
        
        result = await self.pattern_expert_b.analyze({
            'task': 'detect_code_smells',
            'filename': file['name'],
            'content': content[:5000],
            'language': file['classification']['language']
        })
        
        return result.get('smells', [])
    
    async def find_similar_files(self, file, vectorstore):
        """Find similar files using vector search"""
        query = f"{file['concepts']['summary']} {file['concepts']['key_concepts']}"
        similar = vectorstore.similarity_search(query, k=5)
        
        # Filter out the file itself
        similar_files = [s.metadata['file'] for s in similar 
                        if s.metadata['file'] != file['path']]
        
        return similar_files[:3]  # Top 3 similar files
```

### Output Example

```python
{
    'design_patterns': [
        {'file': '/workspace/src/factory.py', 'pattern': 'Factory Pattern', 'confidence': 0.9},
        {'file': '/workspace/src/singleton.py', 'pattern': 'Singleton Pattern', 'confidence': 0.85}
    ],
    'code_smells': [
        {'file': '/workspace/src/utils.py', 'smell': 'Long Method', 'line': 45, 'severity': 'medium'},
        {'file': '/workspace/src/legacy.py', 'smell': 'Duplicate Code', 'line': 120, 'severity': 'high'}
    ],
    'duplicates': [
        {'file': '/workspace/src/handler1.py', 'similar_to': ['/workspace/src/handler2.py', '/workspace/src/handler3.py']},
    ],
    'architectural_patterns': [
        {'pattern': 'MVC', 'confidence': 0.7, 'files': ['/workspace/src/models/', '/workspace/src/views/', '/workspace/src/controllers/']}
    ]
}
```

---

## Layer 5: Expert Knowledge Integration

**Models**: Hermes-3-Llama-3.1-405B, DeepSeek-Chat-v3  
**Purpose**: Expert-level comprehensive analysis  
**Speed**: Slowest (1-2 hours for 10,000 files)  
**Cost**: Highest

### Tasks
- Strategic architecture review
- Security vulnerability assessment
- Performance bottleneck identification
- Consolidation strategy recommendations
- Expert insights and guidance

### Implementation

```python
class Layer5ExpertIntegration:
    """Expert-level comprehensive analysis"""
    
    def __init__(self):
        self.expert_a = ChatOpenRouter(model="nousresearch/hermes-3-405b")
        self.expert_b = ChatOpenRouter(model="deepseek/deepseek-chat-v3")
    
    async def expert_analysis(self, all_layers_output):
        """Expert-level comprehensive analysis"""
        # Synthesize all previous layers
        synthesis = {
            'files': all_layers_output['layer1_basic_scan'],
            'classifications': all_layers_output['layer2_classifications'],
            'semantics': all_layers_output['layer3_semantics'],
            'patterns': all_layers_output['layer4_patterns']
        }
        
        # Expert strategic review
        strategy = await self.expert_strategic_review(synthesis)
        
        # Security analysis
        security = await self.expert_security_analysis(synthesis)
        
        # Performance analysis
        performance = await self.expert_performance_analysis(synthesis)
        
        # Consolidation recommendations
        recommendations = await self.expert_consolidation_strategy(
            synthesis, strategy, security, performance
        )
        
        return {
            'strategy': strategy,
            'security': security,
            'performance': performance,
            'recommendations': recommendations
        }
    
    async def expert_strategic_review(self, synthesis):
        """Strategic architecture review"""
        return await self.expert_a.analyze({
            'task': 'strategic_architecture_review',
            'total_files': len(synthesis['files']),
            'file_types': synthesis['classifications'],
            'patterns': synthesis['patterns'],
            'context': 'platform_consolidation'
        })
    
    async def expert_security_analysis(self, synthesis):
        """Security vulnerability assessment"""
        return await self.expert_b.analyze({
            'task': 'security_vulnerability_assessment',
            'code_files': [f for f in synthesis['files'] 
                          if f['classification']['type'] == 'code'],
            'patterns': synthesis['patterns']['code_smells'],
            'context': 'identify_security_issues'
        })
```

### Output Example

```python
{
    'strategy': {
        'architecture_assessment': 'The system follows a microservices architecture...',
        'strengths': ['Clear separation of concerns', 'Good modularity'],
        'weaknesses': ['Duplicate code in handlers', 'Inconsistent naming'],
        'recommendations': ['Consolidate duplicate handlers', 'Standardize naming conventions']
    },
    'security': {
        'vulnerabilities': [
            {'type': 'SQL Injection', 'file': '/workspace/src/db.py', 'severity': 'high'},
            {'type': 'Hardcoded Secrets', 'file': '/workspace/src/config.py', 'severity': 'critical'}
        ],
        'recommendations': ['Use parameterized queries', 'Move secrets to environment variables']
    },
    'performance': {
        'bottlenecks': [
            {'location': '/workspace/src/processor.py:45', 'issue': 'N+1 query problem'},
            {'location': '/workspace/src/utils.py:120', 'issue': 'Inefficient loop'}
        ],
        'recommendations': ['Add query optimization', 'Use list comprehension']
    },
    'recommendations': {
        'priority_1': ['Fix critical security vulnerabilities', 'Consolidate duplicate handlers'],
        'priority_2': ['Optimize database queries', 'Standardize naming'],
        'priority_3': ['Improve documentation', 'Add type hints']
    }
}
```

---

## Validation Layers

### Validation Layer 1: Cross-Validation

**Purpose**: Validate data consistency across all processing layers

```python
class DiscoveryValidation1:
    """Cross-validate layer outputs"""
    
    async def cross_validate(self, layers_output):
        """Cross-validate between layers"""
        issues = []
        
        # Check: All files from Layer 1 are present in all layers
        layer1_files = set(f['path'] for f in layers_output['layer1'])
        layer2_files = set(f['path'] for f in layers_output['layer2'])
        layer3_files = set(f['path'] for f in layers_output['layer3'])
        
        if layer1_files != layer2_files != layer3_files:
            issues.append("File count mismatch between layers")
        
        # Check: All files have classifications
        for file in layers_output['layer2']:
            if not file.get('classification'):
                issues.append(f"Missing classification: {file['path']}")
        
        # Check: Semantic analysis completed
        for file in layers_output['layer3']:
            if file['classification']['type'] == 'code' and not file.get('concepts'):
                issues.append(f"Missing semantic analysis: {file['path']}")
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'warnings': self.generate_warnings(layers_output)
        }
```

### Validation Layer 2: Quality Checks

**Models**: Mistral-Large-latest  
**Purpose**: AI-powered quality validation

```python
class DiscoveryValidation2:
    """AI validation of analysis quality"""
    
    def __init__(self):
        self.validator = ChatMistral(model="mistral-large-latest")
    
    async def quality_check(self, layers_output):
        """AI validation of quality"""
        quality_report = await self.validator.validate_quality({
            'completeness': self.check_completeness(layers_output),
            'accuracy': self.check_accuracy(layers_output),
            'consistency': self.check_consistency(layers_output),
            'depth': self.check_analysis_depth(layers_output)
        })
        
        return quality_report
    
    def check_completeness(self, layers_output):
        """Check if analysis is complete"""
        total_files = len(layers_output['layer1'])
        classified = len([f for f in layers_output['layer2'] 
                         if f.get('classification')])
        analyzed = len([f for f in layers_output['layer3'] 
                       if f.get('concepts')])
        
        return {
            'classified_percentage': (classified / total_files) * 100,
            'analyzed_percentage': (analyzed / total_files) * 100,
            'complete': classified == analyzed == total_files
        }
```

### Validation Layer 3: Expert Review + Human Approval

**Models**: Hermes-3-405B  
**Purpose**: Final expert validation before Phase 2

```python
class DiscoveryValidation3:
    """Expert review and human approval"""
    
    def __init__(self):
        self.expert = ChatOpenRouter(model="nousresearch/hermes-3-405b")
    
    async def expert_review(self, layers_output, validation_reports):
        """Expert review of all discovery results"""
        expert_report = await self.expert.comprehensive_review({
            'discovery_results': layers_output,
            'validation_results': validation_reports,
            'context': 'phase_1_discovery_completion'
        })
        
        # Request human approval
        human_approved = await self.request_human_approval(expert_report)
        
        return {
            'expert_report': expert_report,
            'human_approved': human_approved,
            'ready_for_phase_2': human_approved and expert_report['quality_score'] >= 8.0
        }
    
    async def request_human_approval(self, expert_report):
        """Request human approval checkpoint"""
        print("\n" + "="*80)
        print("PHASE 1 DISCOVERY - EXPERT REVIEW")
        print("="*80)
        print(f"\nQuality Score: {expert_report['quality_score']}/10")
        print(f"Files Discovered: {expert_report['total_files']}")
        print(f"Files Classified: {expert_report['classified_files']}")
        print(f"Files Analyzed: {expert_report['analyzed_files']}")
        print(f"\nExpert Assessment: {expert_report['assessment']}")
        print(f"\nRecommendations: {', '.join(expert_report['recommendations'][:3])}")
        print("\n" + "="*80)
        
        approval = input("\nProceed to Phase 2? (yes/no): ")
        return approval.lower() in ['yes', 'y']
```

---

## Summary

### Phase 1 Statistics

- **Total Layers**: 8 (5 processing + 3 validation)
- **Timeline**: 3-5 hours (2-3 hours parallelized)
- **Models Used**: 11 different models
- **Output**: Complete repository analysis

### Layer Progression

1. **Layer 1**: Basic scan → File inventory
2. **Layer 2**: Classification → Typed files
3. **Layer 3**: Semantic analysis → Concepts & relationships
4. **Layer 4**: Pattern detection → Design patterns & duplicates
5. **Layer 5**: Expert analysis → Strategic recommendations
6. **Validation 1-3**: Quality assurance → Ready for Phase 2

### Key Outputs

- Complete file inventory with metadata
- File classifications (code, doc, config, data, test)
- Semantic embeddings and vector index
- Design patterns and code smells detected
- Duplicate file identification
- Security vulnerabilities identified
- Performance bottlenecks noted
- Strategic consolidation recommendations

### Next Phase

[Phase 2: Analysis](./PHASE2_ANALYSIS_LAYERS.md) - Analyze discovered files and plan consolidation

---

**Document Status**: Complete ✅  
**Last Updated**: 2025-12-09  
**Related**: [Multi-Layer Workflow Overview](./MULTI_LAYER_WORKFLOW_OVERVIEW.md)
