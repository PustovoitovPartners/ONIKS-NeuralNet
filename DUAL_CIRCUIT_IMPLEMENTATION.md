# Dual-Circuit Decision-Making System Implementation

## Overview

Successfully implemented a complete dual-circuit decision-making system for optimal performance and quality balance in the ONIKS NeuralNet framework.

## Architecture Components

### Circuit 1: Fast Response Circuit (RouterAgent)
- **Model**: phi3:mini (3.8B parameters)
- **Timeout**: 15 seconds (aggressive)
- **Task**: Lightning-fast "SIMPLE" vs "COMPLEX" classification
- **Prompt Size**: ~50 words (ultra-lightweight)
- **Fallback**: Keyword-based classification

### Circuit 2: Deep Planning Circuit (PlannerAgent)  
- **Model**: llama3:8b (8B parameters)
- **Timeout**: 20 minutes (quality over speed)
- **Task**: Detailed hierarchical planning for complex goals
- **Maintains**: All existing quality and functionality

## Implementation Details

### 1. Enhanced OllamaClient (`oniks/llm/client.py`)
- Added multi-model support with `default_model` parameter
- Modified `invoke()` method to accept optional `model` parameter
- Maintains backward compatibility with existing code
- Added model validation and fallback logic

### 2. Updated RouterAgent (`oniks/agents/router_agent.py`)
- Added `routing_model` and `main_model` parameters
- Reduced timeout from 30s to 15s for aggressive fast classification
- Implemented ultra-lightweight 50-word prompts for phi3:mini
- Added multi-layer fallback system:
  1. phi3:mini LLM classification
  2. Keyword-based classification
  3. Default to hierarchical (safety first)
- Enhanced error handling and logging

### 3. Enhanced PlannerAgent (`oniks/agents/planner_agent.py`)
- Explicitly uses llama3:8b model for complex planning
- Maintains 20-minute timeout for quality assurance
- Preserves all existing comprehensive prompts and functionality

### 4. Updated Demo Script (`run_reasoning_test.py`)
- Dual-circuit configuration for RouterAgent and PlannerAgent
- Performance monitoring with execution time tracking
- Enhanced display of routing decisions and performance metrics
- Model availability checking for both phi3:mini and llama3:8b

## Performance Targets

- **Simple tasks**: <3 minutes (vs 15+ minutes previously, 83% improvement)
- **Complex tasks**: Unchanged quality (+15s overhead acceptable)  
- **Classification accuracy**: >80% with phi3:mini
- **Fallback reliability**: 100% (always defaults to hierarchical on failure)

## Installation Requirements

Before running the dual-circuit system, ensure both models are available:

```bash
# Install required models
ollama pull llama3:8b     # Main model for deep planning
ollama pull phi3:mini     # Fast model for routing

# Start Ollama service
ollama serve
```

## Error Handling Strategy

1. **RouterAgent timeout (15s)** → fallback to "COMPLEX" classification
2. **phi3:mini unavailable** → fallback to keyword-based classification
3. **Invalid classification response** → fallback to "COMPLEX" 
4. **Any uncertainty** → choose "COMPLEX" (safety first)

## Testing

The implementation includes comprehensive error handling and has been tested for:
- Basic imports and initialization
- Dual-circuit RouterAgent configuration
- Keyword fallback classification
- Multi-model OllamaClient support

## Expected Performance Improvement

- **Simple tasks**: 83% faster (15+ minutes → 2.5 minutes)
- **Complex tasks**: Maintain quality (add only 15s overhead)
- **System reliability**: No degradation
- **Resource efficiency**: Use lightweight model only when needed

## Files Modified

1. `/Users/danylohorlov/GitHub/ONIKS-NeuralNet/oniks/llm/client.py`
2. `/Users/danylohorlov/GitHub/ONIKS-NeuralNet/oniks/agents/router_agent.py`
3. `/Users/danylohorlov/GitHub/ONIKS-NeuralNet/oniks/agents/planner_agent.py`
4. `/Users/danylohorlov/GitHub/ONIKS-NeuralNet/run_reasoning_test.py`

## Usage

Run the demonstration to see the dual-circuit system in action:

```bash
python run_reasoning_test.py
```

The system will automatically:
1. Use phi3:mini for fast task classification
2. Route simple tasks directly to ReasoningAgent (fast path)
3. Route complex tasks through PlannerAgent (quality path)
4. Display performance metrics and routing decisions
5. Handle all error cases gracefully with fallbacks