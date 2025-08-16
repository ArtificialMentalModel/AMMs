# Artificial Mental Models via LLMs: Enhancing Communication in Medical Rehabilitation

## Project Overview

This research project explores the integration of artificial mental models (AMMs) and user models into adaptive AI systems to assist patients in healthcare decision-making.

### Key Applications
1. **Rehabilitation Support**: After knee/hip surgery
2. **Treatment Decision Assistance**: For patients with cognitive impairments (e.g., multiple sclerosis, dementia)

### Core Technology
The project develops a decision support system built from fine-tuned large language models that employs Artificial Mental Models (AMMs) to represent patient beliefs and decision-making processes.

### Models Used
The following models were utilized in this research:
- **LLaMA-3.1 8B**: 8-billion parameter version for efficient inference
- **LLaMA-3.1 70B**: 70-billion parameter version for high-accuracy predictions
- **Mistral**: 7B parameter model for balanced performance
- **Phi-3**: Microsoft's compact model for resource-constrained environments
- **GPT-4**: OpenAI's model for baseline comparisons and few-shot learning

## Repository Structure

```
├── finetuning/                    # Model fine-tuning scripts
│   ├── llm_finetune_script1.py   # LLM fine-tuning script 1
│   └── llm_finetune_script2.py   # LLM fine-tuning script 2
├── evaluation/                    # Model evaluation scripts
│   ├── accuracy_evaluation_all_exercises.py
│   ├── accuracy_evaluation_one_exercise.py
│   ├── sensitivity_specificity_evaluation.py
│   ├── regression_model_evaluation.py
│   ├── gradient_boost_evaluation.py
│   ├── tensorflow_evaluation.py
│   ├── synthetic_data_evaluation.py
│   └── baseline_model_evaluation.py
├── utils/                         # Utility scripts
│   ├── prompt_templates.py
│   ├── logits_extraction_script1.py
│   ├── logits_extraction_script2.py
│   ├── qa_interface_script.py
│   ├── iterative_prompt_qa.py
│   └── few_shot_prompts_script.py
├── requirements.txt
├── ENVIRONMENT_SETUP.md
└── README.md
```

## Scripts Description

### Fine-tuning Scripts

#### `llm_finetune_script1.py`
- Fine-tunes large language models for healthcare decision support
- Uses LoRA (Low-Rank Adaptation) for efficient training
- Processes synthetic patient data with Big 5 personality traits
- Outputs fine-tuned model for deployment

#### `llm_finetune_script2.py`
- Fine-tunes alternative large language models for similar healthcare applications
- Implements parameter-efficient fine-tuning techniques
- Optimized for resource-constrained environments

### Evaluation Scripts

#### `accuracy_evaluation_all_exercises.py`
- Comprehensive evaluation across all exercise types
- Calculates accuracy, sensitivity, and specificity metrics
- Handles multiple model outputs and formats
- Supports both zero-shot and few-shot evaluation

#### `accuracy_evaluation_one_exercise.py`
- Focused evaluation for single exercise types
- Streamlined metrics calculation
- Optimized for rapid prototyping

#### `sensitivity_specificity_evaluation.py`
- Binary classification evaluation (YES/NO predictions)
- Calculates confusion matrix metrics
- Handles missing predictions appropriately

#### `regression_model_evaluation.py`
- Evaluates regression-based prediction models
- Supports multiple regression algorithms
- Provides detailed performance metrics

#### `gradient_boost_evaluation.py`
- Gradient Boosting model evaluation
- Handles structured patient data
- Optimized for clinical decision support

#### `tensorflow_evaluation.py`
- TensorFlow-based model evaluation
- Neural network performance assessment
- Deep learning model metrics

#### `synthetic_data_evaluation.py`
- Evaluates models on synthetic patient data
- Tests model generalization capabilities
- Validates AMM representation accuracy

#### `baseline_model_evaluation.py`
- Baseline model evaluation for comparison
- Performance benchmarking across different models
- Standardized evaluation metrics

### Utility Scripts

#### `prompt_templates.py`
- Contains standardized prompt templates
- Big 5 personality trait integration
- Multi-language support (German/English)
- Structured output formatting

#### `logits_extraction_script1.py`
- Extracts logits from model outputs
- Few-shot learning support
- Probability distribution analysis

#### `logits_extraction_script2.py`
- Specialized for synthetic data processing
- Enhanced logit extraction for AMM training
- Batch processing capabilities

#### `qa_interface_script.py`
- Chat interface for Q&A interactions
- Real-time patient interaction simulation
- Response quality assessment

#### `iterative_prompt_qa.py`
- Iterative prompt refinement
- Dynamic question-answering system
- Adaptive response generation

#### `few_shot_prompts_script.py`
- Few-shot learning templates
- Comparative analysis support
- Baseline model evaluation

## Usage Instructions

### Prerequisites
```bash
pip install -r requirements.txt
```

### Environment Setup
See `ENVIRONMENT_SETUP.md` for detailed instructions on setting up API keys and tokens.

### Fine-tuning a Model
```bash
cd finetuning
python llm_finetune_script1.py
```

### Evaluating Model Performance
```bash
cd evaluation
python accuracy_evaluation_all_exercises.py
```

### Using Utility Functions
```python
from utils.prompt_templates import get_prompt_templates
from utils.logits_extraction_script1 import extract_logits

# Get prompt templates
prompts = get_prompt_templates()

# Extract model logits
logits = extract_logits(model_output)
```

## Key Features

### Artificial Mental Models (AMMs)
- Represent patient beliefs and decision-making processes
- Integrate Big 5 personality traits
- Provide personalized healthcare recommendations
- Bridge cognitive gaps in patient understanding

### Multi-Model Support
- LLaMA-3.1 8B, LLaMA-3.1 70B, Mistral, Phi-3, GPT-4 integration
- Comparative performance analysis
- Model ensemble capabilities
- Resource-optimized deployment

### Clinical Decision Support
- Rehabilitation exercise recommendations
- Treatment option guidance
- Patient-specific risk assessment
- Evidence-based decision making

### Data Processing
- Synthetic patient data generation
- Real-world clinical data integration
- Multi-language support (German/English)
- Privacy-preserving data handling

## Research Contributions

1. **AMM Development**: Novel approach to representing patient mental models
2. **Multi-Modal Integration**: Combining structured surveys with AI techniques
3. **Personalized Healthcare**: Individualized treatment recommendations
4. **Cognitive Gap Bridging**: Making complex medical information accessible

## Future Work

- Integration with electronic health records (EHR)
- Real-time patient monitoring
- Multi-center clinical validation
- Regulatory compliance and approval
