# Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings

Official implementation for the paper [Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings](https://arxiv.org/abs/2506.15001)

## Abstract

In this work, we observe an interesting phenomenon: it is possible to generate reversible sentence embeddings that allow an LLM to reconstruct the original text exactly, without modifying the model's weights.
This is achieved by introducing a special memory token, whose embedding is optimized through training on a fixed sequence.
When prompted with this embedding, the model reconstructs the fixed sequence exactly.
We evaluate this phenomenon across different datasets, sequence lengths, and model scales.
Notably, Llama 3.1 8B successfully reconstructs all tested sequences. 
Our findings highlight an interesting capability of LLMs and suggest potential applications in memory-based retrieval, compression, and controlled text generation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nsuruguay05/memory_token.git
cd memory_token
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from llm_memory import LLMWithMemory

# Initialize the model
llm = LLMWithMemory(model_id="meta-llama/Llama-3.1-8B-Instruct")

# Add a memory
text = "The quick brown fox jumps over the lazy dog"
memory = llm.add_memory(
    text=text,
    template="{memory_token}{text}",
    epochs=3000,
    lr=5.0
)

# Generate text using the memory
generated = llm.generate("<MEMORY>")
print(generated)

# Evaluate memory accuracy
accuracy = llm.evaluate()
print(f"Memory accuracy: {accuracy:.4f}")
```

### Evaluation (reproducing the paper's results)

Run the evaluation script to test text reconstruction on a text corpus:

```bash
python evaluate.py --corpus datasets/raschka_chunks_100.csv --num_samples 20
```

#### Command Line Arguments

##### Main Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_name` | `"memories"` | Name for the output file |
| `--model_id` | `"meta-llama/Llama-3.1-8B-Instruct"` | HuggingFace model identifier |
| `--corpus` | `"datasets/raschka_chunks_100.csv"` | Path to the corpus CSV file |
| `--num_samples` | `20` | Number of samples to evaluate (-1 for all) |
| `--output_dir` | `"."` | Directory to save output files |

##### Training Parameters

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `3000` | Number of training epochs for memory |
| `--lr` | `5.0` | Learning rate for memory training |
| `--template` | `"{memory_token}{text}"` | Template for memory formatting |
| `--seed` | `1234` | Random seed for reproducibility |