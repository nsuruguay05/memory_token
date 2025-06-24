import argparse
import os
import sys
import pandas as pd
from tqdm import tqdm
from transformers import set_seed
from llm_memory import LLMWithMemory


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM with memory on text corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--output_name", 
        type=str, 
        default="memories",
        help="Name for the output file"
    )
    
    parser.add_argument(
        "--model_id", 
        type=str, 
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="HuggingFace model identifier"
    )
    
    parser.add_argument(
        "--corpus", 
        type=str, 
        default="datasets/raschka_chunks_100.csv",
        help="Path to the corpus CSV file"
    )
    
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=20,
        help="Number of samples to evaluate (use -1 for all)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1234,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3000,
        help="Number of training epochs for memory"
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5.0,
        help="Learning rate for memory training"
    )
    
    parser.add_argument(
        "--template", 
        type=str, 
        default="{memory_token}{text}",
        help="Template for memory formatting"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=".",
        help="Directory to save output files"
    )
    
    return parser.parse_args()


def validate_args(args):
    """Validate command line arguments."""
    # Check if corpus file exists
    if not os.path.exists(args.corpus):
        raise FileNotFoundError(f"Corpus file not found: {args.corpus}")
    
    # Check if output directory exists, create if not
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")
    
    # Validate numeric arguments
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    
    if args.lr <= 0:
        raise ValueError("learning rate must be positive")
    
    if args.seed < 0:
        raise ValueError("seed must be non-negative")


def load_corpus(corpus_path):
    """Load and validate corpus data."""
    try:
        df = pd.read_csv(corpus_path)
        if "chunk" not in df.columns:
            raise ValueError("Corpus CSV must contain a 'chunk' column")
        texts = df["chunk"].to_list()
        print(f"Loaded {len(texts)} texts from {corpus_path}")
        return texts
    except Exception as e:
        raise RuntimeError(f"Failed to load corpus: {e}")


def evaluate_memories(llm_memory, texts, output_path, args):
    """Evaluate memories on the provided texts."""
    # Determine how many samples to process
    num_samples = len(texts) if args.num_samples == -1 else min(args.num_samples, len(texts))
    texts_to_process = texts[:num_samples]
    
    print(f"Evaluating on {num_samples} samples...")
    
    accuracies = []
    description = f"lr={args.lr}, seed={args.seed}, epochs={args.epochs}"
    
    for idx, text in (pbar := tqdm(enumerate(texts_to_process), total=len(texts_to_process))):
        try:
            # Add memory with training
            llm_memory.add_memory(
                text=text, 
                template=args.template, 
                description=description,
                epochs=args.epochs,
                lr=args.lr,
            )
            
            # Evaluate the memory
            accuracy = llm_memory.evaluate()
            accuracies.append(accuracy)
            
            # Update progress bar
            pbar.set_description(f"Accuracy: {accuracies[-1]:.4f}")
            
            # Save memories after each addition
            llm_memory.save_memories(output_path)
            
        except Exception as e:
            print(f"\nError processing text {idx}: {e}")
            accuracies.append(0.0)  # Record failure
            continue
    
    return accuracies


def report_results(accuracies, output_path, args):
    """Report evaluation results."""
    if not accuracies:
        print("No successful evaluations to report.")
        return
    
    exact_reconstructions = sum(1 for acc in accuracies if acc == 1.0)
    avg_accuracy = sum(accuracies) / len(accuracies)
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model: {args.model_id}")
    print(f"Corpus: {args.corpus}")
    print(f"Samples processed: {len(accuracies)}")
    print(f"Exact reconstructions: {exact_reconstructions}/{len(accuracies)} ({exact_reconstructions/len(accuracies)*100:.1f}%)")
    print(f"Average accuracy: {avg_accuracy:.4f}")
    print(f"Memories saved to: {output_path}")
    print("=" * 50)


def main():
    """Main function for evaluation script."""
    try:
        # Parse and validate arguments
        args = parse_args()
        validate_args(args)
        
        # Set seed for reproducibility
        set_seed(args.seed)
        print(f"Set random seed to {args.seed}")
        
        # Load corpus
        texts = load_corpus(args.corpus)
        
        # Initialize LLM with memory
        print(f"Initializing LLM with model: {args.model_id}")
        llm_memory = LLMWithMemory(
            model_id=args.model_id,
        )

        output_path = os.path.join(args.output_dir, f"{args.output_name}.pkl")
        
        # Evaluate memories
        accuracies = evaluate_memories(llm_memory, texts, output_path, args)
        
        # Report results
        report_results(accuracies, output_path, args)
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()