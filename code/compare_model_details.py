#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deep Model Comparison: Architecture, Size, and Feature Analysis
Compares all-mpnet-base-v2 (baseline) vs intfloat/e5-large-v2 (best performer)
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sentence_transformers import SentenceTransformer


def get_model_details(model_name: str) -> Dict[str, Any]:
    """
    Extract comprehensive model details including:
    - Parameter count (total, trainable, non-trainable)
    - Architecture details (layers, hidden size, attention heads)
    - Tokenizer vocab size and max length
    - Model memory footprint
    - Embedding dimension
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}\n")
    
    details = {"model_name": model_name}
    
    try:
        # Load config
        config = AutoConfig.from_pretrained(model_name)
        
        # Architecture details
        details["architecture"] = {
            "model_type": config.model_type,
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "intermediate_size": getattr(config, "intermediate_size", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "hidden_dropout_prob": getattr(config, "hidden_dropout_prob", None),
            "attention_probs_dropout_prob": getattr(config, "attention_probs_dropout_prob", None),
        }
        
        # Load actual model to count parameters
        print("Loading model to count parameters...")
        model = AutoModel.from_pretrained(model_name)
        
        # Parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params
        
        details["parameters"] = {
            "total": int(total_params),
            "total_millions": round(total_params / 1_000_000, 2),
            "trainable": int(trainable_params),
            "trainable_millions": round(trainable_params / 1_000_000, 2),
            "non_trainable": int(non_trainable_params),
            "non_trainable_millions": round(non_trainable_params / 1_000_000, 2),
        }
        
        # Memory footprint (approximate)
        param_bytes = total_params * 4  # 4 bytes per float32
        details["memory"] = {
            "parameters_mb": round(param_bytes / (1024**2), 2),
            "parameters_gb": round(param_bytes / (1024**3), 3),
        }
        
        # Tokenizer details
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        details["tokenizer"] = {
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": tokenizer.model_max_length,
            "special_tokens": {
                "cls_token": tokenizer.cls_token,
                "sep_token": tokenizer.sep_token,
                "pad_token": tokenizer.pad_token,
                "mask_token": getattr(tokenizer, "mask_token", None),
            }
        }
        
        # Embedding dimension (output)
        sample_input = tokenizer("test", return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            output = model(**sample_input)
            # Get pooled output or last hidden state
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                embedding_dim = output.pooler_output.shape[-1]
            else:
                embedding_dim = output.last_hidden_state.shape[-1]
        
        details["embedding_dim"] = int(embedding_dim)
        
        # Training details from config
        details["training_info"] = {
            "initializer_range": getattr(config, "initializer_range", None),
            "layer_norm_eps": getattr(config, "layer_norm_eps", None),
        }
        
        print(f"✓ Analysis complete!")
        print(f"  - Total parameters: {details['parameters']['total_millions']}M")
        print(f"  - Memory footprint: {details['memory']['parameters_mb']:.0f}MB")
        print(f"  - Embedding dimension: {details['embedding_dim']}")
        print(f"  - Max sequence length: {details['tokenizer']['model_max_length']}")
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"✗ Error: {e}")
        details["error"] = str(e)
    
    return details


def compare_models(model1_details: Dict, model2_details: Dict) -> Dict[str, Any]:
    """
    Compare two models and compute differences.
    """
    comparison = {
        "model1": model1_details["model_name"],
        "model2": model2_details["model_name"],
    }
    
    # Parameter comparison
    if "parameters" in model1_details and "parameters" in model2_details:
        m1_params = model1_details["parameters"]["total"]
        m2_params = model2_details["parameters"]["total"]
        
        comparison["parameters"] = {
            "model1_total": f"{model1_details['parameters']['total_millions']}M",
            "model2_total": f"{model2_details['parameters']['total_millions']}M",
            "difference_millions": round((m2_params - m1_params) / 1_000_000, 2),
            "ratio": round(m2_params / m1_params, 2),
            "model2_is_larger_by": f"{round((m2_params / m1_params - 1) * 100, 1)}%",
        }
    
    # Memory comparison
    if "memory" in model1_details and "memory" in model2_details:
        comparison["memory"] = {
            "model1_mb": model1_details["memory"]["parameters_mb"],
            "model2_mb": model2_details["memory"]["parameters_mb"],
            "difference_mb": round(model2_details["memory"]["parameters_mb"] - model1_details["memory"]["parameters_mb"], 2),
            "ratio": round(model2_details["memory"]["parameters_mb"] / model1_details["memory"]["parameters_mb"], 2),
        }
    
    # Architecture comparison
    if "architecture" in model1_details and "architecture" in model2_details:
        arch1 = model1_details["architecture"]
        arch2 = model2_details["architecture"]
        
        comparison["architecture"] = {
            "model1_layers": arch1.get("num_hidden_layers"),
            "model2_layers": arch2.get("num_hidden_layers"),
            "model1_hidden_size": arch1.get("hidden_size"),
            "model2_hidden_size": arch2.get("hidden_size"),
            "model1_attention_heads": arch1.get("num_attention_heads"),
            "model2_attention_heads": arch2.get("num_attention_heads"),
            "model1_intermediate_size": arch1.get("intermediate_size"),
            "model2_intermediate_size": arch2.get("intermediate_size"),
        }
        
        # Compute differences
        if arch1.get("num_hidden_layers") and arch2.get("num_hidden_layers"):
            comparison["architecture"]["layers_difference"] = arch2["num_hidden_layers"] - arch1["num_hidden_layers"]
        if arch1.get("hidden_size") and arch2.get("hidden_size"):
            comparison["architecture"]["hidden_size_difference"] = arch2["hidden_size"] - arch1["hidden_size"]
    
    # Embedding dimension comparison
    if "embedding_dim" in model1_details and "embedding_dim" in model2_details:
        comparison["embedding_dim"] = {
            "model1": model1_details["embedding_dim"],
            "model2": model2_details["embedding_dim"],
            "same": model1_details["embedding_dim"] == model2_details["embedding_dim"],
        }
    
    return comparison


def print_comparison_table(comparison: Dict[str, Any]):
    """
    Print a nicely formatted comparison table.
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    print(f"Model 1: {comparison['model1']}")
    print(f"Model 2: {comparison['model2']}")
    print()
    
    # Parameters
    if "parameters" in comparison:
        p = comparison["parameters"]
        print("PARAMETERS:")
        print(f"  Model 1: {p['model1_total']}")
        print(f"  Model 2: {p['model2_total']}")
        print(f"  Difference: +{p['difference_millions']}M ({p['model2_is_larger_by']} larger)")
        print(f"  Ratio: {p['ratio']}×")
        print()
    
    # Memory
    if "memory" in comparison:
        m = comparison["memory"]
        print("MEMORY FOOTPRINT:")
        print(f"  Model 1: {m['model1_mb']:.0f} MB")
        print(f"  Model 2: {m['model2_mb']:.0f} MB")
        print(f"  Difference: +{m['difference_mb']:.0f} MB")
        print(f"  Ratio: {m['ratio']:.2f}×")
        print()
    
    # Architecture
    if "architecture" in comparison:
        a = comparison["architecture"]
        print("ARCHITECTURE:")
        print(f"  Layers:          {a['model1_layers']} vs {a['model2_layers']} (diff: {a.get('layers_difference', 'N/A')})")
        print(f"  Hidden Size:     {a['model1_hidden_size']} vs {a['model2_hidden_size']} (diff: {a.get('hidden_size_difference', 'N/A')})")
        print(f"  Attention Heads: {a['model1_attention_heads']} vs {a['model2_attention_heads']}")
        print(f"  FF Size:         {a['model1_intermediate_size']} vs {a['model2_intermediate_size']}")
        print()
    
    # Embedding dimension
    if "embedding_dim" in comparison:
        e = comparison["embedding_dim"]
        print("EMBEDDING DIMENSION:")
        print(f"  Model 1: {e['model1']}")
        print(f"  Model 2: {e['model2']}")
        print(f"  Same: {e['same']}")
        print()


def generate_latex_table(details1: Dict, details2: Dict, comparison: Dict) -> str:
    """
    Generate a LaTeX table for the paper.
    """
    latex = []
    latex.append("\\begin{table}[t]")
    latex.append("\\centering")
    latex.append("\\caption{Architectural comparison between all-mpnet-base-v2 (baseline) and intfloat/e5-large-v2 (best performer).}")
    latex.append("\\label{tab:model_architecture}")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Property} & \\textbf{all-mpnet-base-v2} & \\textbf{e5-large-v2} \\\\")
    latex.append("\\midrule")
    
    # Parameters
    if "parameters" in comparison:
        latex.append(f"Parameters & {comparison['parameters']['model1_total']} & {comparison['parameters']['model2_total']} \\\\")
    
    # Memory
    if "memory" in comparison:
        latex.append(f"Memory (MB) & {comparison['memory']['model1_mb']:.0f} & {comparison['memory']['model2_mb']:.0f} \\\\")
    
    # Architecture
    if "architecture" in comparison:
        a = comparison["architecture"]
        latex.append(f"Layers & {a['model1_layers']} & {a['model2_layers']} \\\\")
        latex.append(f"Hidden Size & {a['model1_hidden_size']} & {a['model2_hidden_size']} \\\\")
        latex.append(f"Attention Heads & {a['model1_attention_heads']} & {a['model2_attention_heads']} \\\\")
        latex.append(f"Feed-Forward Size & {a['model1_intermediate_size']} & {a['model2_intermediate_size']} \\\\")
    
    # Embedding dimension
    if "embedding_dim" in comparison:
        latex.append(f"Output Dimension & {comparison['embedding_dim']['model1']} & {comparison['embedding_dim']['model2']} \\\\")
    
    # Performance (from your results)
    latex.append("\\midrule")
    latex.append("nDCG@10 & 0.744 & 0.760 \\\\")
    latex.append("Relative Gain & Baseline & +2.2\\% \\\\")
    
    # Efficiency
    latex.append("\\midrule")
    latex.append(f"Parameters Ratio & 1.0× & {comparison['parameters']['ratio']}× \\\\")
    latex.append(f"Memory Ratio & 1.0× & {comparison['memory']['ratio']:.2f}× \\\\")
    latex.append("Inference Time & 1.0× & 3.1× \\\\")  # From your batch times
    latex.append("Efficiency (nDCG/param) & 6.76 & 2.27 \\\\")  # 0.744/110M vs 0.760/335M
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    return "\n".join(latex)


def main():
    # Models to compare
    baseline = "sentence-transformers/all-mpnet-base-v2"
    best = "intfloat/e5-large-v2"
    
    # Get details for both models
    details1 = get_model_details(baseline)
    details2 = get_model_details(best)
    
    # Compare
    comparison = compare_models(details1, details2)
    
    # Print comparison
    print_comparison_table(comparison)
    
    # Generate LaTeX table
    latex_table = generate_latex_table(details1, details2, comparison)
    
    print("\n" + "="*80)
    print("LATEX TABLE (copy to paper)")
    print("="*80 + "\n")
    print(latex_table)
    
    # Save results
    results = {
        "baseline": details1,
        "best_performer": details2,
        "comparison": comparison,
        "latex_table": latex_table,
    }
    
    with open("model_comparison_details.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open("model_comparison_table.tex", "w") as f:
        f.write(latex_table)
    
    print("\n✓ Results saved to:")
    print("  - model_comparison_details.json")
    print("  - model_comparison_table.tex")


if __name__ == "__main__":
    main()
