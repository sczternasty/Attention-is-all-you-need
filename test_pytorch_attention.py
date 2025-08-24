import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import gc
from wiki_self_attention import SelfAttention

torch.manual_seed(42)

class PyTorchAttentionWrapper(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        if mask is not None:
            mask = mask.squeeze(1) if mask.dim() == 4 else mask
        output, _ = self.attention(query, key, value, attn_mask=mask)
        return output

def generate_test_data(batch_size: int, seq_len: int, d_model: int, device: torch.device):
    X = torch.randn(batch_size, seq_len, d_model, device=device)
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return X, mask


def benchmark_attention_models(device: torch.device = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    print(f"Running benchmarks on device: {device}")
    
    configs = [
        {'batch_size': 8, 'seq_len': 128, 'd_model': 256, 'num_heads': 8},
        {'batch_size': 16, 'seq_len': 256, 'd_model': 512, 'num_heads': 8},
        {'batch_size': 32, 'seq_len': 512, 'd_model': 768, 'num_heads': 12},
    ]
    
    results = {}
    
    for config in configs:
        print(f"\nTesting configuration: {config}")

        X, mask = generate_test_data(
            config['batch_size'], config['seq_len'],
            config['d_model'], device
        )

        custom_attention = SelfAttention(
            emb_size=config['d_model'], 
            heads=config['num_heads']
        ).to(device)
        pytorch_attention = PyTorchAttentionWrapper(
            config['d_model'], config['num_heads']
        ).to(device)
        
        for _ in range(3):
            with torch.no_grad():
                _ = custom_attention(X)
                _ = pytorch_attention(X, X, X, mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                output_custom = custom_attention(query)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        custom_time = (time.time() - start_time) / 100
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(100):
            with torch.no_grad():
                output_pytorch = pytorch_attention(query, key, value, mask)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        pytorch_time = (time.time() - start_time) / 100
        
        if device.type == 'cuda':
            custom_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            torch.cuda.reset_peak_memory_stats(device)
            
            _ = pytorch_attention(query, key, value, mask)
            pytorch_memory = torch.cuda.max_memory_allocated(device) / 1024**2
            torch.cuda.reset_peak_memory_stats(device)
        else:
            custom_memory = pytorch_memory = 0
        
        similarity = F.cosine_similarity(
            output_custom.flatten(), output_pytorch.flatten(), dim=0
        ).item()
        
        config_key = f"b{config['batch_size']}_s{config['seq_len']}_d{config['d_model']}_h{config['num_heads']}"
        results[config_key] = {
            'custom_time': custom_time * 1000,
            'pytorch_time': pytorch_time * 1000,
            'custom_memory': custom_memory,
            'pytorch_memory': pytorch_memory,
            'similarity': similarity,
            'speedup': pytorch_time / custom_time if custom_time > 0 else 0
        }
        
        print(f"Custom: {custom_time*1000:.2f}ms, PyTorch: {pytorch_time*1000:.2f}ms")
        print(f"Speedup: {results[config_key]['speedup']:.2f}x")
        print(f"Output similarity: {similarity:.4f}")
        
        del custom_attention, pytorch_attention, query, key, value, mask
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return results

def plot_benchmark_results(results: Dict):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Custom vs PyTorch Attention: Performance Comparison', fontsize=16, fontweight='bold')
    
    configs = list(results.keys())
    custom_times = [results[config]['custom_time'] for config in configs]
    pytorch_times = [results[config]['pytorch_time'] for config in configs]
    speedups = [results[config]['speedup'] for config in configs]
    similarities = [results[config]['similarity'] for config in configs]
    
    x = np.arange(len(configs))
    width = 0.35
    
    ax1.bar(x - width/2, custom_times, width, label='Custom Implementation', alpha=0.8)
    ax1.bar(x + width/2, pytorch_times, width, label='PyTorch Built-in', alpha=0.8)
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Execution Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels([config.replace('_', '\n') for config in configs], rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.bar(configs, speedups, alpha=0.8, color='green')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('PyTorch Speedup over Custom Implementation')
    ax2.set_xticklabels([config.replace('_', '\n') for config in configs], rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline (1x)')
    ax2.legend()
    
    ax3.bar(configs, similarities, alpha=0.8, color='orange')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_title('Output Quality Comparison')
    ax3.set_xticklabels([config.replace('_', '\n') for config in configs], rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Quality Threshold (0.95)')
    ax3.legend()
    
    custom_memory = [results[config]['custom_memory'] for config in configs]
    pytorch_memory = [results[config]['pytorch_memory'] for config in configs]
    
    if any(m > 0 for m in custom_memory + pytorch_memory):
        ax4.bar(x - width/2, custom_memory, width, label='Custom Implementation', alpha=0.8)
        ax4.bar(x + width/2, pytorch_memory, width, label='PyTorch Built-in', alpha=0.8)
        ax4.set_xlabel('Configuration')
        ax4.set_ylabel('Memory Usage (MB)')
        ax4.set_title('Memory Usage Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels([config.replace('_', '\n') for config in configs], rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Memory metrics not available\nfor CPU execution', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Memory Usage Comparison')
    
    plt.tight_layout()
    plt.savefig('attention_benchmark_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def run_comprehensive_tests():
    print("Starting Comprehensive Attention Model Benchmarking")
    print("=" * 60)
    
    results = benchmark_attention_models()
    
    print("\nCOMPREHENSIVE BENCHMARK REPORT")
    print("=" * 60)
    
    for config, metrics in results.items():
        print(f"\nConfiguration: {config}")
        print(f"  Custom Implementation: {metrics['custom_time']:.2f}ms")
        print(f"  PyTorch Built-in:     {metrics['pytorch_time']:.2f}ms")
        print(f"  Speedup:              {metrics['speedup']:.2f}x")
        print(f"  Output Quality:       {metrics['similarity']:.4f}")
        if metrics['custom_memory'] > 0:
            print(f"  Custom Memory:        {metrics['custom_memory']:.1f}MB")
            print(f"  PyTorch Memory:       {metrics['pytorch_memory']:.1f}MB")
    
    avg_speedup = np.mean([r['speedup'] for r in results.values()])
    avg_similarity = np.mean([r['similarity'] for r in results.values()])
    
    print(f"\nOVERALL ANALYSIS")
    print(f"  Average PyTorch Speedup: {avg_speedup:.2f}x")
    print(f"  Average Output Quality:  {avg_similarity:.4f}")
    
    if avg_similarity > 0.95:
        print("  Output quality is excellent (similarity > 0.95)")
    elif avg_similarity > 0.90:
        print("  Output quality is good (similarity > 0.90)")
    else:
        print("  Output quality needs improvement (similarity < 0.90)")
    
    print("\nGenerating visualizations...")
    plot_benchmark_results(results)
    
    print("\nBenchmarking complete! Results saved to 'attention_benchmark_results.png'")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_tests()

    except Exception as e:
        print(f"Error during benchmarking: {e}")
        print("Please check your PyTorch installation and GPU availability.")


