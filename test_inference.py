import torch
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from collections import defaultdict
import copy

def get_model_size_mb(model):
    """
    Calculate model size in MB
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def convert_to_fp16(model):
    """
    Convert model to FP16 (float16)
    """
    print("Converting model to FP16...")
    model_fp16 = model.to(torch.float16)
    return model_fp16

def convert_to_bf16(model):
    """
    Convert model to BF16 (bfloat16)
    """
    print("Converting model to BF16...")
    model_bf16 = model.to(torch.bfloat16)
    return model_bf16

def print_model_info(model, title="MODEL INFO"):
    """
    Print model precision and size information
    """
    print("\n" + "="*70)
    print(title)
    print("="*70)
    
    # Get precision
    dtypes = set()
    total_params = 0
    
    for param in model.parameters():
        dtypes.add(param.dtype)
        total_params += param.numel()
    
    # Get size
    size_mb = get_model_size_mb(model)
    
    print(f"Precision:        {', '.join(str(d) for d in dtypes)}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size:       {size_mb:.2f} MB")
    print("="*70 + "\n")
    
    return size_mb


def get_model_precision(model):
    """
    Analyze and print the precision (dtype) of a loaded PyTorch model
    """
    print("\n" + "="*70)
    print("MODEL PRECISION ANALYSIS")
    print("="*70)
    
    # Count parameters by dtype
    dtype_counts = defaultdict(int)
    dtype_params = defaultdict(list)
    total_params = 0
    
    for name, param in model.named_parameters():
        dtype = param.dtype
        dtype_counts[dtype] += param.numel()
        dtype_params[dtype].append(name)
        total_params += param.numel()
    
    # Print summary
    print(f"\nTotal Parameters: {total_params:,}")
    print("\nParameter Distribution by Precision:")
    print("-" * 70)
    
    for dtype in sorted(dtype_counts.keys(), key=str):
        count = dtype_counts[dtype]
        percentage = (count / total_params) * 100
        print(f"  {str(dtype):15s}: {count:12,} parameters ({percentage:5.2f}%)")
    
    # Determine dominant precision
    dominant_dtype = max(dtype_counts.items(), key=lambda x: x[1])[0]
    print(f"\nDominant Precision: {dominant_dtype}")
    
    # Show examples of layers for each dtype
    print("\n" + "-"*70)
    print("Sample Layers by Precision:")
    print("-" * 70)
    
    for dtype, param_names in dtype_params.items():
        print(f"\n{str(dtype)} layers (showing up to 5):")
        for name in param_names[:5]:
            param = dict(model.named_parameters())[name]
            print(f"  - {name:50s} {list(param.shape)}")
        if len(param_names) > 5:
            print(f"  ... and {len(param_names) - 5} more layers")
    
    # Check buffers (non-trainable parameters like batch norm running stats)
    buffer_dtypes = defaultdict(int)
    for name, buffer in model.named_buffers():
        buffer_dtypes[buffer.dtype] += buffer.numel()
    
    if buffer_dtypes:
        print("\n" + "-"*70)
        print("Buffer (Non-trainable) Precision:")
        print("-" * 70)
        for dtype, count in buffer_dtypes.items():
            print(f"  {str(dtype):15s}: {count:12,} elements")
    
    print("\n" + "="*70)
    
    return {
        'dominant_dtype': dominant_dtype,
        'dtype_counts': dict(dtype_counts),
        'total_params': total_params,
        'buffer_dtypes': dict(buffer_dtypes)
    }

def load_dinov3_model(model_path):
    """
    Load a pretrained DINOv3 model from local pytorch weights file
    """
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Print weight precision information
    print("\n" + "="*60)
    print("MODEL WEIGHT PRECISION")
    print("="*60)
    
    dtypes = {}
    for name, param in state_dict.items():
        dtype = str(param.dtype)
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} parameters")
    
    # Get a sample parameter to show precision
    sample_param = next(iter(state_dict.values()))
    print(f"\nModel weights precision: {sample_param.dtype}")
    print("="*60 + "\n")
    
    return state_dict, sample_param.dtype

def prepare_image(image_size=256):
    """
    Create a dummy image and preprocessing transform
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create a dummy RGB image
    dummy_image = Image.new('RGB', (image_size, image_size), color='red')
    return transform(dummy_image).unsqueeze(0)

def benchmark_inference(model, input_tensor, device, num_warmup=10, num_iterations=100):
    """
    Benchmark inference time on specified device
    """
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warmup runs
    print(f"Running {num_warmup} warmup iterations on {device}...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    
    # Synchronize for GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Actual benchmark
    print(f"Running {num_iterations} benchmark iterations on {device}...")
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(input_tensor)
            
            # Synchronize for accurate GPU timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
    
    times = np.array(times)
    return {
        'mean': np.mean(times) * 1000,  # Convert to ms
        'std': np.std(times) * 1000,
        'min': np.min(times) * 1000,
        'max': np.max(times) * 1000,
        'median': np.median(times) * 1000
    }

def main():
    # Configuration
    model_path = '/mnt/d/Projects/Astar/dfdino/pretrained_models/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'  # Update this path
    num_warmup = 10
    num_iterations = 100
    
    # Load model weights
    #state_dict, weight_dtype = load_dinov3_model(model_path)
    
    # Create model architecture (you may need to adjust this based on your model)
    # For now, we'll use torch.hub to get the architecture, then load your weights
    print("Loading model architecture...")
    dinov3_vith16plus = torch.hub.load('/mnt/d/Projects/Astar/dfdino/dinov3', 'dinov3_vith16plus', source='local', weights=model_path)
    dinov3_vith16plus.eval()

    # precision_info = get_model_precision(dinov3_vith16plus)

    # # Print summary
    # print("\nSUMMARY:")
    # print(precision_info)

    size_fp32 = print_model_info(dinov3_vith16plus, "ORIGINAL MODEL (FP32)")
    
    # Convert to FP16
    #model_fp16 = convert_to_fp16(copy.deepcopy(dinov3_vith16plus))
    #size_fp16 = print_model_info(model_fp16, "CONVERTED MODEL (FP16)")
    
    
    #Prepare input
    input_tensor = prepare_image()
    
    print(f"\n{'='*60}")
    print(f"Benchmarking DINOv3 Model")
    # print(f"Weight Precision: {weight_dtype}")
    print(f"{'='*60}\n")
    
    # CPU Benchmark
    # cpu_device = torch.device('cpu')
    # print("=" * 40)
    # print("CPU INFERENCE")
    # print("=" * 40)
    # cpu_stats = benchmark_inference(dinov3_vith16plus, input_tensor, cpu_device, 
    #                                 num_warmup, num_iterations)
    
    # print(f"\nCPU Results:")
    # print(f"  Mean:   {cpu_stats['mean']:.2f} ms")
    # print(f"  Std:    {cpu_stats['std']:.2f} ms")
    # print(f"  Min:    {cpu_stats['min']:.2f} ms")
    # print(f"  Max:    {cpu_stats['max']:.2f} ms")
    # print(f"  Median: {cpu_stats['median']:.2f} ms")
    
    # GPU Benchmark (if available)
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        print(f"\n{'='*40}")
        print(f"GPU INFERENCE ({torch.cuda.get_device_name(0)})")
        print("=" * 40)
        gpu_stats = benchmark_inference(dinov3_vith16plus, input_tensor, gpu_device,
                                       num_warmup, num_iterations)
        
        print(f"\nGPU Results:")
        print(f"  Mean:   {gpu_stats['mean']:.2f} ms")
        print(f"  Std:    {gpu_stats['std']:.2f} ms")
        print(f"  Min:    {gpu_stats['min']:.2f} ms")
        print(f"  Max:    {gpu_stats['max']:.2f} ms")
        print(f"  Median: {gpu_stats['median']:.2f} ms")
    else:
        print("\n⚠️  CUDA not available. Skipping GPU benchmark.")

    # Convert to BF16
    model_bf16 = convert_to_bf16(dinov3_vith16plus)
    size_bf16 = print_model_info(model_bf16, "CONVERTED MODEL (BF16)")
    input_tensor = prepare_image().to(torch.bfloat16)
    
    if torch.cuda.is_available():
        gpu_device = torch.device('cuda')
        print(f"\n{'='*40}")
        print(f"GPU INFERENCE ({torch.cuda.get_device_name(0)})")
        print("=" * 40)
        gpu_stats_ = benchmark_inference(model_bf16, input_tensor, gpu_device,
                                       num_warmup, num_iterations)
        
        print(f"\nGPU Results:")
        print(f"  Mean:   {gpu_stats_['mean']:.2f} ms")
        print(f"  Std:    {gpu_stats_['std']:.2f} ms")
        print(f"  Min:    {gpu_stats_['min']:.2f} ms")
        print(f"  Max:    {gpu_stats_['max']:.2f} ms")
        print(f"  Median: {gpu_stats_['median']:.2f} ms")
        
        speedup = gpu_stats['mean'] / gpu_stats_['mean']
        print(f"\n{'='*40}")
        print(f"Speedup: {speedup:.2f}x faster on GPU with BF16")
        print("=" * 40)
    else:
        print("\n⚠️  CUDA not available. Skipping GPU benchmark.")
    
    # Print comparison
    print("="*70)
    print("SIZE COMPARISON")
    print("="*70)
    print(f"FP32:  {size_fp32:8.2f} MB  (baseline)")
    #print(f"FP16:  {size_fp16:8.2f} MB  ({size_fp32/size_fp16:.2f}x smaller)")
    print(f"BF16:  {size_bf16:8.2f} MB  ({size_fp32/size_bf16:.2f}x smaller)")
    
    print(f"\n{'='*60}")
    print("Benchmark Complete!")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
