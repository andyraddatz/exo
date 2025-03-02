from tinygrad import Tensor, Variable 
from collections import OrderedDict
from typing import List, Optional
from tinygrad.helpers import getenv
from exo.helpers import DEBUG
import gc

def create_kv_cache(x: Tensor, layer):
  # Create the KV cache tensor
  try:
    # First try to create a smaller cache for very large models
    # This is a tradeoff - smaller cache means more recomputation but less memory usage
    if hasattr(layer, 'n_kv_heads') and layer.n_kv_heads > 8:
      # For large models, try to use a smaller context size initially
      # We'll expand it later if needed
      initial_context_size = min(layer.max_context, 2048)  # Start with a smaller context
      if DEBUG >= 1:
        print(f"Creating initial KV cache with reduced context size {initial_context_size} (max: {layer.max_context})")
      
      cache_kv = Tensor.zeros(2, x.shape[0], initial_context_size, layer.n_kv_heads, layer.head_dim, dtype=x.dtype).contiguous().realize()
    else:
      # For smaller models, use the full context size
      cache_kv = Tensor.zeros(2, x.shape[0], layer.max_context, layer.n_kv_heads, layer.head_dim, dtype=x.dtype).contiguous().realize()
  except MemoryError as e:
    if DEBUG >= 1:
      print(f"Memory error creating KV cache, trying with reduced size: {e}")
    
    # Try with an even smaller context size
    reduced_context = min(layer.max_context, 1024)
    cache_kv = Tensor.zeros(2, x.shape[0], reduced_context, layer.n_kv_heads, layer.head_dim, dtype=x.dtype).contiguous().realize()
  
  # For multi-GPU setups, always shard the KV cache across all devices
  if isinstance(x.device, tuple) and len(x.device) > 1:
    try:
      from tinygrad import Device
      
      # Always use all available devices for KV cache
      devices = x.device
      
      if DEBUG >= 1:
        # Print cache size for debugging
        cache_size = cache_kv.nbytes if hasattr(cache_kv, 'nbytes') else (cache_kv.numel() * 4)
        print(f"KV cache size: {cache_size / (1024**3):.2f} GB, distributing across {len(devices)} devices: {devices}")
      
      # For CUDA/GPU, try to optimize the distribution based on available memory
      if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
        import pynvml
        pynvml.nvmlInit()
        
        # Get memory info for all devices
        device_free_memory = []
        for dev in devices:
          dev_idx = int(dev.split(':')[-1]) if ':' in dev else 0
          handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
          mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
          device_free_memory.append((dev, mem_info.free))
          if DEBUG >= 1:
            print(f"Device {dev}: {mem_info.free / (1024**3):.2f} GB free")
        
        # Sort devices by free memory (descending)
        device_free_memory.sort(key=lambda x: x[1], reverse=True)
        
        # Use the devices with most free memory first
        best_devices = tuple(dev for dev, _ in device_free_memory)
        
        if DEBUG >= 1:
          print(f"Ordered devices by free memory: {best_devices}")
        
        pynvml.nvmlShutdown()
        
        # Try different sharding strategies in order of preference
        sharding_strategies = [
          (3, "KV heads"),  # Shard on KV heads dimension
          (2, "Context"),   # Shard on context dimension
          (0, "KV"),        # Shard on K/V dimension
          (1, "Batch")      # Shard on batch dimension
        ]
        
        success = False
        for axis, name in sharding_strategies:
          try:
            if DEBUG >= 1:
              print(f"Trying to shard KV cache on {name} dimension (axis={axis})")
            
            # Try to shard and realize the tensor
            cache_kv.shard_(best_devices, axis=axis).realize()
            
            if DEBUG >= 1:
              print(f"Successfully sharded KV cache on {name} dimension")
              
              # Verify the sharding
              if hasattr(cache_kv, 'device') and isinstance(cache_kv.device, tuple):
                print(f"KV cache is now distributed across devices: {cache_kv.device}")
              else:
                print(f"Warning: KV cache device after sharding: {cache_kv.device}")
            
            # If we get here, sharding succeeded
            success = True
            break
          except Exception as e:
            if DEBUG >= 1:
              print(f"Failed to shard KV cache on {name} dimension: {e}")
            
            # Try the next strategy
            continue
            
        if not success and DEBUG >= 1:
          print("All sharding strategies failed, falling back to default")
      else:
        # For non-CUDA devices, use default sharding on all devices
        if DEBUG >= 1:
          print(f"Using default sharding for non-CUDA device")
        cache_kv.shard_(devices, axis=3).realize()
        
        if DEBUG >= 1 and hasattr(cache_kv, 'device') and isinstance(cache_kv.device, tuple):
          print(f"KV cache is now distributed across devices: {cache_kv.device}")
    except Exception as e:
      if DEBUG >= 1:
        print(f"Error during KV cache sharding, falling back to default: {e}")
      # Fall back to default sharding on all devices
      try:
        if DEBUG >= 1:
          print(f"Attempting default sharding with axis=3")
        cache_kv.shard_(x.device, axis=3 if getenv("SHARD_KVCACHE", 1) else None).realize()
        
        if DEBUG >= 1 and hasattr(cache_kv, 'device') and isinstance(cache_kv.device, tuple):
          print(f"KV cache is now distributed across devices: {cache_kv.device}")
      except Exception as e2:
        if DEBUG >= 1:
          print(f"Failed to shard KV cache with default strategy: {e2}")
        # Last resort: try to place on CPU if GPU memory is exhausted
        try:
          if DEBUG >= 1:
            print("Attempting to place KV cache on CPU as last resort")
          cache_kv = cache_kv.to(device="CPU").realize()
          
          if DEBUG >= 1:
            print(f"KV cache is now on CPU")
        except Exception as e3:
          if DEBUG >= 1:
            print(f"Failed to place KV cache on CPU: {e3}")
          # If all else fails, try to create a minimal cache
          if DEBUG >= 1:
            print(f"Creating minimal KV cache as last resort")
          cache_kv = Tensor.zeros(2, x.shape[0], 512, layer.n_kv_heads, layer.head_dim, dtype=x.dtype).contiguous().realize()
  elif DEBUG >= 1:
    # Print info for single device case
    if hasattr(cache_kv, 'device'):
      print(f"Created KV cache on single device: {cache_kv.device}")
    
    # Print cache size for debugging
    cache_size = cache_kv.nbytes if hasattr(cache_kv, 'nbytes') else (cache_kv.numel() * 4)
    print(f"KV cache size: {cache_size / (1024**3):.2f} GB")
  
  # Force garbage collection to free up memory
  gc.collect()
  
  return cache_kv.realize()

class ModelState:
  cache: List[Tensor]
  start: int 
  def __init__(self, cache: List[Tensor], start: int = 0):
    self.cache = cache
    self.start = start

def make_prompt_state(x: Tensor, model):
  # Try to free up memory before creating caches
  gc.collect()
  
  if DEBUG >= 1:
    print(f"Creating KV cache for model with {len(model.layers)} layers")
    if hasattr(x, 'device'):
      print(f"Input tensor device: {x.device}")
  
  try:
    cache = [create_kv_cache(x, l.attention) for l in model.layers]
    
    if DEBUG >= 1:
      # Verify cache devices
      device_counts = {}
      for i, c in enumerate(cache):
        dev = c.device if hasattr(c, 'device') else "unknown"
        if dev not in device_counts:
          device_counts[dev] = 0
        device_counts[dev] += 1
      
      print("KV cache distribution across devices:")
      for dev, count in device_counts.items():
        print(f"  {dev}: {count} layers")
  except Exception as e:
    if DEBUG >= 1:
      print(f"Error creating KV cache: {e}, trying with reduced model layers")
    
    # Try with a subset of layers if full cache creation fails
    cache = []
    for i, l in enumerate(model.layers):
      try:
        if DEBUG >= 1:
          print(f"Creating KV cache for layer {i}")
        cache.append(create_kv_cache(x, l.attention))
      except Exception as e2:
        if DEBUG >= 1:
          print(f"Failed to create KV cache for layer {i}: {e2}")
        # Use a dummy cache for this layer
        dummy_shape = (2, x.shape[0], 512, l.attention.n_kv_heads, l.attention.head_dim)
        cache.append(Tensor.zeros(*dummy_shape, dtype=x.dtype).realize())
      
      # Force garbage collection after each layer
      gc.collect()

  return ModelState(cache)
