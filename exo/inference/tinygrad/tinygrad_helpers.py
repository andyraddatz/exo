from tinygrad.nn.state import safe_load, torch_load
from tinygrad import Tensor
from pathlib import Path
import json
from typing import List, Tuple, Dict, Optional
from exo.inference.shard import Shard
from exo.helpers import DEBUG
from exo.download.hf.hf_helpers import get_allow_patterns
from fnmatch import fnmatch
import re


# New utility function for GPU memory management
def get_gpu_memory_info(devices: Optional[Tuple[str, ...]] = None) -> Dict[str, Dict[str, int]]:
  """
  Get memory information for available GPU devices.
  
  Args:
    devices: Optional tuple of device strings. If None, will detect all available devices.
    
  Returns:
    Dictionary mapping device strings to memory info dictionaries with 'total', 'used', and 'free' keys.
  """
  from tinygrad import Device
  
  result = {}
  
  try:
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
      import pynvml
      pynvml.nvmlInit()
      
      if devices is None:
        # Detect all available devices
        num_gpus = pynvml.nvmlDeviceGetCount()
        devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(num_gpus))
      
      for dev in devices:
        dev_idx = int(dev.split(':')[-1]) if ':' in dev else 0
        handle = pynvml.nvmlDeviceGetHandleByIndex(dev_idx)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        result[dev] = {
          'total': mem_info.total,
          'used': mem_info.used,
          'free': mem_info.free
        }
      
      pynvml.nvmlShutdown()
  except Exception as e:
    if DEBUG >= 2:
      print(f"Error getting GPU memory info: {e}")
  
  return result


def find_best_device_for_tensor(tensor_size: int, devices: Tuple[str, ...], memory_info: Dict[str, Dict[str, int]]) -> str:
  """
  Find the best device to place a tensor based on available memory.
  
  Args:
    tensor_size: Size of the tensor in bytes
    devices: Tuple of device strings to consider
    memory_info: Dictionary of device memory information to use (will be updated)
    
  Returns:
    The device string with the most available memory
  """
  if not devices or len(devices) == 1:
    return devices[0] if devices else None
  
  # Find device with most free memory
  best_device = devices[0]
  max_free_memory = 0
  
  for dev in devices:
    if dev in memory_info and memory_info[dev]['free'] > max_free_memory:
      max_free_memory = memory_info[dev]['free']
      best_device = dev
  
  # Update the memory info to reflect this allocation
  if best_device in memory_info:
    memory_info[best_device]['free'] -= tensor_size
    memory_info[best_device]['used'] += tensor_size
    
    if DEBUG >= 3:  # Very verbose debug
      print(f"Allocated {tensor_size / (1024**3):.3f} GB on {best_device}, remaining free: {memory_info[best_device]['free'] / (1024**3):.3f} GB")
  
  return best_device


# **** helper functions ****
def concat_weights(models, devices=None):
  # Get all unique weight names first
  weight_names = {name: None for model in models for name in model}
  
  # If we're not using multiple devices, use the simple approach
  if not isinstance(devices, tuple) or len(devices) <= 1:
    target_device = devices[0] if isinstance(devices, tuple) else devices
    
    if DEBUG >= 1:
      print(f"Using single device mode with target device: {target_device}")
    
    def convert_single_device(name) -> Tensor:
      disk_tensors: List[Tensor] = [model[name] for model in models if name in model]
      if not disk_tensors:
        return None
      if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
        return disk_tensors[0].to(device=target_device)
      
      axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
      lazy_tensors = [data.to(device=target_device) for data in disk_tensors]
      return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
    
    return {name: convert_single_device(name) for name in weight_names if convert_single_device(name) is not None}
  
  # For multi-GPU setup, we need to be more strategic
  if DEBUG >= 1:
    print(f"Using multi-GPU mode with devices: {devices}")
  
  # First, get memory info for all devices - we'll maintain this throughout the process
  memory_info = get_gpu_memory_info(devices)
  if DEBUG >= 1:
    for dev, info in memory_info.items():
      print(f"Device {dev}: {info['free'] / (1024**3):.2f} GB free of {info['total'] / (1024**3):.2f} GB")
  
  # Group weights by layer to distribute them evenly
  layer_weights = {}
  non_layer_weights = []
  
  # Identify large tensors for special handling
  large_tensors = []
  
  for name in weight_names:
    # Estimate tensor size
    disk_tensors = [model[name] for model in models if name in model]
    if not disk_tensors:
      continue
      
    tensor_size = 0
    for t in disk_tensors:
        if hasattr(t, 'nbytes'):
            if callable(t.nbytes):
                tensor_size += t.nbytes()
            else:
                tensor_size += t.nbytes
        else:
            tensor_size += t.numel() * 4
    
    # Consider tensors larger than 500MB as "large"
    if tensor_size > 500 * 1024 * 1024:
      large_tensors.append((name, tensor_size))
      continue
      
    layer_match = re.search(r"layers\.(\d+)\.", name)
    if layer_match:
      layer_num = int(layer_match.group(1))
      if layer_num not in layer_weights:
        layer_weights[layer_num] = []
      layer_weights[layer_num].append(name)
    else:
      non_layer_weights.append(name)
  
  # Sort large tensors by size (descending)
  large_tensors.sort(key=lambda x: x[1], reverse=True)
  
  if DEBUG >= 1 and large_tensors:
    print(f"Identified {len(large_tensors)} large tensors:")
    for name, size in large_tensors[:5]:  # Show top 5
      print(f"  {name}: {size / (1024**3):.2f} GB")
  
  # Calculate total layers and distribute them across devices
  total_layers = len(layer_weights)
  layers_per_device = total_layers // len(devices)
  remainder = total_layers % len(devices)
  
  # Create a mapping of layer to device
  layer_to_device = {}
  current_device_idx = 0
  current_device_count = 0
  
  # Distribute layers evenly across devices
  for layer_num in sorted(layer_weights.keys()):
    layer_to_device[layer_num] = devices[current_device_idx]
    current_device_count += 1
    
    # Move to next device when we've assigned enough layers to the current one
    layers_for_this_device = layers_per_device + (1 if current_device_idx < remainder else 0)
    if current_device_count >= layers_for_this_device:
      current_device_idx = (current_device_idx + 1) % len(devices)
      current_device_count = 0
  
  if DEBUG >= 1:
    for dev_idx, dev in enumerate(devices):
      layers_on_device = [layer for layer, d in layer_to_device.items() if d == dev]
      print(f"Device {dev}: assigned layers {layers_on_device}")
  
  # Pre-assign large tensors to devices with most memory
  large_tensor_device = {}
  for name, size in large_tensors:
    best_device = find_best_device_for_tensor(size, devices, memory_info)
    large_tensor_device[name] = best_device
    
    if DEBUG >= 2:
      print(f"Assigned large tensor '{name}' ({size / (1024**3):.2f} GB) to device {best_device}")
  
  # Distribute non-layer weights across devices
  non_layer_device = {}
  for i, name in enumerate(non_layer_weights):
    # Get tensor size for better allocation
    disk_tensors = [model[name] for model in models if name in model]
    if not disk_tensors:
      continue
      
    tensor_size = 0
    for t in disk_tensors:
        if hasattr(t, 'nbytes'):
            if callable(t.nbytes):
                tensor_size += t.nbytes()
            else:
                tensor_size += t.nbytes
        else:
            tensor_size += t.numel() * 4
    
    # Find best device based on current memory state
    target_device = find_best_device_for_tensor(tensor_size, devices, memory_info)
    non_layer_device[name] = target_device
    
    if DEBUG >= 2:
      print(f"Assigned non-layer weight '{name}' ({tensor_size / (1024**3):.3f} GB) to device {target_device}")
  
  # Function to convert tensors with our layer-to-device mapping
  def convert_multi_device(name) -> Tensor:
    try:
      disk_tensors: List[Tensor] = [model[name] for model in models if name in model]
      if not disk_tensors:
        if DEBUG >= 2:
          print(f"Warning: No data found for weight {name}")
        return None
      
      # Calculate tensor size for memory tracking
      tensor_size = 0
      for t in disk_tensors:
          if hasattr(t, 'nbytes'):
              if callable(t.nbytes):
                  tensor_size += t.nbytes()
              else:
                  tensor_size += t.nbytes
          else:
              tensor_size += t.numel() * 4
      
      # For small tensors, use the first device
      if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
        # For non-layer weights, use our pre-assigned device
        if name in non_layer_device:
          target_device = non_layer_device[name]
        else:
          # Small tensors go to the device with most free memory
          target_device = find_best_device_for_tensor(tensor_size, devices, memory_info)
        
        if DEBUG >= 3:
          print(f"Placing small tensor '{name}' on device {target_device}")
        return disk_tensors[0].to(device=target_device)
      
      # For pre-assigned large tensors, use the pre-determined device
      if name in large_tensor_device:
        target_device = large_tensor_device[name]
        if DEBUG >= 2:
          print(f"Placing large tensor '{name}' on pre-assigned device {target_device}")
      # For layer weights, use our pre-computed mapping
      elif (layer_match := re.search(r"layers\.(\d+)\.", name)):
        layer_num = int(layer_match.group(1))
        target_device = layer_to_device[layer_num]
        if DEBUG >= 3:  # More verbose debug level
          print(f"Placing layer {layer_num} weight '{name}' on device {target_device}")
      else:
        # For non-layer weights, use the device with most free memory
        target_device = find_best_device_for_tensor(tensor_size, devices, memory_info)
      
      # Concatenate tensors on the target device
      axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
      
      try:
        lazy_tensors = [data.to(device=target_device) for data in disk_tensors]
        result = lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
        return result
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error placing tensor {name} on device {target_device}: {e}")
        
        # Try all other devices in order of available memory
        devices_by_memory = sorted(
          [(d, info['free']) for d, info in memory_info.items()],
          key=lambda x: x[1],
          reverse=True
        )
        
        for alt_device, _ in devices_by_memory:
          if alt_device != target_device:
            try:
              if DEBUG >= 1:
                print(f"Trying alternate device {alt_device} for tensor {name}")
              lazy_tensors = [data.to(device=alt_device) for data in disk_tensors]
              result = lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)
              
              # Update memory info for the successful device
              if alt_device in memory_info:
                memory_info[alt_device]['free'] -= tensor_size
                memory_info[alt_device]['used'] += tensor_size
                
              return result
            except Exception:
              continue
        
        # If all devices fail, try to split the tensor across devices
        if len(devices) > 1:
          try:
            if DEBUG >= 1:
              print(f"Attempting to shard tensor {name} across all devices")
            
            # First move to CPU, then shard
            cpu_tensors = [data.to(device="CPU") for data in disk_tensors]
            result = cpu_tensors[0].cat(*cpu_tensors[1:], dim=axis)
            
            # Shard across all devices
            result = result.shard_(devices).realize()
            if DEBUG >= 1:
              print(f"Successfully sharded tensor {name} across devices {devices}")
              
            # Update memory info - distribute evenly across devices
            per_device_size = tensor_size / len(devices)
            for dev in devices:
              if dev in memory_info:
                memory_info[dev]['free'] -= per_device_size
                memory_info[dev]['used'] += per_device_size
                
            return result
          except Exception as e2:
            if DEBUG >= 1:
              print(f"Failed to shard tensor {name}: {e2}")
        
        # If all attempts fail, raise the original error
        raise
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to process weight {name}: {e}")
      raise
  
  # Convert all weights using our strategy
  result = {}
  device_distribution = {d: 0 for d in devices}
  
  for name in weight_names:
    try:
      tensor = convert_multi_device(name)
      if tensor is not None:
        result[name] = tensor
        
        # Track device distribution
        if hasattr(tensor, 'device'):
          if tensor.device in device_distribution:
            device_distribution[tensor.device] += 1
          else:
            device_distribution[tensor.device] = 1
    except Exception as e:
      if DEBUG >= 1:
        print(f"Failed to load weight {name}: {e}")
      # Continue with other weights instead of failing completely
      continue
  
  # Print final distribution stats
  if DEBUG >= 1:
    print("Final weight distribution across devices:")
    for dev, count in device_distribution.items():
      print(f"  {dev}: {count} tensors")
    
    # Print memory usage after distribution
    for dev, info in get_gpu_memory_info(devices).items():
      print(f"Device {dev} after distribution: {info['free'] / (1024**3):.2f} GB free, {info['used'] / (1024**3):.2f} GB used")
  
  return result


def load(fn: str, shard: Shard):
  if fn.endswith('.index.json'):
    with open(fn) as fp:
      weight_map = json.load(fp)['weight_map']
    parts = {}
    filtered_weight_map = {}
    allow_patterns = get_allow_patterns(weight_map, shard)
    for k, n in weight_map.items():
      if allow_patterns is not None and not any(fnmatch(n, r) for r in allow_patterns):
        continue
      if k.startswith("model.layers."):
        layer_num = int(k.split('.')[2])
        if layer_num < shard.start_layer or layer_num > shard.end_layer:
          continue

      parts[n] = load(str(Path(fn).parent/Path(n).name), shard)
      filtered_weight_map[k] = n
    if DEBUG >= 2: print(f"Excluded model param keys for {shard=}: {sorted(set(weight_map.keys()) - set(filtered_weight_map.keys()))}")
    return {k: parts[n][k] for k, n in filtered_weight_map.items()}
  elif fn.endswith(".safetensors"):
    weight_map = safe_load(fn)
    for k in list(weight_map):
      if (n := re.search(r"\.(\d+)\.", k)) and not (shard.start_layer <= int(n.group(1)) <= shard.end_layer):
          del weight_map[k]
    return weight_map
  else:
    return torch_load(fn)


def print_gpu_memory_usage(message="Current GPU memory usage", devices=None):
  """
  Print detailed memory usage information for all GPUs.
  
  Args:
    message: Optional message to print before memory info
    devices: Optional list of devices to check, if None will check all available GPUs
  """
  if DEBUG < 1:
    return
    
  from tinygrad import Device
  
  try:
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
      import pynvml
      pynvml.nvmlInit()
      
      num_gpus = pynvml.nvmlDeviceGetCount()
      if devices is None:
        devices = range(num_gpus)
      else:
        # Extract device indices from device strings
        devices = [int(dev.split(':')[-1]) if ':' in dev else 0 for dev in devices]
      
      print(f"{message}:")
      for i in devices:
        if i >= num_gpus:
          continue
          
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Get process info to see what's using memory
        try:
          processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
          process_info = ""
          if processes:
            process_info = " - Processes: " + ", ".join(
              f"{p.pid}({p.usedGpuMemory / (1024**2):.0f}MB)" for p in processes
            )
        except:
          process_info = ""
        
        print(f"  GPU {i}: {mem_info.used / (1024**3):.2f}GB used, {mem_info.free / (1024**3):.2f}GB free of {mem_info.total / (1024**3):.2f}GB total{process_info}")
      
      pynvml.nvmlShutdown()
  except Exception as e:
    print(f"Error getting GPU memory info: {e}")


def force_memory_cleanup():
  """
  Aggressively clean up memory by forcing garbage collection and clearing CUDA cache if available.
  """
  import gc
  
  # Force garbage collection
  gc.collect()
  
  # Try to clear CUDA cache if available
  from tinygrad import Device
  
  try:
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
      # Try to access CUDA memory management functions
      try:
        import torch
        if hasattr(torch.cuda, 'empty_cache'):
          torch.cuda.empty_cache()
          if DEBUG >= 2:
            print("Cleared PyTorch CUDA cache")
      except:
        pass
      
      # Try tinygrad's own memory management
      try:
        # Try to synchronize the device
        from tinygrad.runtime.driver.cuda import cuda
        if hasattr(cuda, 'cuCtxSynchronize'):
          cuda.cuCtxSynchronize()
          if DEBUG >= 2:
            print("Synchronized CUDA context")
      except:
        pass
      
      # Try another approach to synchronize
      try:
        from tinygrad.runtime.ops_cuda import CUDADevice
        for i in range(16):  # Try all possible device indices
          try:
            dev = CUDADevice(i)
            dev.synchronize()
            if DEBUG >= 2:
              print(f"Synchronized CUDA device {i}")
          except:
            pass
      except:
        pass
  except Exception as e:
    if DEBUG >= 2:
      print(f"Error cleaning up GPU memory: {e}")
