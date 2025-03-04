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

# **** helper functions ****
def concat_weights(models, device=None):
  def convert(name) -> Tensor:
    disk_tensors: List[Tensor] = [model[name] for model in models]
    if len(disk_tensors) == 1 or len(disk_tensors[0].shape) == 1:
      return disk_tensors[0].to(device=device)
    axis = 1 if name.endswith(".attention.wo.weight") or name.endswith(".feed_forward.w2.weight") else 0
    lazy_tensors = [data.to(device=device) for data in disk_tensors]
    return lazy_tensors[0].cat(*lazy_tensors[1:], dim=axis)

  return {name: convert(name) for name in {name: None for model in models for name in model}}

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

def get_available_devices() -> Tuple[str, ...]:
  # Determine available devices first, before building the model
  try:
    from tinygrad import Device
    # Try to get available GPU devices
    if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
      import pynvml
      pynvml.nvmlInit()
      num_gpus = pynvml.nvmlDeviceGetCount()
      
      # Print memory info for debugging
      if DEBUG >= 1:
        print(f"Found {num_gpus} GPUs")
        for i in range(num_gpus):
          handle = pynvml.nvmlDeviceGetHandleByIndex(i)
          mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
          print(f"GPU {i}: {mem_info.free / (1024**3):.2f}GB free of {mem_info.total / (1024**3):.2f}GB total")
      
      devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(num_gpus))
      pynvml.nvmlShutdown()
    else:
      # Default to single device
      devices = (Device.DEFAULT,)
  except Exception as e:
    # If any error occurs, default to single device
    if DEBUG >= 1:
      print(f"Error detecting GPUs: {e}")
    devices = (Device.DEFAULT,)
    
  return devices

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

def assign_weights(weights, devices): 
  def assign_best_device_for_tensor(tensor: Tensor, devices: Tuple[str, ...], memory_info: Dict[str, Dict[str, int]]) -> Tensor:
    """
    Find the best device to place a tensor based on available memory.
    
    Args:
      tensor: Tensor to place
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

    # get the tensor size
    tensor_size = 0
    if hasattr(tensor, 'nbytes'):
      if callable(tensor.nbytes):
          tensor_size += tensor.nbytes()
      else:
          tensor_size += tensor.nbytes
    else:
      tensor_size += tensor.numel() * 4
    
    # Update the memory info to reflect this allocation
    if best_device in memory_info:
      memory_info[best_device]['free'] -= tensor_size
      memory_info[best_device]['used'] += tensor_size

    # TODO: now w'ere just hacking
    # if best_device == "NV:0":
    #   best_device = "CUDA:0"
    # elif best_device == "NV:1":
    #   best_device = "CUDA:1"

    tensor = tensor.to(best_device) 
      
    if DEBUG >= 3:  # Very verbose debug
      print(f"Allocated {tensor_size / (1024**3):.3f} GB on {best_device}, remaining free: {memory_info[best_device]['free'] / (1024**3):.3f} GB")
  
    return tensor
  
  # First, get memory info for all devices - we'll maintain this throughout the process
  memory_info = get_gpu_memory_info(devices)
  if DEBUG >= 1:
    for dev, info in memory_info.items():
      print(f"Device {dev}: {info['free'] / (1024**3):.2f} GB free of {info['total'] / (1024**3):.2f} GB")

  for name, tensor in weights.items():
    # Find the best device based on current memory state
    if len(memory_info) > 1:
      weights[name] = assign_best_device_for_tensor(tensor, devices, memory_info)
    else:
      weights[name] = tensor.to(devices[0])

  return weights