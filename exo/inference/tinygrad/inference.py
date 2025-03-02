from pathlib import Path
import json
import os
from exo.inference.tinygrad.models.llama import Transformer, TransformerShard, convert_from_huggingface, fix_bf16, sample_logits
from exo.inference.shard import Shard
from exo.inference.tokenizers import resolve_tokenizer
from exo.helpers import DEBUG
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict
from tinygrad import Device, Tensor, nn, Context, TinyJit
from exo.inference.inference_engine import InferenceEngine
import numpy as np
from exo.inference.tinygrad.tinygrad_helpers import concat_weights, load, print_gpu_memory_usage, force_memory_cleanup
from exo.download.shard_download import ShardDownloader
from concurrent.futures import ThreadPoolExecutor
from .stateful_model import make_prompt_state
from .losses import length_masked_ce_loss
from collections import OrderedDict
import asyncio
from typing import Optional
Tensor.no_grad = True 
# default settings
TEMPERATURE = int(os.getenv("TEMPERATURE", 0.85))
TOP_K = 25
TOP_P = 0.9
ALPHA_F = 0.1
ALPHA_P = 0.0
MODEL_PARAMS = {
  "1B": {
    "args": {
      "dim": 2048, "n_heads": 32, "n_kv_heads": 8, "n_layers": 16, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "3B": {
    "args": {
      "dim": 3072, "n_heads": 24, "n_kv_heads": 8, "n_layers": 28, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 8192,
      "rope_scaling": {"factor": 32.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0, "original_max_position_embeddings": 8192, "rope_type": "llama3"}, "tie_word_embeddings": True
    }, "files": 1
  }, "8B": {"args": {"dim": 4096, "n_heads": 32, "n_kv_heads": 8, "n_layers": 32, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 14336}, "files": 1},
  "70B": {"args": {"dim": 8192, "n_heads": 64, "n_kv_heads": 8, "n_layers": 80, "norm_eps": 1e-5, "rope_theta": 500000, "vocab_size": 128256, "hidden_dim": 28672}, "files": 8}
}


def build_transformer(model_path: Path, shard: Shard, model_size="8B", devices=None):
  # Import memory management utilities
  from exo.inference.tinygrad.tinygrad_helpers import print_gpu_memory_usage, force_memory_cleanup
  
  # Clean up memory before starting
  force_memory_cleanup()
  
  # Determine available devices first, before building the model
  if devices is None:
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
        
        from tinygrad import Device
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
  else:
    # Use provided device(s)
    devices = devices if isinstance(devices, tuple) else (devices,)

  # ALWAYS use multi-GPU for large models if available
  if len(devices) > 1 and (model_size == "70B" or model_size == "8B" or model_size == "3B"):
    if DEBUG >= 1:
      print(f"Forcing multi-GPU distribution for {model_size} model across {devices}")
    multi_gpu = True
  else:
    # For smaller models or single GPU, just use the first device
    if DEBUG >= 1 and isinstance(devices, tuple) and len(devices) > 1:
      print(f"Using only first device {devices[0]} for {model_size} model (not using multi-GPU)")
    devices = devices[0] if isinstance(devices, tuple) else devices
    multi_gpu = False
  
  # Print memory usage before building model
  if DEBUG >= 1:
    print_gpu_memory_usage("GPU memory before building model")
  
  # build model - explicitly set the device for the model
  linear = nn.Linear
  if multi_gpu:
    from tinygrad import Device
    # For multi-GPU, we'll explicitly set the device for the model to the first device
    # but we'll distribute weights across all devices later
    with Tensor.train(False):
      model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)
    if DEBUG >= 1:
      print(f"Created model structure on device {Device.DEFAULT}")
  else:
    # For single GPU, just build the model normally
    model = Transformer(**MODEL_PARAMS[model_size]["args"], linear=linear, max_context=8192, jit=True, shard=shard)

  # Print memory usage before loading weights
  if DEBUG >= 1:
    print_gpu_memory_usage("GPU memory before loading weights")

  # Try to free memory before loading weights
  force_memory_cleanup()

  # load weights
  try:
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists(): 
        if DEBUG >= 1:
          print(f"Loading safetensors index and distributing across devices: {devices}")
        # Load the weights first
        raw_weights = load(str(model_path/"model.safetensors.index.json"), shard)
        # Then distribute them across devices if we're in multi-GPU mode
        if isinstance(devices, tuple) and len(devices) > 1:
          if DEBUG >= 1:
            print(f"Distributing safetensors index weights across devices: {devices}")
          weights = concat_weights([raw_weights], devices)
        else:
          weights = raw_weights
      elif (model_path/"model.safetensors").exists(): 
        if DEBUG >= 1:
          print(f"Loading safetensors and distributing across devices: {devices}")
        # Load the weights first
        raw_weights = load(str(model_path/"model.safetensors"), shard)
        # Then distribute them across devices if we're in multi-GPU mode
        if isinstance(devices, tuple) and len(devices) > 1:
          if DEBUG >= 1:
            print(f"Distributing safetensors weights across devices: {devices}")
          weights = concat_weights([raw_weights], devices)
        else:
          weights = raw_weights
      else:
        if DEBUG >= 1:
          print(f"Loading consolidated weights and distributing across devices: {devices}")
        weights = concat_weights([load(str(model_path/f"consolidated.{i:02d}.pth"), shard) for i in range(MODEL_PARAMS[model_size]["files"])], devices)
    else:
      if DEBUG >= 1:
        print(f"Loading single file weights and distributing across devices: {devices}")
      # For single files too, distribute across devices if in multi-GPU mode
      raw_weights = load(str(model_path), shard)
      if isinstance(devices, tuple) and len(devices) > 1:
        if DEBUG >= 1:
          print(f"Distributing single file weights across devices: {devices}")
        weights = concat_weights([raw_weights], devices)
      else:
        weights = raw_weights
  except MemoryError as e:
    if DEBUG >= 1:
      print(f"Memory error loading weights: {e}, trying with more aggressive memory management")
      print_gpu_memory_usage("GPU memory after weight loading failure")
    
    # Force garbage collection
    force_memory_cleanup()
    
    # Try again with more aggressive memory management
    if model_path.is_dir():
      if (model_path/"model.safetensors.index.json").exists() or (model_path/"model.safetensors").exists():
        # For safetensors, we can't do much more
        raise
      else:
        # For consolidated weights, load one by one with GC in between
        weights = {}
        for i in range(MODEL_PARAMS[model_size]["files"]):
          if DEBUG >= 1:
            print(f"Loading weight file {i+1}/{MODEL_PARAMS[model_size]['files']}")
          
          # Load one file at a time
          file_weights = load(str(model_path/f"consolidated.{i:02d}.pth"), shard)
          
          # Merge into main weights dict
          weights.update(file_weights)
          
          # Force garbage collection
          force_memory_cleanup()
        
        # Now distribute across devices
        if DEBUG >= 1 and multi_gpu:
          print(f"Distributing incrementally loaded weights across devices: {devices}")
        weights = concat_weights([weights], devices)
    else:
      # For single files, we can't do much more
      raise
  
  # Print memory usage after loading weights
  if DEBUG >= 1:
    print_gpu_memory_usage("GPU memory after loading weights")
    
    # Check which devices the weights are on
    if multi_gpu:
      device_counts = {}
      for name, tensor in weights.items():
        dev = tensor.device
        if dev == "NV":
          print(f"Weight {name} is on device {dev}")
        if dev not in device_counts:
          device_counts[dev] = 0
        device_counts[dev] += 1
      
      print("Weight distribution across devices:")
      for dev, count in device_counts.items():
        print(f"  {dev}: {count} tensors")
  
  weights = convert_from_huggingface(weights, model, MODEL_PARAMS[model_size]["args"]["n_heads"], MODEL_PARAMS[model_size]["args"]["n_kv_heads"])
  weights = fix_bf16(weights)

  # Force garbage collection before loading state dict
  force_memory_cleanup()

  # Print memory usage before loading state dict
  if DEBUG >= 1:
    print_gpu_memory_usage("GPU memory before loading state dict")

  with Context(BEAM=0):
    try:
      # replace weights in model
      if DEBUG >= 1:
        print(f"Loading state dict with consume=True")

      # test changing device to "NV"
      weights = {k: v.to("NV") for k, v in weights.items() if v.device.startswith("NV")}
      load_state_dict(model, weights, strict=False, consume=True)  # Use consume=True to free memory
      model = TransformerShard(shard, model)
    except MemoryError as e:
      if DEBUG >= 1:
        print(f"Memory error loading state dict: {e}, trying with consume=False")
        print_gpu_memory_usage("GPU memory after state dict loading failure")
      
      # Force garbage collection
      force_memory_cleanup()
      
      # Try again with consume=False
      if DEBUG >= 1:
        print(f"Loading state dict with consume=False")
      load_state_dict(model, weights, strict=False, consume=False)
      model = TransformerShard(shard, model)
  
  # Print final memory usage
  if DEBUG >= 1:
    print_gpu_memory_usage("GPU memory after building transformer")
    
    # Check if the model is using multiple devices
    if multi_gpu:
      print("Verifying model is using multiple devices:")
      # Check a few sample layers to see which devices they're on
      for i, layer in enumerate(model.layers):
        if hasattr(layer.attention.wq, 'weight'):
          print(f"  Layer {i} attention.wq device: {layer.attention.wq.weight.device}")
        if i >= 3:  # Just check a few layers
          break

  return model

_executor = ThreadPoolExecutor(max_workers=1) # singleton so tinygrad always runs on the same thread
class TinygradDynamicShardInferenceEngine(InferenceEngine):
  def __init__(self, shard_downloader: ShardDownloader):
    self.shard = None
    self.shard_downloader = shard_downloader
    self.states = OrderedDict()
    self.executor = _executor

  def poll_state(self, x, request_id: str, max_states=2):
    if request_id not in self.states:
      if len(self.states) >= max_states:
        self.states.popitem(last=False)
      self.states[request_id] = make_prompt_state(x, self.model)
    else:
      self.states.move_to_end(request_id)
    state = self.states[request_id]
    return {"start_pos": state.start, "cache": state.cache}

  async def sample(self, x: np.ndarray, temp=TEMPERATURE, top_p: float = 0.0) -> np.ndarray:
    def sample_wrapper():
      logits = x[:, -1, :]
      return sample_logits(Tensor(logits).flatten(), temp, 0, 0.8, top_p, 0.0).realize().numpy().astype(int)
    return await asyncio.get_running_loop().run_in_executor(self.executor, sample_wrapper)

  async def encode(self, shard: Shard, prompt: str) -> np.ndarray:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.encode, prompt)
    return await asyncio.get_running_loop().run_in_executor(self.executor, np.array, tokens)
  
  async def decode(self, shard: Shard, tokens) -> str:
    await self.ensure_shard(shard)
    tokens = await asyncio.get_running_loop().run_in_executor(self.executor, self.tokenizer.decode, tokens)
    return tokens
  
  async def load_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = safe_load(path)
    await asyncio.get_running_loop().run_in_executor(self.executor, load_state_dict, self.model, state_dict)
  
  async def save_checkpoint(self, shard: Shard, path: str):
    await self.ensure_shard(shard)
    state_dict = await asyncio.get_running_loop().run_in_executor(self.executor, get_state_dict, self.model)
    safe_save(state_dict, path) 
  
  async def infer_tensor(self, request_id: str, shard: Shard, input_data: np.ndarray, inference_state: Optional[dict] = None) -> tuple[np.ndarray, Optional[dict]]:
    await self.ensure_shard(shard)
    def wrap_infer():
      x = Tensor(input_data)
      h = self.model.embed(x)
      state = self.poll_state(h, request_id)
      out = self.model.forward(h, **state)
      self.states[request_id].start += x.shape[1]
      return out.numpy()
    output_data = await asyncio.get_running_loop().run_in_executor(self.executor, wrap_infer)
    return output_data, inference_state

  async def evaluate(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss):
    def step(x, y, l):
      Tensor.training = False
      return self.session['loss'](self.model, x, y, l)
    await self.ensure_shard(shard)
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths))
    out = score.numpy()
    return out
  
  async def train(self, request_id: str, shard: Shard, inputs, targets, lengths, loss=length_masked_ce_loss, opt=nn.optim.Adam, lr=1e-5):
    def step(x, y, l):
      Tensor.training = True
      score = self.session['loss'](self.model, x, y, l)
      self.session['opt'].zero_grad()
      score.backward()
      self.session['opt'].step()
      return score
    await self.ensure_shard(shard)
      
    score = await asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.session['jit'](Tensor(inputs), targets, lengths).realize())
    
    return loss.numpy(), loss.numpy()

  async def ensure_shard(self, shard: Shard):
    if self.shard == shard:
      return

    model_path = await self.shard_downloader.ensure_shard(shard, self.__class__.__name__)

    if self.shard != shard:
      loop = asyncio.get_running_loop()
      parameters = "1B" if "1b" in shard.model_id.lower() else "3B" if "3b" in shard.model_id.lower() else "8B" if "8b" in shard.model_id.lower() else "70B"
      
      # Clean up memory before starting
      force_memory_cleanup()
      
      # Print initial memory usage
      if DEBUG >= 1:
        print_gpu_memory_usage("Initial GPU memory usage before loading model")
      
      # Detect available GPU devices - ALWAYS use all available GPUs for large models
      from tinygrad import Device
      devices = None
      try:
        if Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
          import pynvml
          pynvml.nvmlInit()
          num_gpus = pynvml.nvmlDeviceGetCount()
          
          if num_gpus > 1:
            # For large models, ALWAYS use all available GPUs
            if parameters == "70B" or parameters == "8B" or parameters == "3B":
              devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(num_gpus))
              if DEBUG >= 1:
                print(f"Using ALL available GPUs ({num_gpus}) for {parameters} model: {devices}")
                
              # Print memory info for debugging
              for i in range(num_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if DEBUG >= 1:
                  print(f"GPU {i}: {mem_info.free / (1024**3):.2f}GB free of {mem_info.total / (1024**3):.2f}GB total")
          
          pynvml.nvmlShutdown()
      except Exception as e:
        if DEBUG >= 1:
          print(f"Error detecting multiple GPUs: {e}")
      
      # Free memory before loading model
      force_memory_cleanup()
      
      # Multiple attempts with different strategies if needed
      max_attempts = 3
      for attempt in range(max_attempts):
        try:
          if DEBUG >= 1 and attempt > 0:
            print(f"Retry attempt {attempt+1}/{max_attempts} loading model with devices: {devices}")
            print_gpu_memory_usage(f"GPU memory before attempt {attempt+1}")
          
          # Build the transformer with the detected devices
          model_shard = await loop.run_in_executor(self.executor, build_transformer, model_path, shard, parameters, devices)
          
          # If we get here, it worked
          tokenizer_path = str((model_path if model_path.is_dir() else model_path.parent))
          self.tokenizer = await resolve_tokenizer(tokenizer_path)
          self.shard = shard
          self.model = model_shard
          
          if DEBUG >= 1:
            print_gpu_memory_usage("GPU memory after successful model loading")
          
          return
        except MemoryError as e:
          if attempt == max_attempts - 1:
            # Last attempt failed, re-raise the exception
            raise
          
          if DEBUG >= 1:
            print(f"Memory error loading model (attempt {attempt+1}/{max_attempts}): {e}")
            print_gpu_memory_usage("GPU memory after failed attempt")
          
          # Try to free memory more aggressively
          force_memory_cleanup()
          
          # If we have multiple devices, make sure we're using them all
          if devices and len(devices) > 1:
            # We're already using multiple GPUs, try to be more aggressive with memory
            if DEBUG >= 1:
              print("Already using multiple GPUs, trying with more aggressive memory management")
          elif Device.DEFAULT == "CUDA" or Device.DEFAULT == "NV" or Device.DEFAULT == "GPU":
            # Try to use all available GPUs if we weren't already
            import pynvml
            pynvml.nvmlInit()
            num_gpus = pynvml.nvmlDeviceGetCount()
            if num_gpus > 1:
              devices = tuple(f"{Device.DEFAULT}:{i}" for i in range(num_gpus))
              if DEBUG >= 1:
                print(f"Switching to multi-GPU mode with devices: {devices}")
            pynvml.nvmlShutdown()
        except Exception as e:
          if attempt == max_attempts - 1:
            # Last attempt failed, re-raise the exception
            raise
          
          if DEBUG >= 1:
            print(f"Error loading model (attempt {attempt+1}/{max_attempts}): {e}")
            print_gpu_memory_usage("GPU memory after failed attempt")
          
          # Try to free memory
          force_memory_cleanup()
