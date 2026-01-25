import transformers
import torch
from .models_utils import BaseLM, find_layers
from .lora_utils import (
    is_mixtral_model, 
    load_lora_weights, 
    replace_router_with_lora,
    freeze_base_model,
    get_lora_parameter_count,
    prepare_mixtral_for_lora_eval
)
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm
import os
import pdb


class LMClass(BaseLM):
    def __init__(self, args):

        super().__init__()

        self.args = args
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = args.model
        self.batch_size_per_gpu = args.batch_size

        self.model_config = args.model
        config = AutoConfig.from_pretrained(
            args.model, attn_implementation=args.attn_implementation
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        # self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=config.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print("vocab size: ", self.vocab_size)
        
        # Store LoRA modules if applied
        self.lora_modules = {}
        self.has_lora = False
        
        # Check if we need to load LoRA for Mixtral evaluation
        self._maybe_load_lora_for_eval(args)

    @property
    def eot_token(self) -> str:
        return self.tokenizer.eos_token

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        try:
            return self.gpt2.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self):
        print("max_gen_toks fn")
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_encode_batch(self, strings):
        return self.tokenizer(
            strings,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

    def tok_decode(self, tokens):
        return self.tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():

            return self.model(inps)["logits"]

    def model_batched_set(self, inps):
        dataset_logits = []
        for batch in inps:
            multi_logits = F.log_softmax(
                self._model_call(batch), dim=-1
            ).cpu()  # [batch, padding_length, vocab]
            dataset_logits.append(multi_logits)
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

    def _maybe_load_lora_for_eval(self, args):
        """
        Check if this is a Mixtral model and if LoRA checkpoint is provided.
        If so, automatically load LoRA adapters for router layers.
        
        This method:
        1. Detects if the model is Mixtral based on args.net or args.model
        2. Checks for lora_checkpoint_path in args or infers from output_dir
        3. Loads LoRA weights while keeping base model frozen
        """
        model_name = getattr(args, 'net', '') or args.model.split('/')[-1]
        
        if not is_mixtral_model(model_name):
            return
        
        print("Detected Mixtral model, checking for LoRA configuration...")
        
        # Determine LoRA checkpoint path
        checkpoint_path = None
        
        # Priority 1: Explicit lora_checkpoint_path
        if hasattr(args, 'lora_checkpoint_path') and args.lora_checkpoint_path:
            checkpoint_path = args.lora_checkpoint_path
        # Priority 2: Look in output_dir for omni_parameters.pth
        elif hasattr(args, 'output_dir') and args.output_dir:
            potential_path = os.path.join(args.output_dir, 'omni_parameters.pth')
            if os.path.exists(potential_path):
                checkpoint_path = potential_path
        # Priority 3: Look in resume path
        elif hasattr(args, 'resume') and args.resume:
            if os.path.isfile(args.resume):
                checkpoint_path = args.resume
            elif os.path.isdir(args.resume):
                potential_path = os.path.join(args.resume, 'omni_parameters.pth')
                if os.path.exists(potential_path):
                    checkpoint_path = potential_path
        
        # Get LoRA hyperparameters from args
        lora_rank = getattr(args, 'lora_rank', 8)
        lora_alpha = getattr(args, 'lora_alpha', 16.0)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading LoRA weights from: {checkpoint_path}")
            self.lora_modules = load_lora_weights(
                self.model,
                checkpoint_path,
                rank=lora_rank,
                lora_alpha=lora_alpha,
                device=None  # Will be moved to device later
            )
            self.has_lora = True
            
            # Ensure base model is frozen
            freeze_base_model(self.model)
            
            # Print LoRA statistics
            lora_count, total_count = get_lora_parameter_count(self.model)
            print(f"LoRA successfully loaded!")
            print(f"  - LoRA parameters: {lora_count:,}")
            print(f"  - Total parameters: {total_count:,}")
            print(f"  - LoRA ratio: {100 * lora_count / total_count:.6f}%")
        else:
            # Check if LoRA should be initialized without pretrained weights
            if hasattr(args, 'init_lora') and args.init_lora:
                print("Initializing LoRA adapters without pretrained weights...")
                self.lora_modules = replace_router_with_lora(
                    self.model,
                    rank=lora_rank,
                    lora_alpha=lora_alpha,
                    device=None
                )
                self.has_lora = True
                freeze_base_model(self.model)
                print(f"LoRA adapters initialized (rank={lora_rank}, alpha={lora_alpha})")
            else:
                print("No LoRA checkpoint found. Running evaluation without LoRA.")
    
    def load_lora_checkpoint(self, checkpoint_path: str, rank: int = 8, lora_alpha: float = 16.0):
        """
        Manually load a LoRA checkpoint after model initialization.
        Useful for evaluation scripts that want to load different checkpoints.
        
        Args:
            checkpoint_path: Path to the omni_parameters.pth file
            rank: LoRA rank (should match training configuration)
            lora_alpha: LoRA alpha (should match training configuration)
        """
        if not is_mixtral_model(self.model_name):
            print("Warning: LoRA loading is only supported for Mixtral models")
            return False
        
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return False
        
        print(f"Loading LoRA checkpoint from: {checkpoint_path}")
        
        # If LoRA modules already exist, just load weights
        if self.has_lora and self.lora_modules:
            # Reload weights into existing modules
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            loaded_count = 0
            
            for layer_idx, layer_params in checkpoint.items():
                if not isinstance(layer_params, dict):
                    continue
                for param_name, param_value in layer_params.items():
                    if 'lora_A' not in param_name and 'lora_B' not in param_name:
                        continue
                    
                    for module_name, lora_module in self.lora_modules.items():
                        import re
                        match = re.search(r'layers\.(\d+)\.', module_name)
                        if match and int(match.group(1)) == layer_idx:
                            if 'lora_A' in param_name:
                                lora_module.lora_A.data = param_value.to(lora_module.lora_A.device)
                                loaded_count += 1
                            elif 'lora_B' in param_name:
                                lora_module.lora_B.data = param_value.to(lora_module.lora_B.device)
                                loaded_count += 1
                            break
            
            print(f"Loaded {loaded_count} LoRA parameters")
        else:
            # First time loading - replace layers and load weights
            self.lora_modules = load_lora_weights(
                self.model,
                checkpoint_path,
                rank=rank,
                lora_alpha=lora_alpha,
                device=None
            )
            self.has_lora = True
            freeze_base_model(self.model)
        
        return True
