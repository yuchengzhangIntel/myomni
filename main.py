import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

os.environ["windows_host"] = "http://child-prc.intel.com"
os.environ["HTTP_PROXY"] = f"{os.environ['windows_host']}:913"
os.environ["ALL_PROXY"] = f"{os.environ['windows_host']}:913"
os.environ["http_proxy"] = os.environ["HTTP_PROXY"]
os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"]
os.environ["https_proxy"] = os.environ["HTTP_PROXY"]
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import sys
import random
from numbers import Number
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils import get_balanced_memory

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

import pdb

torch.backends.cudnn.benchmark = True


def get_max_memory_map(ratio=0.95):
    if ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be in (0, 1]")
    max_memory = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            max_memory[i] = int(total_memory * ratio)
    return max_memory


net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-2.7b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    "llama-7b",
    "llama-13b",
    "llama-30b",
    "llama-65b",
    "Llama-2-7b",
    "Llama-2-13b",
    "Llama-2-70b",
    "Llama-2-7b-chat",
    "Llama-2-13b-chat",
    "llava-llama-2-13b-chat-lightning-preview",
    "falcon-180b",
    "falcon-7b",
    "mixtral-8x7b",
    "deepseek-moe-16b-base",
    "Qwen1.5-MoE-A2.7B",
    "Qwen3-30B-A3B-Base"
]


def get_no_split_module_classes(net_name):
    classes = ["LlamaDecoderLayer", "QuantLlamaDecoderLayer", "MixtralDecoderLayer"]
    net_name = (net_name or "").lower()

    if "qwen" in net_name:
        classes.extend([
            "Qwen2MoeDecoderLayer",
            "Qwen3MoeDecoderLayer",
        ])

    return classes


@torch.no_grad()
def evaluate(lm, args, logger):
    results = {}

    # === 1. GPU / 并行策略 ===
    if getattr(args, 'parallelize', False) and args.multigpu:
        raise ValueError("Cannot use both --parallelize and --multigpu")

    if args.multigpu:
        # 手动多卡映射逻辑 (保持不变)
        if "opt" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)
            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "llama" in args.net.lower() or "vicuna" in args.net.lower() or "mixtral" in args.net.lower() or "qwen" in args.net.lower() or "deepseek" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.model.embed_tokens.to(input_device)
            lm.model.model.norm.to(output_device)
            lm.model.lm_head.to(output_device)
        elif "falcon" in args.net.lower():
            map_layers_to_multi_gpus(lm.model.transformer.h)
            input_device = lm.model.transformer.h[0].device
            output_device = lm.model.transformer.h[-1].device
            assert input_device == output_device
            lm._device = input_device
            lm.model.transformer.word_embeddings.to(input_device)
            lm.model.transformer.ln_f.to(output_device)
            lm.model.lm_head.to(output_device)

    elif getattr(args, 'parallelize', False):
        # Prevent decoder layers from being split across devices when using accelerate.
        no_split = get_no_split_module_classes(args.net)

        balanced_mem = get_balanced_memory(
            lm.model,
            max_memory=get_max_memory_map(0.95),
            no_split_module_classes=no_split
        )
        logger.info(f"Auto-balancing memory: {balanced_mem}")
        device_map = infer_auto_device_map(
            lm.model,
            max_memory=balanced_mem,
            no_split_module_classes=no_split
        )
        lm.model = dispatch_model(lm.model, device_map=device_map)

    else:
        # 单卡逻辑
        if "opt" in args.net.lower():
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower() or "mixtral" in args.net.lower() or "deepseek" in args.net.lower():
            lm.model = lm.model.to(lm.device)
        elif "falcon" in args.net.lower():
            lm.model.transformer = lm.model.transformer.to(lm.device)

    # === 2. PPL 评测 (计算但不保存CSV) ===
    if args.eval_ppl:
        # for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
        for dataset in ["wikitext2", "c4"]:
            cache_testloader = f'{args.cache_dir}/testloader_{args.model_family}_{dataset}_all.cache'
            if os.path.exists(cache_testloader):
                testloader = torch.load(cache_testloader, weights_only=False)
                logger.info(f"load calibration from {cache_testloader}")
            else:
                dataloader, testloader = get_loaders(
                    dataset,
                    seed=args.seed,
                    model=args.model,
                    seqlen=lm.seqlen,
                )
                torch.save(testloader, cache_testloader)
            if "c4" in dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []
            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(lm.device)
                if "opt" in args.net.lower():
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.net.lower() or "mixtral" in args.net.lower() or "deepseek" in args.net.lower() or "qwen" in args.net.lower():
                    outputs = lm.model.model(batch)
                elif "falcon" in args.model:
                    outputs = lm.model.transformer(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            logger.info(f'{dataset} : {ppl.item()}')
            lm.model.config.use_cache = use_cache
            results[dataset] = ppl.item()

    # === 3. 下游任务评测 (LM Eval) & CSV 保存 ===
    if args.tasks != "":
        task_list = args.tasks.split(",") if isinstance(args.tasks, str) else args.tasks

        import lm_eval
        from lm_eval.models.huggingface import HFLM

        try:
            task_manager = lm_eval.tasks.TaskManager(include_path="./datasets_local/lm_eval_configs/tasks",
                                                     include_defaults=True)
        except Exception:
            task_manager = lm_eval.tasks.TaskManager(include_defaults=True)

        # === 回答问题2：处理 batch size ===
        # 优先使用 lm_eval_batch_size，如果不存在或为None，则使用 'auto'
        # HFLM 支持 batch_size='auto' (自动寻找最大batch size)
        eval_batch_size = getattr(args, 'lm_eval_batch_size', 'auto')
        if eval_batch_size is None:
            eval_batch_size = 'auto'

        print(f"Initializing HFLM with batch_size={eval_batch_size}...")

        hflm = HFLM(pretrained=lm.model, tokenizer=lm.tokenizer, batch_size=eval_batch_size)

        t_results = lm_eval.simple_evaluate(
            model=hflm,
            tasks=task_list,
            batch_size=eval_batch_size,
            task_manager=task_manager,
            gen_kwargs=args.gen_kwargs,
        )['results']

        metric_vals = {}
        for task, result in t_results.items():
            metric_vals[task] = round(result.get('acc_norm,none', result.get('acc,none', 0)), 4)

        logger.info(f"Task Results: {metric_vals}")
        pprint(metric_vals)
        results.update(metric_vals)

        # === 4. CSV 保存逻辑 (仅在跑了 Task 时触发) ===
        # 过滤 metric_vals，只保留需要写入 CSV 的数据
        reported_metric_vals = {}
        for k, v in metric_vals.items():
            if "mmlu" in k:
                if k == "mmlu":
                    reported_metric_vals[k] = v
            else:
                reported_metric_vals[k] = v

        import pandas as pd
        csv_path = f"{args.output_dir}/results.csv"

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            new_df = pd.DataFrame(reported_metric_vals, index=[0])
            # 补齐列
            for col in new_df.columns:
                if col not in df.columns:
                    df[col] = None
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(reported_metric_vals, index=[0])

        # 计算 Average 指标
        if len(task_list) >= 5:
            cols = ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande']
            if all(c in df.columns for c in cols):
                df["avg-5"] = df[cols].mean(axis=1)
        if len(task_list) >= 6:
            cols = ['piqa', 'arc_easy', 'arc_challenge', 'hellaswag', 'winogrande', 'boolq']
            if all(c in df.columns for c in cols):
                df["avg-6"] = df[cols].mean(axis=1)

        logger.info(f"Saving task results to {csv_path}...")
        logger.info(df)
        df.to_csv(csv_path, index=False)

    model = lm.model
    if "llama" in args.net.lower() or "vicuna" in args.net.lower() or "qwen" in args.net.lower():
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    elif "opt" in args.net.lower():
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif 'falcon' in args.model:
        model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    else:
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral/qwen/deepseek now")

    # === [Added] WandB Visualization Logic ===
    if args.enable_wandb and len(results) > 0:
        try:
            import wandb

            # 1. Group metrics by type for organized visualization
            # PPL tasks (wikitext2, c4) -> eval_ppl/*
            # Other tasks -> eval_task/*
            ppl_tasks = {"wikitext2", "c4", "ptb", "ptb-new", "c4-new"}
            grouped_results = {}
            for k, v in results.items():
                if isinstance(v, (int, float)):
                    if k.lower() in ppl_tasks:
                        grouped_results[f"eval_ppl/{k}"] = v
                    else:
                        grouped_results[f"eval_task/{k}"] = v
            
            # Log grouped metrics (separate charts for PPL and Tasks)
            wandb.log(grouped_results)

            # 2. Bar Chart Visualization (Summary) - PPL tasks
            ppl_data = [[k.replace("eval_ppl/", ""), v] for k, v in grouped_results.items() if k.startswith("eval_ppl/")]
            if ppl_data:
                ppl_table = wandb.Table(data=ppl_data, columns=["Dataset", "PPL"])
                wandb.log({
                    "eval_ppl/summary_chart": wandb.plot.bar(
                        ppl_table, "Dataset", "PPL", title="PPL Evaluation (wikitext2, c4)"
                    )
                })
            
            # 3. Bar Chart Visualization (Summary) - Task results
            task_data = [[k.replace("eval_task/", ""), v] for k, v in grouped_results.items() if k.startswith("eval_task/")]
            if task_data:
                task_table = wandb.Table(data=task_data, columns=["Task", "Accuracy"])
                wandb.log({
                    "eval_task/summary_chart": wandb.plot.bar(
                        task_table, "Task", "Accuracy", title="Task Evaluation Summary"
                    )
                })
            
            logger.info("Uploaded evaluation summary to WandB (grouped by eval_ppl and eval_task).")

        except Exception as e:
            logger.warning(f"Failed to upload results to wandb: {e}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str,
                        help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="real quantization, which can see memory reduce. Note that due to the limitations of AutoGPTQ kernels, the real quantization of weight-only quantization can only lead memory reduction, but with slower inference speed.")
    parser.add_argument("--calib_dataset", type=str, default="wikitext2",
                        choices=["wikitext2", "ptb", "c4", "mix", "pile"],
                        help="Where to extract calibration data from.",
                        )
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "Generation kwargs for lm-eval generate_until tasks, e.g. "
            "temperature=0.6,top_p=0.95,top_k=20,min_p=0,do_sample=True"
        ),
    )
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--attn_wbits", type=int, default=None,
                        help="Override wbits for attention linear layers (e.g. q/k/v/o proj).")
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=None,
                        help="Enable gradient clipping with the provided max norm; disabled when omitted")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--attn_epochs", type=int, default=None,
                        help="Base epoch count for decoupled attention training; defaults to --epochs and increases by 1 every 2 layers")
    parser.add_argument("--let", default=False, action="store_true",
                        help="activate learnable equivalent transformation")
    parser.add_argument("--lwc", default=False, action="store_true", help="activate learnable weight clipping")
    parser.add_argument("--aug_loss", default=False, action="store_true",
                        help="calculate additional loss with same input")
    parser.add_argument("--symmetric", default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--disable_zero_point", default=False, action="store_true",
                        help="quantization without zero_point")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--multigpu", action="store_true", help="at eval, map model to multiple gpus")
    parser.add_argument("--lm_eval_batch_size", type=str, default="auto",
                        help="Batch size for lm-eval tasks. Can be an integer or 'auto'.")
    parser.add_argument("--enable_wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="omniquant-moe", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--parallelize", action="store_true",
                        help="auto device_map with Accelerate; incompatible with --multigpu")
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        "--net",
        type=str,
        default=None,
        help=(
            "Model family or variant name. Known examples: "
            + ", ".join(net_choices)
            + ". New variants such as Qwen3 can also be passed directly."
        ),
    )
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--train_shared_gate", default=False, action="store_true",
                        help="Train shared_expert_gate layers during calibration (only applies to Qwen MoE variants that expose shared_expert_gate)")
    parser.add_argument("--train_gate_lora", default=False, action="store_true",
                        help="Apply LoRA to mlp.gate (router) layers and train them (for MoE models)")
    parser.add_argument("--shared_gate_lr", type=float, default=1e-5,
                        help="Learning rate for shared_expert_gate updates")
    parser.add_argument("--gate_lora_lr", type=float, default=1e-5,
                        help="Learning rate for router gate updates (or gate LoRA when supported)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank for gate training")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha (scaling factor) for gate training")
    parser.add_argument("--use_linear_lora", default=False, action="store_true",
                        help="Enable LoRA on QuantLinear layers during block-wise quantization")
    parser.add_argument("--linear_lora_r", type=int, default=16,
                        help="LoRA rank for QuantLinear layers")
    parser.add_argument("--linear_lora_alpha", type=float, default=16.0,
                        help="LoRA alpha for QuantLinear layers")
    parser.add_argument("--linear_lora_lr", type=float, default=1e-4,
                        help="Learning rate for QuantLinear LoRA parameters")
    
    # Router Calibration arguments (for Qwen MoE models)
    parser.add_argument("--calibrate_router", default=False, action="store_true",
                        help="Enable Router Calibration using TopK-MSE loss for Qwen MoE models")
    parser.add_argument("--router_lr", type=float, default=1e-2,
                        help="Learning rate for router calibration (default 1e-2, higher than LWC)")
    parser.add_argument("--router_epochs", type=int, default=5,
                        help="Number of epochs for router calibration per layer")
    parser.add_argument("--k_loss", type=int, default=20,
                        help="TopK for loss calculation (number of experts to cache for TopK-MSE)")
    parser.add_argument("--k_routing", type=int, default=4,
                        help="TopK for expert shift metric (actual routing k used in the model)")
    parser.add_argument("--quant_routing_top_n", type=int, default=None,
                        help="Top-N experts to cache for MoE self-supervision; defaults to the layer routing top-k")
    parser.add_argument("--use_router_weight_in_loss", default=False, action="store_true",
                        help="Weight each token-expert self-supervision loss by the normalized FP16 router probability")
    parser.add_argument("--block_eval_interval", type=int, default=0,
                        help="Run student-only block-wise evaluation every N MoE epochs; disabled when < 1")
    parser.add_argument("--block_loss_attn", default=False, action="store_true",
                        help="Allow in-step block-wise loss to update attention LWC/LoRA parameters")
    parser.add_argument("--block_loss_router", default=False, action="store_true",
                        help="Allow in-step block-wise loss to update router and shared-gate parameters")
    parser.add_argument("--block_loss_expert", default=False, action="store_true",
                        help="Allow in-step block-wise loss to update routed/shared expert parameters")
    parser.add_argument("--expert_loss_attn", default=False, action="store_true",
                        help="Allow dynamic expert self-supervision loss to update attention LWC/LoRA parameters")
    parser.add_argument("--enable_block_loss_update", dest="enable_block_loss_update", default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_update_epochs", type=int, default=1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_update_attn", dest="block_loss_attn", default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_update_router", dest="block_loss_router", default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_update_expert", dest="block_loss_expert", default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_aux_loss", default=False, action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--block_aux_loss_weight", type=float, default=0.1,
                        help=argparse.SUPPRESS)
    parser.add_argument("--max_train_layers", type=int, default=-1,
                        help="Train at most the first N layers; use -1 for all layers, 1 for first block only")

    args = parser.parse_args()
    if args.attn_epochs is None:
        args.attn_epochs = args.epochs
    if args.epochs < 0:
        raise ValueError("--epochs must be non-negative")
    if args.attn_epochs < 0:
        raise ValueError("--attn_epochs must be non-negative")
    if args.max_grad_norm is not None and args.max_grad_norm <= 0:
        raise ValueError("--max_grad_norm must be positive when provided")
    if args.block_update_epochs < 0:
        raise ValueError("--block_update_epochs must be non-negative")
    if args.block_aux_loss_weight < 0:
        raise ValueError("--block_aux_loss_weight must be non-negative")
    if args.max_train_layers < -1:
        raise ValueError("--max_train_layers must be -1 or a non-negative integer")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    args._deprecated_cli_flags = []
    if getattr(args, "enable_block_loss_update", False):
        args._deprecated_cli_flags.append("--enable_block_loss_update")
    if getattr(args, "block_update_epochs", 1) != 1:
        args._deprecated_cli_flags.append("--block_update_epochs")
    if getattr(args, "block_aux_loss", False):
        args._deprecated_cli_flags.append("--block_aux_loss")
    if float(getattr(args, "block_aux_loss_weight", 0.1)) != 0.1:
        args._deprecated_cli_flags.append("--block_aux_loss_weight")

    if args.epochs > 0 or args.attn_epochs > 0:
        has_any_trainable_mechanism = any([
            args.lwc,
            args.let,
            args.use_linear_lora,
            args.train_gate_lora,
            args.train_shared_gate,
            args.calibrate_router,
            args.block_loss_attn,
            args.block_loss_router,
            args.block_loss_expert,
            args.expert_loss_attn,
        ])
        if not has_any_trainable_mechanism:
            raise ValueError(
                "Training epochs are set, but no trainable mechanism is enabled. "
                "Enable at least one of --lwc, --let, --use_linear_lora, --train_gate_lora, "
                "--train_shared_gate, --block_loss_attn, --block_loss_router, --block_loss_expert, "
                "--expert_loss_attn, or --calibrate_router."
            )

    if args.use_linear_lora and args.let:
        raise ValueError("--use_linear_lora is not supported together with --let in this implementation")

    if args.use_linear_lora and args.resume:
        raise ValueError("--use_linear_lora does not currently support --resume")

    if args.use_linear_lora and args.linear_lora_r <= 0:
        raise ValueError("--linear_lora_r must be positive when --use_linear_lora is enabled")

    effective_net_name = (args.net or args.model.split('/')[-1]).lower()
    if ("qwen" in effective_net_name or "deepseek" in effective_net_name) and args.let:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --let")
    if ("qwen" in effective_net_name or "deepseek" in effective_net_name) and args.train_gate_lora:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --train_gate_lora without an explicit router loss")
    if ("qwen" in effective_net_name or "deepseek" in effective_net_name) and args.train_shared_gate:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --train_shared_gate without an explicit shared-gate loss")
    if ("qwen" in effective_net_name or "deepseek" in effective_net_name) and args.calibrate_router:
        raise ValueError("Decoupled Qwen/DeepSeek MoE training does not support --calibrate_router")

    if (args.wbits < 16 and args.wbits >= 8) or (args.abits < 16 and args.abits >= 8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    if args._deprecated_cli_flags:
        logger.warning(
            "Deprecated decoupled MoE CLI flags are ignored in the Qwen/DeepSeek joint training path: %s",
            ", ".join(sorted(set(args._deprecated_cli_flags))),
        )

    # === Training Config Summary (Begin) ===
    logger.info("=" * 60)
    logger.info("Training Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"  Total Epochs              : {args.epochs}")
    logger.info(f"  Attention Epochs Base     : {args.attn_epochs}")
    logger.info(f"  Router Calibration        : {'ON' if args.calibrate_router else 'OFF'}"
                + (f"  (lr={args.router_lr}, router_epochs={args.router_epochs})" if args.calibrate_router else ""))
    logger.info(f"  Train Gate LoRA           : {'ON' if args.train_gate_lora else 'OFF'}"
                + (f"  (lr={args.gate_lora_lr})" if args.train_gate_lora else ""))
    logger.info(f"  Linear Quant LoRA         : {'ON' if args.use_linear_lora else 'OFF'}"
                + (f"  (r={args.linear_lora_r}, alpha={args.linear_lora_alpha}, lr={args.linear_lora_lr})" if args.use_linear_lora else ""))
    logger.info(f"  Train Shared Gate         : {'ON' if args.train_shared_gate else 'OFF'}"
                + (f"  (lr={args.shared_gate_lr})" if args.train_shared_gate else ""))
    logger.info(f"  MoE Quant Routing Top-N   : {args.quant_routing_top_n if args.quant_routing_top_n is not None else 'layer top-k'}")
    logger.info(f"  Router Weight In Loss     : {'ON' if args.use_router_weight_in_loss else 'OFF'}")
    logger.info(f"  Block Eval Interval       : {args.block_eval_interval} ({'OFF' if args.block_eval_interval < 1 else 'ON'})")
    logger.info(f"  Block Loss Attention      : {'ON' if args.block_loss_attn else 'OFF'}")
    logger.info(f"  Block Loss Router/Gates   : {'ON' if args.block_loss_router else 'OFF'}")
    logger.info(f"  Block Loss Experts        : {'ON' if args.block_loss_expert else 'OFF'}")
    logger.info(f"  Expert Loss Attention     : {'ON' if args.expert_loss_attn else 'OFF'}")
    logger.info(f"  Max Train Layers          : {args.max_train_layers if args.max_train_layers >= 0 else 'ALL'}")
    logger.info("=" * 60)

    if args.enable_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args)
            )
            logger.info(f"WandB initialized: project={args.wandb_project}, run={args.wandb_run_name or 'auto'}")
        except ImportError:
            logger.warning("WandB not installed. Disabling WandB logging. Install with: pip install wandb")
            args.enable_wandb = False

    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    # assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    lm = LMClass(args)
    lm.seqlen = 2048
    lm.model.eval()
    for param in lm.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": args.group_size,
        "lwc": args.lwc,
        "disable_zero_point": args.disable_zero_point
    }
    if args.attn_wbits is not None:
        args.attn_weight_quant_params = {
            **args.weight_quant_params,
            "n_bits": args.attn_wbits
        }
    else:
        args.attn_weight_quant_params = None
    args.act_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.q_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "dynamic_method": args.a_dynamic_method,
    }
    args.p_quant_params = {
        "n_bits": 16,
        "metric": "fix0to1",
    }

    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        logger.info(f"set quantization in gpu {gpu_id}")

    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    final_loss = None
    if args.wbits < 16 or args.abits < 16:
        logger.info("=== start quantization ===")
        tick = time.time()
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader, weights_only=False)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
            )
            torch.save(dataloader, cache_dataloader)
        act_scales = None
        act_shifts = None
        if args.let:
            act_scales = torch.load(args.act_scales, weights_only=False)
            act_shifts = torch.load(args.act_shifts, weights_only=False)
        _, final_loss = omniquant(
            lm,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
            train_shared_gate=args.train_shared_gate,
            train_gate_lora=args.train_gate_lora,
            shared_gate_lr=args.shared_gate_lr,
            gate_lora_lr=args.gate_lora_lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # Router Calibration parameters
            calibrate_router=args.calibrate_router,
            router_lr=args.router_lr,
            router_epochs=args.router_epochs,
            k_loss=args.k_loss,
            k_routing=args.k_routing,
            quant_routing_top_n=args.quant_routing_top_n,
            use_router_weight_in_loss=args.use_router_weight_in_loss,
        )
        logger.info(time.time() - tick)
    if args.save_dir:
        # delete omni parameters
        for name, module in lm.model.named_modules():
            if isinstance(module, QuantLinear):
                del module.weight_quantizer.lowbound_factor
                del module.weight_quantizer.upbound_factor
            if isinstance(module, QuantLlamaDecoderLayer) or isinstance(module, QuantOPTDecoderLayer):
                if args.let:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.out_smooth_scale
                    del module.out_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift
        lm.model.save_pretrained(args.save_dir)
        lm.tokenizer.save_pretrained(args.save_dir)
    results = evaluate(lm, args, logger)

    # === Training & Evaluation Summary (End) ===
    logger.info("=" * 60)
    logger.info("Final Summary")
    logger.info("=" * 60)
    logger.info(f"  Total Epochs              : {args.epochs}")
    logger.info(f"  Attention Epochs Base     : {args.attn_epochs}")
    logger.info(f"  Router Calibration        : {'ON' if args.calibrate_router else 'OFF'}"
                + (f"  (lr={args.router_lr}, router_epochs={args.router_epochs})" if args.calibrate_router else ""))
    logger.info(f"  Train Gate LoRA           : {'ON' if args.train_gate_lora else 'OFF'}"
                + (f"  (lr={args.gate_lora_lr})" if args.train_gate_lora else ""))
    logger.info(f"  Linear Quant LoRA         : {'ON' if args.use_linear_lora else 'OFF'}"
                + (f"  (r={args.linear_lora_r}, alpha={args.linear_lora_alpha}, lr={args.linear_lora_lr})" if args.use_linear_lora else ""))
    logger.info(f"  Train Shared Gate         : {'ON' if args.train_shared_gate else 'OFF'}"
                + (f"  (lr={args.shared_gate_lr})" if args.train_shared_gate else ""))
    logger.info(f"  MoE Quant Routing Top-N   : {args.quant_routing_top_n if args.quant_routing_top_n is not None else 'layer top-k'}")
    logger.info(f"  Router Weight In Loss     : {'ON' if args.use_router_weight_in_loss else 'OFF'}")
    logger.info(f"  Block Eval Interval       : {args.block_eval_interval} ({'OFF' if args.block_eval_interval < 1 else 'ON'})")
    logger.info(f"  Block Loss Attention      : {'ON' if args.block_loss_attn else 'OFF'}")
    logger.info(f"  Block Loss Router/Gates   : {'ON' if args.block_loss_router else 'OFF'}")
    logger.info(f"  Block Loss Experts        : {'ON' if args.block_loss_expert else 'OFF'}")
    logger.info(f"  Expert Loss Attention     : {'ON' if args.expert_loss_attn else 'OFF'}")
    logger.info(f"  Max Train Layers          : {args.max_train_layers if args.max_train_layers >= 0 else 'ALL'}")
    logger.info(f"  Final Loss                : {final_loss if final_loss is not None else 'N/A'}")
    # PPL results
    wiki2_ppl = results.get('wikitext2', 'N/A')
    c4_ppl = results.get('c4', 'N/A')
    logger.info(f"  Wikitext2 PPL             : {wiki2_ppl}")
    logger.info(f"  C4 PPL                    : {c4_ppl}")
    # Task evaluation results
    ppl_keys = {"wikitext2", "c4", "ptb", "ptb-new", "c4-new"}
    task_keys = [k for k in results if k not in ppl_keys]
    # Only include numeric task results when computing the mean
    task_scores = [v for k, v in results.items() if k in task_keys and isinstance(v, (int, float))]
    task_score_mean = round(sum(task_scores) / len(task_scores), 4) if task_scores else "N/A"
    logger.info(f"  Task Score Mean           : {task_score_mean}")
    if task_keys:
        logger.info("  Task Evaluation Results:")
        for k in task_keys:
            logger.info(f"    {k:25s}: {results[k]}")
    else:
        logger.info("  Task Evaluation Results    : N/A")
    logger.info("=" * 60)


if __name__ == "__main__":
    print(sys.argv)
    main()
