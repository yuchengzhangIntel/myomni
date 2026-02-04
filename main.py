import os
import sys
import random
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
    "Qwen1.5-MoE-A2.7B"
]


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
        # === 回答问题4：防止切分关键层 ===
        # Qwen2MoeDecoderLayer 是 Qwen1.5-MoE 在 HF transformers 中的标准名称
        # 加上它，accelerate 就会保证这一层完整地放在同一张卡上
        no_split = ["LlamaDecoderLayer", "QuantLlamaDecoderLayer", "Qwen2MoeDecoderLayer", "MixtralDecoderLayer"]

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
        raise ValueError("Only support for opt/llama/Llama-2/falcon/mixtral now")

    # === [Added] WandB Visualization Logic ===
    if args.enable_wandb and len(results) > 0:
        try:
            import wandb

            # 1. Standard Logging (Line charts/history)
            wandb.log(results)

            # 2. Bar Chart Visualization (Summary)
            # Filter for numeric values only
            table_data = [[k, v] for k, v in results.items() if isinstance(v, (int, float))]

            # Create a WandB Table
            table = wandb.Table(data=table_data, columns=["Task Name", "Metric Value"])

            # Create and log the Bar Chart
            wandb.log({
                "final_evaluation_summary": wandb.plot.bar(
                    table, "Task Name", "Metric Value", title="Evaluation Metrics Summary"
                )
            })
            logger.info("Uploaded evaluation summary to WandB.")

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
    parser.add_argument("--abits", type=int, default=16)
    parser.add_argument("--group_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--let_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=10)
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
    parser.add_argument("--wandb_project", type=str, default="omniquant", help="Weights & Biases project name")
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
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    parser.add_argument("--train_shared_gate", default=False, action="store_true",
                        help="Train shared_expert_gate layers during calibration (for Qwen2-MoE)")
    parser.add_argument("--train_gate_lora", default=False, action="store_true",
                        help="Apply LoRA to mlp.gate (router) layers and train them (for MoE models)")
    parser.add_argument("--shared_gate_lr", type=float, default=1e-4,
                        help="Learning rate for shared_expert_gate training")
    parser.add_argument("--gate_lora_lr", type=float, default=1e-4, help="Learning rate for LoRA gate training")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank for gate training")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha (scaling factor) for gate training")
    
    # Router Calibration arguments (for Qwen2-MoE)
    parser.add_argument("--calibrate_router", default=False, action="store_true",
                        help="Enable Router Calibration using TopK-MSE loss for Qwen2-MoE")
    parser.add_argument("--router_lr", type=float, default=1e-3,
                        help="Learning rate for router calibration")
    parser.add_argument("--router_epochs", type=int, default=5,
                        help="Number of epochs for router calibration per layer")
    parser.add_argument("--k_loss", type=int, default=20,
                        help="TopK for loss calculation (number of experts to cache for TopK-MSE)")
    parser.add_argument("--k_routing", type=int, default=4,
                        help="TopK for expert shift metric (actual routing k used in the model)")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.lwc or args.let

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

    if args.enable_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except ImportError:
            logger.warning("WandB not installed but enable_wandb=True. Skipping WandB logging.")

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
        omniquant(
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
    evaluate(lm, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()