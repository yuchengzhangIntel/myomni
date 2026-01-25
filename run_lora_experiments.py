#!/usr/bin/env python
"""
Batch experiment script for LoRA training and evaluation on Mixtral models.

This script reads experiment configurations from a YAML or JSON file and:
1. Runs training (main.py) for each configuration serially
2. Runs evaluation for each generated checkpoint
3. Aggregates all results into a CSV file

Usage:
    python run_lora_experiments.py --config config/lora_experiments.yaml
    python run_lora_experiments.py --config config/lora_experiments.json --output results.csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import shutil


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML or JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required to read YAML configs. Install with: pip install pyyaml")
        elif config_path.endswith('.json'):
            return json.load(f)
        else:
            # Try YAML first, then JSON
            content = f.read()
            try:
                import yaml
                return yaml.safe_load(content)
            except:
                return json.loads(content)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_lr: float = 5e-3
    let_lr: float = 5e-3
    lwc_lr: float = 1e-2
    epochs: int = 10
    batch_size: int = 1
    nsamples: int = 128
    wbits: int = 4
    abits: int = 16
    group_size: Optional[int] = 128
    seed: int = 2
    lwc: bool = True
    let: bool = False
    aug_loss: bool = False
    extra_args: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}


@dataclass
class GlobalConfig:
    """Global configuration shared across experiments."""
    model_path: str
    net: str
    output_base_dir: str = "./experiments"
    cache_dir: str = "./cache"
    calib_dataset: str = "wikitext2"
    eval_ppl: bool = True
    tasks: str = ""
    real_quant: bool = False
    multigpu: bool = False
    attn_implementation: str = "eager"
    act_scales: Optional[str] = None
    act_shifts: Optional[str] = None


@dataclass
class ExperimentResult:
    """Results from a single experiment."""
    experiment_name: str
    config: Dict
    train_time: float
    eval_time: float
    checkpoint_path: str
    ppl_wikitext2: Optional[float] = None
    ppl_c4: Optional[float] = None
    task_results: Dict[str, float] = field(default_factory=dict)
    status: str = "success"
    error_message: str = ""


class ExperimentRunner:
    """Runs batch LoRA experiments."""
    
    def __init__(
        self,
        global_config: GlobalConfig,
        experiments: List[ExperimentConfig],
        output_csv: str = "experiment_results.csv",
        dry_run: bool = False,
        skip_training: bool = False,
        skip_evaluation: bool = False,
        verbose: bool = True
    ):
        self.global_config = global_config
        self.experiments = experiments
        self.output_csv = output_csv
        self.dry_run = dry_run
        self.skip_training = skip_training
        self.skip_evaluation = skip_evaluation
        self.verbose = verbose
        self.results: List[ExperimentResult] = []
        
        # Create output directory
        os.makedirs(global_config.output_base_dir, exist_ok=True)
    
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def build_training_command(self, exp: ExperimentConfig) -> List[str]:
        """Build the training command for an experiment."""
        exp_output_dir = os.path.join(self.global_config.output_base_dir, exp.name, "logs")
        exp_save_dir = os.path.join(self.global_config.output_base_dir, exp.name, "checkpoint")
        
        cmd = [
            sys.executable, "main.py",
            "--model", self.global_config.model_path,
            "--net", self.global_config.net,
            "--output_dir", exp_output_dir,
            "--save_dir", exp_save_dir,
            "--cache_dir", self.global_config.cache_dir,
            "--calib_dataset", self.global_config.calib_dataset,
            "--lora_rank", str(exp.lora_rank),
            "--lora_alpha", str(exp.lora_alpha),
            "--lora_lr", str(exp.lora_lr),
            "--let_lr", str(exp.let_lr),
            "--lwc_lr", str(exp.lwc_lr),
            "--epochs", str(exp.epochs),
            "--batch_size", str(exp.batch_size),
            "--nsamples", str(exp.nsamples),
            "--wbits", str(exp.wbits),
            "--abits", str(exp.abits),
            "--seed", str(exp.seed),
            "--attn_implementation", self.global_config.attn_implementation,
        ]
        
        if exp.group_size is not None:
            cmd.extend(["--group_size", str(exp.group_size)])
        
        if exp.lwc:
            cmd.append("--lwc")
        if exp.let:
            cmd.append("--let")
        if exp.aug_loss:
            cmd.append("--aug_loss")
        
        if self.global_config.act_scales:
            cmd.extend(["--act-scales", self.global_config.act_scales])
        if self.global_config.act_shifts:
            cmd.extend(["--act-shifts", self.global_config.act_shifts])
        
        # Add extra arguments
        for key, value in exp.extra_args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", str(value)])
        
        return cmd
    
    def build_evaluation_command(self, exp: ExperimentConfig, checkpoint_path: str) -> List[str]:
        """Build the evaluation command for an experiment."""
        exp_output_dir = os.path.join(self.global_config.output_base_dir, exp.name, "eval_logs")
        
        cmd = [
            sys.executable, "main.py",
            "--model", self.global_config.model_path,
            "--net", self.global_config.net,
            "--output_dir", exp_output_dir,
            "--cache_dir", self.global_config.cache_dir,
            "--lora_rank", str(exp.lora_rank),
            "--lora_alpha", str(exp.lora_alpha),
            "--lora_checkpoint_path", checkpoint_path,
            "--batch_size", str(exp.batch_size),
            "--seed", str(exp.seed),
            "--attn_implementation", self.global_config.attn_implementation,
            "--wbits", str(exp.wbits),
            "--abits", str(exp.abits),
            "--epochs", "0",  # No training, just evaluation
        ]
        
        if exp.group_size is not None:
            cmd.extend(["--group_size", str(exp.group_size)])
        
        if self.global_config.eval_ppl:
            cmd.append("--eval_ppl")
        
        if self.global_config.tasks:
            cmd.extend(["--tasks", self.global_config.tasks])
        
        if self.global_config.multigpu:
            cmd.append("--multigpu")
        
        return cmd
    
    def run_command(self, cmd: List[str], description: str) -> tuple:
        """Run a command and return (success, output, elapsed_time)."""
        self.log(f"Running: {description}")
        self.log(f"Command: {' '.join(cmd)}")
        
        if self.dry_run:
            self.log("(Dry run - command not executed)")
            return True, "", 0.0
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)) or "."
            )
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                self.log(f"Command failed with return code {result.returncode}", "ERROR")
                self.log(f"STDERR: {result.stderr}", "ERROR")
                return False, result.stderr, elapsed
            
            return True, result.stdout, elapsed
        except Exception as e:
            elapsed = time.time() - start_time
            self.log(f"Exception running command: {e}", "ERROR")
            return False, str(e), elapsed
    
    def parse_evaluation_output(self, output: str, log_dir: str) -> Dict[str, float]:
        """Parse evaluation output to extract metrics."""
        metrics = {}
        
        # Try to parse from output
        for line in output.split('\n'):
            line = line.strip()
            # Parse PPL results like "wikitext2 : 5.123"
            if ':' in line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    try:
                        value = float(parts[1].strip())
                        if 'wikitext' in key or 'c4' in key or 'ptb' in key:
                            metrics[f"ppl_{key}"] = value
                    except ValueError:
                        pass
        
        # Also try to read from log files
        if os.path.exists(log_dir):
            for log_file in os.listdir(log_dir):
                if log_file.startswith('log_'):
                    log_path = os.path.join(log_dir, log_file)
                    try:
                        with open(log_path, 'r') as f:
                            for line in f:
                                if ':' in line and ('wikitext2' in line.lower() or 'c4' in line.lower()):
                                    parts = line.split(':')
                                    if len(parts) >= 2:
                                        key = parts[-2].strip().split()[-1].lower()
                                        try:
                                            value = float(parts[-1].strip())
                                            metrics[f"ppl_{key}"] = value
                                        except ValueError:
                                            pass
                    except Exception:
                        pass
        
        return metrics
    
    def run_experiment(self, exp: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment (training + evaluation)."""
        self.log(f"=" * 60)
        self.log(f"Starting experiment: {exp.name}")
        self.log(f"Config: rank={exp.lora_rank}, alpha={exp.lora_alpha}, lr={exp.lora_lr}")
        self.log(f"=" * 60)
        
        exp_dir = os.path.join(self.global_config.output_base_dir, exp.name)
        checkpoint_dir = os.path.join(exp_dir, "checkpoint")
        checkpoint_path = os.path.join(exp_dir, "logs", "omni_parameters.pth")
        
        result = ExperimentResult(
            experiment_name=exp.name,
            config=asdict(exp),
            train_time=0.0,
            eval_time=0.0,
            checkpoint_path=checkpoint_path
        )
        
        # Training phase
        if not self.skip_training:
            os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            train_cmd = self.build_training_command(exp)
            success, output, elapsed = self.run_command(train_cmd, f"Training {exp.name}")
            result.train_time = elapsed
            
            if not success:
                result.status = "train_failed"
                result.error_message = output[:500]
                return result
            
            self.log(f"Training completed in {elapsed:.2f}s")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path) and not self.dry_run:
            self.log(f"Warning: Checkpoint not found at {checkpoint_path}", "WARN")
            # Try alternative locations
            alt_path = os.path.join(checkpoint_dir, "omni_parameters.pth")
            if os.path.exists(alt_path):
                checkpoint_path = alt_path
                result.checkpoint_path = checkpoint_path
        
        # Evaluation phase
        if not self.skip_evaluation:
            eval_log_dir = os.path.join(exp_dir, "eval_logs")
            os.makedirs(eval_log_dir, exist_ok=True)
            
            eval_cmd = self.build_evaluation_command(exp, checkpoint_path)
            success, output, elapsed = self.run_command(eval_cmd, f"Evaluating {exp.name}")
            result.eval_time = elapsed
            
            if not success:
                result.status = "eval_failed"
                result.error_message = output[:500]
                return result
            
            # Parse metrics
            metrics = self.parse_evaluation_output(output, eval_log_dir)
            result.ppl_wikitext2 = metrics.get('ppl_wikitext2')
            result.ppl_c4 = metrics.get('ppl_c4')
            result.task_results = {k: v for k, v in metrics.items() if not k.startswith('ppl_')}
            
            self.log(f"Evaluation completed in {elapsed:.2f}s")
            if result.ppl_wikitext2:
                self.log(f"WikiText-2 PPL: {result.ppl_wikitext2:.4f}")
            if result.ppl_c4:
                self.log(f"C4 PPL: {result.ppl_c4:.4f}")
        
        result.status = "success"
        return result
    
    def run_all(self):
        """Run all experiments and aggregate results."""
        self.log(f"Starting batch experiments with {len(self.experiments)} configurations")
        self.log(f"Output directory: {self.global_config.output_base_dir}")
        self.log(f"Results will be saved to: {self.output_csv}")
        
        for i, exp in enumerate(self.experiments):
            self.log(f"\n[{i+1}/{len(self.experiments)}] Running experiment: {exp.name}")
            try:
                result = self.run_experiment(exp)
                self.results.append(result)
                self.log(f"Experiment {exp.name} completed with status: {result.status}")
            except Exception as e:
                self.log(f"Experiment {exp.name} failed with exception: {e}", "ERROR")
                self.results.append(ExperimentResult(
                    experiment_name=exp.name,
                    config=asdict(exp),
                    train_time=0.0,
                    eval_time=0.0,
                    checkpoint_path="",
                    status="exception",
                    error_message=str(e)[:500]
                ))
            
            # Save intermediate results
            self.save_results()
        
        self.log(f"\nAll experiments completed!")
        self.print_summary()
    
    def save_results(self):
        """Save results to CSV file."""
        if not self.results:
            return
        
        # Determine all columns
        base_columns = [
            'experiment_name', 'status', 'train_time', 'eval_time',
            'lora_rank', 'lora_alpha', 'lora_lr', 'epochs',
            'ppl_wikitext2', 'ppl_c4', 'checkpoint_path', 'error_message'
        ]
        
        # Collect all task result keys
        task_keys = set()
        for result in self.results:
            task_keys.update(result.task_results.keys())
        
        all_columns = base_columns + sorted(task_keys)
        
        with open(self.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_columns)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'experiment_name': result.experiment_name,
                    'status': result.status,
                    'train_time': f"{result.train_time:.2f}",
                    'eval_time': f"{result.eval_time:.2f}",
                    'lora_rank': result.config.get('lora_rank', ''),
                    'lora_alpha': result.config.get('lora_alpha', ''),
                    'lora_lr': result.config.get('lora_lr', ''),
                    'epochs': result.config.get('epochs', ''),
                    'ppl_wikitext2': f"{result.ppl_wikitext2:.4f}" if result.ppl_wikitext2 else '',
                    'ppl_c4': f"{result.ppl_c4:.4f}" if result.ppl_c4 else '',
                    'checkpoint_path': result.checkpoint_path,
                    'error_message': result.error_message,
                }
                row.update(result.task_results)
                writer.writerow(row)
        
        self.log(f"Results saved to {self.output_csv}")
    
    def print_summary(self):
        """Print a summary of all experiment results."""
        self.log("\n" + "=" * 80)
        self.log("EXPERIMENT SUMMARY")
        self.log("=" * 80)
        
        successful = [r for r in self.results if r.status == "success"]
        failed = [r for r in self.results if r.status != "success"]
        
        self.log(f"Total experiments: {len(self.results)}")
        self.log(f"Successful: {len(successful)}")
        self.log(f"Failed: {len(failed)}")
        
        if successful:
            self.log("\nSuccessful experiments:")
            for r in successful:
                ppl_str = f"PPL(wiki2)={r.ppl_wikitext2:.4f}" if r.ppl_wikitext2 else "PPL=N/A"
                self.log(f"  - {r.experiment_name}: {ppl_str}, train={r.train_time:.1f}s")
        
        if failed:
            self.log("\nFailed experiments:")
            for r in failed:
                self.log(f"  - {r.experiment_name}: {r.status} - {r.error_message[:100]}")
        
        # Find best result
        if successful:
            ppl_results = [(r.experiment_name, r.ppl_wikitext2) for r in successful if r.ppl_wikitext2]
            if ppl_results:
                best = min(ppl_results, key=lambda x: x[1])
                self.log(f"\nBest WikiText-2 PPL: {best[1]:.4f} ({best[0]})")


def parse_experiments_from_config(config: Dict) -> tuple:
    """Parse configuration file into GlobalConfig and list of ExperimentConfig."""
    # Parse global config
    global_dict = config.get('global', {})
    global_config = GlobalConfig(
        model_path=global_dict.get('model_path', ''),
        net=global_dict.get('net', ''),
        output_base_dir=global_dict.get('output_base_dir', './experiments'),
        cache_dir=global_dict.get('cache_dir', './cache'),
        calib_dataset=global_dict.get('calib_dataset', 'wikitext2'),
        eval_ppl=global_dict.get('eval_ppl', True),
        tasks=global_dict.get('tasks', ''),
        real_quant=global_dict.get('real_quant', False),
        multigpu=global_dict.get('multigpu', False),
        attn_implementation=global_dict.get('attn_implementation', 'eager'),
        act_scales=global_dict.get('act_scales'),
        act_shifts=global_dict.get('act_shifts'),
    )
    
    # Parse experiments
    experiments = []
    defaults = config.get('defaults', {})
    
    for exp_dict in config.get('experiments', []):
        # Merge defaults with experiment-specific config
        merged = {**defaults, **exp_dict}
        
        exp = ExperimentConfig(
            name=merged.get('name', f"exp_{len(experiments)}"),
            lora_rank=merged.get('lora_rank', 8),
            lora_alpha=merged.get('lora_alpha', 16.0),
            lora_lr=merged.get('lora_lr', 5e-3),
            let_lr=merged.get('let_lr', 5e-3),
            lwc_lr=merged.get('lwc_lr', 1e-2),
            epochs=merged.get('epochs', 10),
            batch_size=merged.get('batch_size', 1),
            nsamples=merged.get('nsamples', 128),
            wbits=merged.get('wbits', 4),
            abits=merged.get('abits', 16),
            group_size=merged.get('group_size', 128),
            seed=merged.get('seed', 2),
            lwc=merged.get('lwc', True),
            let=merged.get('let', False),
            aug_loss=merged.get('aug_loss', False),
            extra_args=merged.get('extra_args', {}),
        )
        experiments.append(exp)
    
    return global_config, experiments


def generate_param_sweep_experiments(
    base_name: str,
    ranks: List[int] = None,
    alphas: List[float] = None,
    learning_rates: List[float] = None,
    defaults: Dict = None
) -> List[Dict]:
    """Generate experiment configurations for parameter sweep."""
    ranks = ranks or [4, 8, 16]
    alphas = alphas or [8.0, 16.0, 32.0]
    learning_rates = learning_rates or [1e-3, 5e-3, 1e-2]
    defaults = defaults or {}
    
    experiments = []
    for rank in ranks:
        for alpha in alphas:
            for lr in learning_rates:
                exp = {
                    **defaults,
                    'name': f"{base_name}_r{rank}_a{int(alpha)}_lr{lr}",
                    'lora_rank': rank,
                    'lora_alpha': alpha,
                    'lora_lr': lr,
                }
                experiments.append(exp)
    
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="Run batch LoRA experiments for Mixtral models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiments from config file
  python run_lora_experiments.py --config config/lora_experiments.yaml
  
  # Run with custom output CSV
  python run_lora_experiments.py --config config/lora_experiments.yaml --output results.csv
  
  # Dry run to see commands without executing
  python run_lora_experiments.py --config config/lora_experiments.yaml --dry-run
  
  # Skip training and only run evaluation
  python run_lora_experiments.py --config config/lora_experiments.yaml --skip-training
  
  # Generate a parameter sweep config
  python run_lora_experiments.py --generate-sweep --model /path/to/mixtral --net mixtral-8x7b
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML or JSON)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="experiment_results.csv",
        help="Output CSV file for results (default: experiment_results.csv)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training phase, only run evaluation"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation phase, only run training"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--generate-sweep",
        action="store_true",
        help="Generate a parameter sweep configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model path (required for --generate-sweep)"
    )
    parser.add_argument(
        "--net",
        type=str,
        help="Network name (required for --generate-sweep)"
    )
    parser.add_argument(
        "--sweep-output",
        type=str,
        default="config/lora_sweep.yaml",
        help="Output path for generated sweep config"
    )
    
    args = parser.parse_args()
    
    # Generate parameter sweep config
    if args.generate_sweep:
        if not args.model or not args.net:
            parser.error("--model and --net are required for --generate-sweep")
        
        sweep_config = {
            'global': {
                'model_path': args.model,
                'net': args.net,
                'output_base_dir': './experiments/lora_sweep',
                'cache_dir': './cache',
                'calib_dataset': 'wikitext2',
                'eval_ppl': True,
                'tasks': '',
            },
            'defaults': {
                'epochs': 10,
                'batch_size': 1,
                'nsamples': 128,
                'wbits': 4,
                'abits': 16,
                'group_size': 128,
                'lwc': True,
                'let': False,
            },
            'experiments': generate_param_sweep_experiments(
                base_name='mixtral_lora',
                ranks=[4, 8, 16],
                alphas=[8.0, 16.0, 32.0],
                learning_rates=[1e-3, 5e-3, 1e-2]
            )
        }
        
        os.makedirs(os.path.dirname(args.sweep_output) or '.', exist_ok=True)
        
        if args.sweep_output.endswith('.yaml') or args.sweep_output.endswith('.yml'):
            try:
                import yaml
                with open(args.sweep_output, 'w') as f:
                    yaml.dump(sweep_config, f, default_flow_style=False, sort_keys=False)
            except ImportError:
                # Fall back to JSON
                args.sweep_output = args.sweep_output.replace('.yaml', '.json').replace('.yml', '.json')
                with open(args.sweep_output, 'w') as f:
                    json.dump(sweep_config, f, indent=2)
        else:
            with open(args.sweep_output, 'w') as f:
                json.dump(sweep_config, f, indent=2)
        
        print(f"Generated parameter sweep config with {len(sweep_config['experiments'])} experiments")
        print(f"Saved to: {args.sweep_output}")
        return
    
    # Run experiments from config
    if not args.config:
        parser.error("--config is required (or use --generate-sweep to generate a config)")
    
    # Load configuration
    config = load_config(args.config)
    global_config, experiments = parse_experiments_from_config(config)
    
    if not experiments:
        print("No experiments found in configuration file")
        return
    
    print(f"Loaded {len(experiments)} experiment configurations")
    
    # Create and run experiment runner
    runner = ExperimentRunner(
        global_config=global_config,
        experiments=experiments,
        output_csv=args.output,
        dry_run=args.dry_run,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        verbose=not args.quiet
    )
    
    runner.run_all()


if __name__ == "__main__":
    main()
