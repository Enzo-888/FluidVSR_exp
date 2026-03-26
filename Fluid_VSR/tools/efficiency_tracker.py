"""
效率追踪工具 - 用于外部模型自动记录训练和推理效率
"""
import time
import json
import torch
from pathlib import Path


class EfficiencyTracker:
    """效率追踪器"""

    def __init__(self, model_name, dataset_name, save_dir):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 训练统计
        self.train_start_time = None
        self.train_end_time = None
        self.total_iters = 0
        self.batch_size = None
        self.device = None

        # 推理统计
        self.infer_times = []
        self.infer_start_memory = None

    def start_training(self, batch_size, device='cuda:0'):
        """开始训练"""
        self.train_start_time = time.time()
        self.batch_size = batch_size
        self.device = device

        # Reset peak memory stats (works on current device)
        device_str = str(device) if isinstance(device, torch.device) else device
        if device_str.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()

    def end_training(self, total_iters):
        """结束训练"""
        self.train_end_time = time.time()
        self.total_iters = total_iters

    def start_inference(self, device='cuda:0'):
        """开始推理（在预热后调用）"""
        self.device = device

        # Reset peak memory stats and synchronize (works on current device)
        device_str = str(device) if isinstance(device, torch.device) else device
        if device_str.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

    def record_sequence_time(self, elapsed_time):
        """记录单个序列的推理时间"""
        self.infer_times.append(elapsed_time)

    def save_stats(self, checkpoint_path=None, model_size_mb=None, params_total=None):
        """保存统计数据"""
        import numpy as np

        stats = {
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "params_total": params_total,
            "params_trainable": params_total,
            "model_size_mb": model_size_mb,
            "training": {
                "batch_size": self.batch_size,
                "batch_size_note": "sequence-level for video models",
                "total_epochs": None,
                "total_iters": self.total_iters if self.total_iters > 0 else None,
                "time_per_epoch_sec": None,
                "time_per_iter_sec": None,
                "total_time_sec": None,
                "peak_memory_mb": None
            },
            "inference": {
                "total_sequences": len(self.infer_times) if self.infer_times else None,
                "time_per_sequence_sec": None,
                "total_time_sec": None,
                "peak_memory_mb": None
            }
        }

        # 训练统计
        if self.train_start_time and self.train_end_time:
            total_time = self.train_end_time - self.train_start_time
            stats['training']['total_time_sec'] = round(total_time, 2)
            if self.total_iters > 0:
                stats['training']['time_per_iter_sec'] = round(total_time / self.total_iters, 4)

        # 训练显存
        if self.device and self.train_end_time:
            device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
            if device_str.startswith('cuda'):
                peak_memory = torch.cuda.max_memory_allocated()
                stats['training']['peak_memory_mb'] = round(peak_memory / (1024 * 1024), 2)

        # 推理统计
        if self.infer_times:
            avg_time = np.mean(self.infer_times)
            total_time = np.sum(self.infer_times)
            stats['inference']['time_per_sequence_sec'] = round(avg_time, 4)
            stats['inference']['total_time_sec'] = round(total_time, 2)

            # 推理显存
            if self.device:
                device_str = str(self.device) if isinstance(self.device, torch.device) else self.device
                if device_str.startswith('cuda'):
                    peak_memory = torch.cuda.max_memory_allocated()
                    stats['inference']['peak_memory_mb'] = round(peak_memory / (1024 * 1024), 2)

        # 模型大小
        if checkpoint_path:
            import os
            if os.path.exists(checkpoint_path):
                size_bytes = os.path.getsize(checkpoint_path)
                stats['model_size_mb'] = round(size_bytes / (1024 * 1024), 2)

        # 参数量
        if params_total is None and checkpoint_path:
            try:
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                if 'params' in ckpt:
                    state_dict = ckpt['params']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                elif 'model' in ckpt:
                    state_dict = ckpt['model']
                else:
                    state_dict = ckpt
                params_total = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
                stats['params_total'] = params_total
                stats['params_trainable'] = params_total
            except:
                pass

        # 保存
        output_file = self.save_dir / f'{self.model_name}_{self.dataset_name}_efficiency.json'
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n{'='*60}")
        print(f"Efficiency stats saved to: {output_file}")
        print(f"{'='*60}")
        if stats['training']['total_time_sec']:
            print(f"Training time: {stats['training']['total_time_sec']/3600:.2f} hours")
        if stats['inference']['time_per_sequence_sec']:
            print(f"Inference time/seq: {stats['inference']['time_per_sequence_sec']:.4f} sec")
        print(f"{'='*60}\n")

        return output_file
