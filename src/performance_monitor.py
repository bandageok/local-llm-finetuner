"""推理性能监控增强

集成 AdvancedPerformanceMonitor，提供：
- 首 Token 延迟 / 纯生成速度 / 实时平均速度
- GPU 利用率 / 温度 / 功耗追踪
- 显存峰值 + 增长趋势分析
- 速度评级 / 显存效率评级
- 优化建议
"""
import time
import logging
import torch
from threading import Thread
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)

# ── pynvml GPU 监控 ───────────────────────────────────────────
try:
    import pynvml

    pynvml.nvmlInit()
    _HAS_PYNVML = True
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _HAS_PYNVML = False
    _GPU_HANDLE = None


class AdvancedPerformanceMonitor:
    """
    推理过程性能监控器。

    用法（与 evaluate.py 的 generate_sample_output 集成）：
        monitor = AdvancedPerformanceMonitor()
        monitor.start()
        monitor.start_inference_timing(input_token_count)

        # ... streaming inference ...

        monitor.record_first_token()
        for new_text in streamer:
            monitor.record_token_generation(new_text)
            print(new_text, end="", flush=True)

        monitor.generation_end_time = time.time()
        monitor.output_tokens = len(tokenizer.encode(full_response))
        monitor.stop()

        speeds = monitor.calculate_speeds()
        summary = monitor.get_summary()
        memory_analysis = monitor.get_memory_analysis()
        print_detailed_performance_report(speeds, summary, memory_analysis)
    """

    def __init__(self):
        self.start_time: Optional[float] = None
        self.metrics: List[Dict] = []
        self.monitoring: bool = False

        # 速度计算
        self.inference_start_time: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self.generation_end_time: Optional[float] = None

        # Token 计数
        self.input_tokens: int = 0
        self.output_tokens: int = 0

        # 实时速度跟踪
        self.token_timestamps: List[float] = []
        self.response_chunks: List[str] = []

        # 内存峰值
        self.memory_baseline: Optional[Dict] = None
        self.system_memory_peak: float = 0.0
        self.gpu_memory_peak: float = 0.0
        self.pytorch_memory_peak: float = 0.0
        self.memory_snapshots: List[Dict] = []

        # 可选：tokenizer 用于精确 token 计数
        self.tokenizer = None

    # ── 生命周期 ──────────────────────────────────────────────

    def start(self):
        """开始监控（记录基线）"""
        self.start_time = time.time()
        self.monitoring = True
        self.metrics = []
        self.token_timestamps = []
        self.response_chunks = []
        self.memory_snapshots = []
        self.record_memory_baseline()

    def stop(self):
        """停止监控"""
        self.monitoring = False

    def reset(self):
        """重置所有状态"""
        self.__init__()

    # ── 内存 ───────────────────────────────────────────────────

    def record_memory_baseline(self):
        """记录内存使用基线"""
        import psutil

        memory = psutil.virtual_memory()
        self.memory_baseline = {
            "system_memory_used": memory.used / 1024 ** 3,
            "system_memory_percent": memory.percent,
        }

        if torch.cuda.is_available():
            self.memory_baseline["gpu_memory_allocated"] = (
                torch.cuda.memory_allocated() / 1024 ** 3
            )
            self.memory_baseline["gpu_memory_reserved"] = (
                torch.cuda.memory_reserved() / 1024 ** 3
            )

            if _HAS_PYNVML and _GPU_HANDLE is not None:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                    self.memory_baseline["gpu_memory_used"] = mem_info.used / 1024 ** 3
                    self.memory_baseline["gpu_memory_total"] = mem_info.total / 1024 ** 3
                except Exception:
                    pass

    def update_memory_peaks(self):
        """更新内存峰值"""
        import psutil

        memory = psutil.virtual_memory()
        current_system_memory = memory.used / 1024 ** 3
        self.system_memory_peak = max(self.system_memory_peak, current_system_memory)

        if torch.cuda.is_available():
            current_pytorch_allocated = torch.cuda.memory_allocated() / 1024 ** 3
            current_pytorch_reserved = torch.cuda.memory_reserved() / 1024 ** 3
            self.pytorch_memory_peak = max(self.pytorch_memory_peak, current_pytorch_allocated)

            if _HAS_PYNVML and _GPU_HANDLE is not None:
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                    self.gpu_memory_peak = max(
                        self.gpu_memory_peak, mem_info.used / 1024 ** 3
                    )
                except Exception:
                    pass

        # 每秒记录一次快照
        current_time = time.time()
        if (
            not self.memory_snapshots
            or (current_time - self.memory_snapshots[-1]["timestamp"]) >= 1.0
        ):
            snapshot = {
                "timestamp": current_time - self.start_time,
                "system_memory": current_system_memory,
                "system_memory_percent": memory.percent,
            }

            if torch.cuda.is_available():
                snapshot["pytorch_allocated"] = torch.cuda.memory_allocated() / 1024 ** 3
                snapshot["pytorch_reserved"] = torch.cuda.memory_reserved() / 1024 ** 3

                if _HAS_PYNVML and _GPU_HANDLE is not None:
                    try:
                        mem_info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                        snapshot["gpu_memory_used"] = mem_info.used / 1024 ** 3
                        snapshot["gpu_memory_utilization"] = (
                            mem_info.used / mem_info.total * 100
                        )
                    except Exception:
                        pass

            self.memory_snapshots.append(snapshot)

    # ── 推理计时 ───────────────────────────────────────────────

    def start_inference_timing(self, input_token_count: int):
        """开始推理计时"""
        self.inference_start_time = time.time()
        self.input_tokens = input_token_count
        self.output_tokens = 0
        self.first_token_time = None

    def record_first_token(self) -> Optional[float]:
        """记录首个 Token 生成时间，返回延迟（秒）"""
        if self.first_token_time is None:
            self.first_token_time = time.time()
            return self.first_token_time - self.inference_start_time
        return None

    def record_token_generation(self, new_text: str):
        """记录每个 Token 生成时间"""
        self.token_timestamps.append(time.time())
        self.response_chunks.append(new_text)

    def end_inference_timing(self, total_generated_text: str):
        """结束推理计时并计算最终 Token 数"""
        self.generation_end_time = time.time()
        if self.tokenizer:
            self.output_tokens = len(self.tokenizer.encode(total_generated_text))
        else:
            self.output_tokens = len(self.response_chunks)

    # ── 快照收集 ───────────────────────────────────────────────

    def collect_snapshot(self):
        """收集当前资源使用快照（后台线程调用）"""
        import psutil

        if not self.monitoring:
            return

        import threading

        snapshot = {
            "timestamp": time.time() - self.start_time,
            "cpu": psutil.cpu_percent(),
        }

        self.update_memory_peaks()

        if torch.cuda.is_available():
            snapshot["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1024 ** 3
            snapshot["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1024 ** 3

            if _HAS_PYNVML and _GPU_HANDLE is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
                    snapshot["gpu_usage"] = util.gpu

                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
                    snapshot["gpu_memory_percent"] = mem_info.used / mem_info.total * 100

                    try:
                        snapshot["gpu_temp"] = pynvml.nvmlDeviceGetTemperature(
                            _GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU
                        )
                    except Exception:
                        pass
                    try:
                        snapshot["gpu_power"] = (
                            pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

        self.metrics.append(snapshot)

    # ── 指标计算 ───────────────────────────────────────────────

    def calculate_speeds(self) -> Dict[str, Any]:
        """计算各种速度指标"""
        if not self.inference_start_time or not self.generation_end_time:
            return {}

        total_time = self.generation_end_time - self.inference_start_time
        first_latency = (
            (self.first_token_time - self.inference_start_time)
            if self.first_token_time
            else 0
        )
        generation_time = (
            (self.generation_end_time - self.first_token_time)
            if self.first_token_time
            else total_time
        )

        speeds = {
            "total_inference_time": total_time,
            "first_token_latency": first_latency,
            "pure_generation_time": generation_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }

        if total_time > 0:
            speeds["total_inference_speed"] = self.output_tokens / total_time
            speeds["overall_processing_speed"] = (
                self.input_tokens + self.output_tokens
            ) / total_time

        if generation_time > 0 and self.output_tokens > 1:
            speeds["pure_generation_speed"] = (
                self.output_tokens - 1
            ) / generation_time
            speeds["response_speed"] = (
                self.output_tokens - 1
            ) / generation_time

        if len(self.token_timestamps) > 1:
            intervals = []
            for i in range(1, len(self.token_timestamps)):
                iv = self.token_timestamps[i] - self.token_timestamps[i - 1]
                if iv > 0:
                    intervals.append(iv)
            if intervals:
                speeds["real_time_avg_speed"] = 1.0 / (
                    sum(intervals) / len(intervals)
                )
                speeds["real_time_intervals"] = intervals

        return speeds

    def get_summary(self) -> Dict[str, Any]:
        """获取完整监控摘要"""
        if not self.metrics:
            return {}

        cpu_values = [m["cpu"] for m in self.metrics]
        summary: Dict[str, Any] = {
            "duration": self.metrics[-1]["timestamp"],
            "cpu_avg": sum(cpu_values) / len(cpu_values),
            "cpu_max": max(cpu_values),
            "cpu_min": min(cpu_values),
            "memory_peaks": {
                "system_memory_peak": self.system_memory_peak,
                "pytorch_memory_peak": self.pytorch_memory_peak,
                "gpu_memory_peak": self.gpu_memory_peak,
            },
        }

        if self.memory_baseline:
            summary["memory_growth"] = {
                "system_memory_growth": (
                    self.system_memory_peak
                    - self.memory_baseline.get("system_memory_used", 0)
                ),
                "pytorch_memory_growth": (
                    self.pytorch_memory_peak
                    - self.memory_baseline.get("gpu_memory_allocated", 0)
                ),
            }

        if torch.cuda.is_available():
            gpu_mem_values = [
                m.get("gpu_memory_allocated", 0) for m in self.metrics
            ]
            summary["gpu_memory_avg"] = sum(gpu_mem_values) / len(gpu_mem_values)

            gpu_usage_vals = [m["gpu_usage"] for m in self.metrics if "gpu_usage" in m]
            if gpu_usage_vals:
                summary["gpu_usage_avg"] = sum(gpu_usage_vals) / len(gpu_usage_vals)
                summary["gpu_usage_max"] = max(gpu_usage_vals)
                summary["gpu_usage_min"] = min(gpu_usage_vals)

            gpu_temp_vals = [m["gpu_temp"] for m in self.metrics if "gpu_temp" in m]
            if gpu_temp_vals:
                summary["gpu_temp_avg"] = sum(gpu_temp_vals) / len(gpu_temp_vals)
                summary["gpu_temp_max"] = max(gpu_temp_vals)

            gpu_power_vals = [m["gpu_power"] for m in self.metrics if "gpu_power" in m]
            if gpu_power_vals:
                summary["gpu_power_avg"] = sum(gpu_power_vals) / len(gpu_power_vals)
                summary["gpu_power_max"] = max(gpu_power_vals)

        return summary

    def get_memory_analysis(self) -> Dict[str, Any]:
        """获取详细内存分析"""
        if not self.memory_snapshots:
            return {}

        analysis: Dict[str, Any] = {
            "baseline": self.memory_baseline,
            "peaks": {
                "system_memory_peak": self.system_memory_peak,
                "pytorch_memory_peak": self.pytorch_memory_peak,
                "gpu_memory_peak": self.gpu_memory_peak,
            },
        }

        if len(self.memory_snapshots) > 1:
            sys_mem = [s["system_memory"] for s in self.memory_snapshots]
            py_mem = [s.get("pytorch_allocated", 0) for s in self.memory_snapshots]
            analysis["trends"] = {
                "system_memory_trend": (
                    "increasing" if sys_mem[-1] > sys_mem[0] else "stable"
                ),
                "pytorch_memory_trend": (
                    "increasing" if py_mem[-1] > py_mem[0] else "stable"
                ),
                "system_memory_variance": max(sys_mem) - min(sys_mem),
                "pytorch_memory_variance": max(py_mem) - min(py_mem),
            }

        if self.memory_baseline and torch.cuda.is_available():
            baseline_py = self.memory_baseline.get("gpu_memory_allocated", 0)
            model_footprint = self.pytorch_memory_peak - baseline_py
            eff = "good" if model_footprint < 6 else "high" if model_footprint < 8 else "very_high"
            analysis["efficiency"] = {
                "model_memory_footprint": model_footprint,
                "memory_efficiency": eff,
            }

        return analysis


# ── 报告打印 ───────────────────────────────────────────────────────────────

def print_detailed_performance_report(speeds: Dict, summary: Dict, memory_analysis: Dict):
    """打印详细的性能分析报告（控制台友好）"""
    print(f"\n{'=' * 60}")
    print("Performance Report")
    print(f"{'=' * 60}")

    # 速度
    if speeds:
        print(f"\nSpeed Metrics:")
        print(f"  First Token Latency: {speeds.get('first_token_latency', 0):.3f}s")
        print(f"  Total Inference Time: {speeds.get('total_inference_time', 0):.3f}s")
        print(f"  Pure Generation Time: {speeds.get('pure_generation_time', 0):.3f}s")
        print(f"  Input Tokens: {speeds.get('input_tokens', 0)}")
        print(f"  Output Tokens: {speeds.get('output_tokens', 0)}")
        print(f"\n  Inference Speed:")
        total = speeds.get("total_inference_speed", 0)
        pure = speeds.get("pure_generation_speed", 0)
        realtime = speeds.get("real_time_avg_speed", 0)
        print(f"    Overall:  {total:.1f} tokens/s (w/ first token)")
        print(f"    Pure:     {pure:.1f} tokens/s (excl. first token)")
        print(f"    Realtime: {realtime:.1f} tokens/s")

        # 速度评级
        pure_spd = speeds.get("pure_generation_speed", 0)
        if pure_spd > 30:
            rating = "Excellent (>30)"
        elif pure_spd > 20:
            rating = "Good (20-30)"
        elif pure_spd > 15:
            rating = "Average (15-20)"
        elif pure_spd > 10:
            rating = "Below Average (10-15)"
        else:
            rating = "Needs Optimization (<10)"
        print(f"    Rating: {rating}")

    # 内存
    if memory_analysis:
        baseline = memory_analysis.get("baseline", {})
        peaks = memory_analysis.get("peaks", {})
        trends = memory_analysis.get("trends", {})
        efficiency = memory_analysis.get("efficiency", {})

        print(f"\nMemory Usage:")
        if baseline:
            print(
                f"  System: {baseline.get('system_memory_used', 0):.2f}GB "
                f"({baseline.get('system_memory_percent', 0):.0f}%)"
            )
            print(
                f"  GPU Allocated: {baseline.get('gpu_memory_allocated', 0):.2f}GB"
            )

        print(f"  System Peak: {peaks.get('system_memory_peak', 0):.2f}GB")
        print(f"  PyTorch GPU Peak: {peaks.get('pytorch_memory_peak', 0):.2f}GB")
        if peaks.get("gpu_memory_peak", 0) > 0:
            print(f"  GPU Memory Peak: {peaks.get('gpu_memory_peak', 0):.2f}GB")

        if summary and "memory_growth" in summary:
            g = summary["memory_growth"]
            print(
                f"  System Growth: {g.get('system_memory_growth', 0):.2f}GB"
            )
            print(
                f"  GPU Growth: {g.get('pytorch_memory_growth', 0):.2f}GB"
            )

        if efficiency:
            mp = efficiency.get("model_memory_footprint", 0)
            eff = efficiency.get("memory_efficiency", "unknown")
            print(f"  Model Footprint: {mp:.2f}GB ({eff})")

        if trends:
            print(
                f"  System Trend: {trends.get('system_memory_trend', 'unknown')} "
                f"(variance: {trends.get('system_memory_variance', 0):.2f}GB)"
            )

    # 资源
    if summary:
        print(f"\nResource Usage:")
        print(
            f"  CPU: avg={summary.get('cpu_avg', 0):.1f}%, "
            f"max={summary.get('cpu_max', 0):.1f}%"
        )

        if torch.cuda.is_available():
            print(
                f"  GPU Usage: avg={summary.get('gpu_usage_avg', 0):.1f}%, "
                f"max={summary.get('gpu_usage_max', 0):.1f}%"
            )
            print(
                f"  GPU Memory: avg={summary.get('gpu_memory_avg', 0):.2f}GB"
            )
            if "gpu_temp_avg" in summary:
                print(
                    f"  GPU Temp: avg={summary.get('gpu_temp_avg', 0):.1f}C, "
                    f"max={summary.get('gpu_temp_max', 0):.1f}C"
                )
            if "gpu_power_avg" in summary:
                print(
                    f"  GPU Power: avg={summary.get('gpu_power_avg', 0):.1f}W, "
                    f"max={summary.get('gpu_power_max', 0):.1f}W"
                )

        # 硬件配置
        print(f"\nHardware:")
        gpu_name = torch.cuda.get_device_name()
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  VRAM: {gpu_mem:.1f}GB")

        if memory_analysis and "peaks" in memory_analysis:
            peaks = memory_analysis["peaks"]
            pytorch_peak = peaks.get("pytorch_memory_peak", 0)
            util = (pytorch_peak / gpu_mem) * 100 if gpu_mem > 0 else 0
            print(f"  VRAM Utilization: {util:.1f}% ({pytorch_peak:.2f}GB / {gpu_mem:.1f}GB)")

    # 优化建议
    if speeds or summary or memory_analysis:
        print(f"\nOptimization Suggestions:")
        suggestions = []

        if speeds:
            if speeds.get("first_token_latency", 0) > 1.5:
                suggestions.append("Add model warmup to reduce first token latency")
            if speeds.get("pure_generation_speed", 0) < 20:
                suggestions.append(
                    "Consider FP16 instead of 4bit if VRAM allows "
                    "(check CPU-GPU transfer bottleneck)"
                )

        if summary:
            if summary.get("gpu_usage_avg", 0) < 70:
                suggestions.append(
                    "Low GPU utilization - check batch size or input preprocessing"
                )
            if summary.get("cpu_avg", 0) > 70:
                suggestions.append(
                    "High CPU usage - optimize data preprocessing pipeline"
                )

        if memory_analysis:
            eff = memory_analysis.get("efficiency", {})
            if eff.get("memory_efficiency") == "very_high":
                suggestions.append(
                    "High memory usage - consider more aggressive quantization"
                )
            peaks = memory_analysis.get("peaks", {})
            if peaks.get("pytorch_memory_peak", 0) > 7:
                suggestions.append(
                    "VRAM near limit - reduce max_new_tokens or batch size"
                )

        if suggestions:
            for i, s in enumerate(suggestions, 1):
                print(f"  {i}. {s}")
        else:
            print("  All good - no special optimization needed!")

    print(f"{'=' * 60}")
