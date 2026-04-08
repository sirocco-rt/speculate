"""System resource monitoring bars for marimo sidebars."""

import psutil
import shutil
import subprocess


def _make_bar(label, pct, detail, color):
    clamped = max(0, min(100, pct))
    return (
        f'<div style="margin-bottom:8px;">'
        f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#999;margin-bottom:2px;">'
        f'<span>{label}</span><span>{detail}</span></div>'
        f'<div style="height:5px;background:rgba(255,255,255,0.1);border-radius:3px;overflow:hidden;">'
        f'<div style="height:100%;width:{clamped:.0f}%;background:{color};border-radius:3px;'
        f'transition:width 0.5s ease;"></div></div></div>'
    )


def get_usage_html():
    """Return an HTML string with CPU, RAM, and GPU usage bars."""
    try:
        cpu_pct = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)

        html = (
            '<div style="font-size:12px;font-weight:600;color:#ccc;margin-bottom:8px;">'
            'System Resources</div>'
        )
        html += _make_bar("CPU", cpu_pct, f"{cpu_pct:.0f}%", "#4c78a8")
        html += _make_bar("RAM", mem.percent, f"{mem_used_gb:.1f}/{mem_total_gb:.0f} GB", "#e45756")

        if shutil.which("nvidia-smi"):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        idx, name = parts[0], parts[1]
                        used_mb, total_mb = float(parts[2]), float(parts[3])
                        gpu_pct = (used_mb / total_mb * 100) if total_mb > 0 else 0
                        short = name.replace("NVIDIA ", "")
                        html += _make_bar(
                            f"GPU {idx} ({short})", gpu_pct,
                            f"{used_mb / 1024:.1f}/{total_mb / 1024:.0f} GB", "#54a24b"
                        )
            except Exception:
                pass

        return html
    except Exception:
        return ""
