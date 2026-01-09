import pytest
import matplotlib.pyplot as plt
import humanize
import csv
import numpy as np
from matplotlib.ticker import FuncFormatter
from pathlib import Path


session_key = pytest.StashKey[pytest.Session]()


def pytest_sessionstart(session):
    session.config.stash[session_key] = session


def pytest_benchmark_generate_json(config, benchmarks, machine_info):
    session = config.stash.get(session_key, None)
    session.stash["huffman_benchmarks"] = benchmarks


def pytest_sessionfinish(session, exitstatus):
    benchmarks_data = session.stash.get("huffman_benchmarks", None)
    if not benchmarks_data:
        return

    histogram_times = {}
    parallel_times = {}
    parallel_64bit_times = {}
    fast_times = {}
    compression_data = {}  # {data_type: {payload_size: [ratios]}}

    # Encode span benchmark data
    encode_sequential_times = {}
    encode_parallel_times = {}
    encode_twopass_times = {}

    for bench in benchmarks_data:
        name = bench["name"]

        if "test_benchmark_huffman_tree" in name:
            param = bench["params"]["payload_size"]
            if "histogram_fast" in name:
                fast_times[param] = bench["stats"].mean
            elif "histogram_parallel_64bit" in name:
                parallel_64bit_times[param] = bench["stats"].mean
            elif "histogram_parallel" in name:
                parallel_times[param] = bench["stats"].mean
            elif "histogram" in name:
                histogram_times[param] = bench["stats"].mean

        elif "test_compression_ratio" in name:
            data_type = bench["params"]["data_type"]
            payload_size = bench["params"]["payload_size"]
            if "ratio" in bench["extra_info"]:
                ratio = bench["extra_info"]["ratio"]

                if data_type not in compression_data:
                    compression_data[data_type] = {}
                if payload_size not in compression_data[data_type]:
                    compression_data[data_type][payload_size] = []

                compression_data[data_type][payload_size].append(ratio)

        elif "test_benchmark_huffman_encode_span" in name:
            param = bench["params"]["payload_size"]
            if "parallel_twopass" in name:
                encode_twopass_times[param] = bench["stats"].mean
            elif "parallel" in name:
                encode_parallel_times[param] = bench["stats"].mean
            elif "sequential" in name:
                encode_sequential_times[param] = bench["stats"].mean

    if histogram_times and fast_times:
        sizes = sorted(histogram_times.keys())

        speedups_parallel = [histogram_times[s] / parallel_times[s] if s in parallel_times else 1.0 for s in sizes]
        speedups_parallel_64bit = [histogram_times[s] / parallel_64bit_times[s] if s in parallel_64bit_times else 1.0 for s in sizes]
        speedups_fast = [histogram_times[s] / fast_times[s] for s in sizes]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

        ax1.loglog(sizes, [histogram_times[s] for s in sizes], marker='o', linestyle='-', label='histogram (baseline)')
        if parallel_times:
            ax1.loglog(sizes, [parallel_times.get(s, float('nan')) for s in sizes], marker='^', linestyle='--', label='histogram_parallel')
        if parallel_64bit_times:
            ax1.loglog(sizes, [parallel_64bit_times.get(s, float('nan')) for s in sizes], marker='v', linestyle='--', label='histogram_parallel_64bit')
        ax1.loglog(sizes, [fast_times[s] for s in sizes], marker='s', linestyle='-', label='histogram_fast')
        ax1.set_xlabel('Payload Size')
        ax1.set_ylabel('Time (seconds)')
        ax1.set_title('Huffman Tree Creation Benchmark')
        ax1.grid(True)
        ax1.legend()
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
        ax1.set_xlim(min(sizes), max(sizes))

        if parallel_times:
            ax2.semilogx(sizes, speedups_parallel, marker='^', linestyle='--', color='orange', linewidth=2, label='parallel')
        if parallel_64bit_times:
            ax2.semilogx(sizes, speedups_parallel_64bit, marker='v', linestyle='--', color='purple', linewidth=2, label='parallel_64bit')
        ax2.semilogx(sizes, speedups_fast, marker='d', linestyle='-', color='green', linewidth=2, label='fast')
        ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
        ax2.axhline(y=2.5, color='red', linestyle=':', alpha=0.7, label='2.5x target')
        ax2.set_xlabel('Payload Size')
        ax2.set_ylabel('Speedup (x)')
        ax2.set_title('Speedup Ratio vs histogram (baseline)')
        ax2.grid(True)
        ax2.legend()
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
        ax2.set_xlim(min(sizes), max(sizes))
        all_speedups = speedups_fast + (speedups_parallel if parallel_times else []) + (speedups_parallel_64bit if parallel_64bit_times else [])
        ax2.set_ylim(0, max(all_speedups) * 1.1)

        plt.tight_layout()
        output_dir = Path('.benchmarks')
        output_dir.mkdir(exist_ok=True)
        plot_path = output_dir / 'huffman_benchmark_plot.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Benchmark plot saved to {plot_path}")

    if encode_sequential_times:
        _generate_encode_span_report(encode_sequential_times, encode_parallel_times, encode_twopass_times)

    if compression_data:
        _generate_compression_ratio_report(compression_data)


def _generate_encode_span_report(sequential_times, parallel_times, twopass_times):
    """Generate plot for huffman_encode_span benchmark comparing implementations."""

    sizes = sorted(sequential_times.keys())

    # Calculate speedup ratios vs sequential baseline
    speedups_parallel = [sequential_times[s] / parallel_times[s] if s in parallel_times else 1.0 for s in sizes]
    speedups_twopass = [sequential_times[s] / twopass_times[s] if s in twopass_times else 1.0 for s in sizes]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Top plot: timing comparison (log scale)
    ax1.loglog(sizes, [sequential_times[s] for s in sizes], marker='o', linestyle='-', label='sequential (baseline)')
    if parallel_times:
        ax1.loglog(sizes, [parallel_times.get(s, float('nan')) for s in sizes], marker='^', linestyle='--', label='parallel')
    if twopass_times:
        ax1.loglog(sizes, [twopass_times.get(s, float('nan')) for s in sizes], marker='s', linestyle='-', label='parallel_twopass')
    ax1.set_xlabel('Payload Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Huffman Encode Span Benchmark')
    ax1.grid(True)
    ax1.legend()
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
    ax1.set_xlim(min(sizes), max(sizes))

    # Bottom plot: speedup ratio (linear scale)
    if parallel_times:
        ax2.semilogx(sizes, speedups_parallel, marker='^', linestyle='--', color='orange', linewidth=2, label='parallel')
    if twopass_times:
        ax2.semilogx(sizes, speedups_twopass, marker='s', linestyle='-', color='green', linewidth=2, label='parallel_twopass')
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    ax2.set_xlabel('Payload Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup Ratio vs sequential (baseline)')
    ax2.grid(True)
    ax2.legend()
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
    ax2.set_xlim(min(sizes), max(sizes))
    all_speedups = (speedups_parallel if parallel_times else []) + (speedups_twopass if twopass_times else [])
    if all_speedups:
        ax2.set_ylim(0, max(max(all_speedups) * 1.1, 2.0))

    plt.tight_layout()
    output_dir = Path('.benchmarks')
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / 'huffman_encode_span_benchmark_plot.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Encode span benchmark plot saved to {plot_path}")


def _generate_compression_ratio_report(compression_data):
    """Generate CSV and plot for compression ratio benchmarks."""

    stats_data = {}
    for data_type, size_dict in compression_data.items():
        stats_data[data_type] = {}
        for payload_size, ratios in size_dict.items():
            mean_ratio = np.mean(ratios)
            var_ratio = np.var(ratios)
            stats_data[data_type][payload_size] = {
                'mean': mean_ratio,
                'variance': var_ratio,
                'std': np.sqrt(var_ratio)
            }

    output_dir = Path('.benchmarks')
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / 'compression_ratio_results.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Data Type', 'Payload Size (bytes)', 'Payload Size (human)', 'Mean Compression Ratio', 'Variance', 'Std Dev'])

        for data_type in sorted(stats_data.keys()):
            for payload_size in sorted(stats_data[data_type].keys()):
                stats = stats_data[data_type][payload_size]
                writer.writerow([
                    data_type,
                    payload_size,
                    humanize.naturalsize(payload_size),
                    f"{stats['mean']:.4f}",
                    f"{stats['variance']:.6f}",
                    f"{stats['std']:.4f}"
                ])

    print(f"Compression ratio results saved to {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'random': 'blue', 'repetitive': 'green', 'uniform': 'red'}
    markers = {'random': 'o', 'repetitive': '^', 'uniform': 's'}

    for data_type in sorted(stats_data.keys()):
        sizes = sorted(stats_data[data_type].keys())
        means = [stats_data[data_type][s]['mean'] for s in sizes]
        stds = [stats_data[data_type][s]['std'] for s in sizes]

        color = colors.get(data_type, 'gray')
        marker = markers.get(data_type, 'x')

        ax.errorbar(sizes, means, yerr=stds, marker=marker, linestyle='-',
                   linewidth=2, markersize=8, capsize=5, capthick=2,
                   color=color, label=data_type.capitalize())

    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='No compression (ratio = 1.0)')

    ax.set_xlabel('Payload Size', fontsize=12)
    ax.set_ylabel('Compression Ratio (compressed/original)', fontsize=12)
    ax.set_title('Huffman Compression Ratio by Data Type and Size', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))

    all_means = [stats['mean'] for dt in stats_data.values() for stats in dt.values()]
    if all_means:
        y_min = max(0, min(all_means) - 0.1)
        y_max = max(all_means) + 0.1
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plot_path = output_dir / 'compression_ratio_plot.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Compression ratio plot saved to {plot_path}")
