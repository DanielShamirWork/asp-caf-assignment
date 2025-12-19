import pytest
import matplotlib.pyplot as plt
import humanize
from matplotlib.ticker import FuncFormatter


session_key = pytest.StashKey[pytest.Session]()


def pytest_sessionstart(session):
    session.config.stash[session_key] = session


def pytest_benchmark_generate_json(config, benchmarks, machine_info):
    # Store benchmarks in session stash
    session = config.stash.get(session_key, None)
    session.stash["huffman_benchmarks"] = benchmarks


def pytest_sessionfinish(session, exitstatus):
    benchmarks_data = session.stash.get("huffman_benchmarks", None)
    if not benchmarks_data:
        return

    # Extract sizes and times for both functions
    histogram_times = {}
    fast_times = {}
    for bench in benchmarks_data:
        name = bench["name"]
        if "test_benchmark_huffman_tree" in name:
            param = bench["params"]["payload_size"]
            if "histogram_fast" in name:
                fast_times[param] = bench["stats"].mean
            elif "histogram" in name:
                histogram_times[param] = bench["stats"].mean

    if not histogram_times or not fast_times:
        return

    # Assume sizes are the same for both
    sizes = sorted(histogram_times.keys())

    # Calculate speedup ratios
    speedups = [histogram_times[s] / fast_times[s] for s in sizes]

    # Plot with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])

    # Top plot: timing comparison (log scale)
    ax1.loglog(sizes, [histogram_times[s] for s in sizes], marker='o', linestyle='-', label='histogram')
    ax1.loglog(sizes, [fast_times[s] for s in sizes], marker='s', linestyle='-', label='histogram_fast')
    ax1.set_xlabel('Payload Size')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Huffman Tree Creation Benchmark: histogram vs histogram_fast')
    ax1.grid(True)
    ax1.legend()
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
    ax1.set_xlim(min(sizes), max(sizes))

    # Bottom plot: speedup ratio (linear scale)
    ax2.semilogx(sizes, speedups, marker='d', linestyle='-', color='green', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='No speedup')
    ax2.axhline(y=2.5, color='red', linestyle=':', alpha=0.7, label='2.5x speedup')
    ax2.set_xlabel('Payload Size')
    ax2.set_ylabel('Speedup (x)')
    ax2.set_title('Speedup Ratio (histogram / histogram_fast)')
    ax2.grid(True)
    ax2.legend()
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: humanize.naturalsize(x)))
    ax2.set_xlim(min(sizes), max(sizes))
    ax2.set_ylim(0, max(speedups) * 1.1)

    plt.tight_layout()
    fig.savefig('huffman_benchmark_plot.png', dpi=150)
    plt.close(fig)
    print("Benchmark plot saved to huffman_benchmark_plot.png")
