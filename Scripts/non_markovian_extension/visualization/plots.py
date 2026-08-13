from pathlib import Path


def plot_population_dynamics(times, populations, output_path):
    """Plot excited-state population dynamics."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(times, populations)
    ax.set_xlabel("time")
    ax.set_ylabel("excited population")
    ax.set_title("Single-Qubit Non-Markovian Benchmark")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_regime_comparison(rows, output_path):
    """Plot sigma_z and purity for structured-bath regimes."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    regimes = list(dict.fromkeys(row["regime"] for row in rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for regime in regimes:
        regime_rows = [row for row in rows if row["regime"] == regime]
        times = [row["time"] for row in regime_rows]
        sigma_z = [row["sigma_z"] for row in regime_rows]
        purities = [row["purity"] for row in regime_rows]
        ax1.plot(times, sigma_z, label=regime)
        ax2.plot(times, purities, label=regime)

    ax1.set_xlabel("time")
    ax1.set_ylabel(r"$\langle \sigma_z \rangle$")
    ax1.set_title("Structured-Bath Qubit Dynamics")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    ax2.set_xlabel("time")
    ax2.set_ylabel(r"$Tr(\rho^2)$")
    ax2.set_title("Purity")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_convergence_scan(rows, output_path):
    """Plot observable drift and runtime for a convergence scan."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [f"dt={row['dt']}\nmem={row['memory_time']}" for row in rows]
    drift = [row["drift_from_first"] for row in rows]
    runtime = [row["elapsed_seconds"] for row in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax1.plot(labels, drift, marker="o")
    ax1.set_ylabel(r"$|\Delta \langle\sigma_z\rangle|$")
    ax1.set_title("Observable Drift")
    ax1.grid(True, alpha=0.25)

    ax2.plot(labels, runtime, marker="o", color="#d62728")
    ax2.set_ylabel("seconds")
    ax2.set_title("Runtime")
    ax2.grid(True, alpha=0.25)

    fig.suptitle("Phase 3 Convergence Scan")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path
