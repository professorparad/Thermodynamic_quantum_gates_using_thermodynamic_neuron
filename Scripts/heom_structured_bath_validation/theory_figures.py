import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))


def drude_lorentz_j(omega, lam=0.075, gamma=0.8):
    return 2.0 * lam * gamma * omega / (omega**2 + gamma**2)


def bath_correlation(times, beta=1 / 0.75, lam=0.075, gamma=0.8):
    omega = np.linspace(1e-4, 8.0, 6000)
    j = drude_lorentz_j(omega, lam=lam, gamma=gamma)
    real = []
    imag = []
    coth = 1.0 / np.tanh(0.5 * beta * omega)
    for time in times:
        real.append(np.trapz(j * coth * np.cos(omega * time), omega))
        imag.append(-np.trapz(j * np.sin(omega * time), omega))
    return np.array(real), np.array(imag)


def lamb_shift(transition_frequencies, lam=0.075, gamma=0.8):
    omega = np.linspace(1e-4, 10.0, 20000)
    j = drude_lorentz_j(omega, lam=lam, gamma=gamma)
    shifts = []
    for w0 in transition_frequencies:
        mask = np.abs(omega - w0) > 2e-3
        integrand = j[mask] * (1.0 / (w0 - omega[mask]) + 1.0 / (w0 + omega[mask]))
        shifts.append(np.trapz(integrand, omega[mask]) / (2.0 * np.pi))
    return np.array(shifts)


def make_path_integral_memory_figure(output_path):
    import matplotlib.pyplot as plt

    times = np.linspace(0.0, 10.0, 300)
    cutoffs = [0.8, 1.8, 3.0]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for cutoff in cutoffs:
        real, imag = bath_correlation(times, gamma=cutoff)
        axes[0].plot(times, real / np.max(np.abs(real)), label=f"gamma={cutoff}")
        axes[1].plot(times, imag / np.max(np.abs(imag)), label=f"gamma={cutoff}")
    axes[0].set_title("Influence-kernel real part")
    axes[1].set_title("Influence-kernel imaginary part")
    for ax in axes:
        ax.set_xlabel("time")
        ax.set_ylabel("normalized C(t)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_lamb_shift_figure(output_path):
    import matplotlib.pyplot as plt

    frequencies = np.linspace(0.2, 3.5, 250)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cutoff in [0.8, 1.8, 3.0]:
        shifts = lamb_shift(frequencies, gamma=cutoff)
        ax.plot(frequencies, shifts, label=f"gamma={cutoff}")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(1.0, color="gray", linestyle="--", linewidth=0.9, label="gate gap")
    ax.set_xlabel("transition frequency")
    ax.set_ylabel("Lamb-shift proxy")
    ax.set_title("Structured-bath Lamb shift")
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def make_qed_implementation_figure(output_path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)

    def box(x, y, w, h, label, color):
        rect = Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)

    def qubit(x, y, label):
        circ = Circle((x, y), 0.35, facecolor="#f2cf5b", edgecolor="black", linewidth=1.2)
        ax.add_patch(circ)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label):
        arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<->", mutation_scale=12, linewidth=1.2)
        ax.add_patch(arr)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha="center", fontsize=8)

    qubit(1.0, 2.0, "Cz")
    qubit(3.0, 2.0, "Fz")
    box(4.7, 1.45, 1.4, 1.1, "lossy\nresonator", "#b7d7f0")
    box(7.0, 1.25, 1.7, 1.5, "engineered\nline", "#d9d9d9")
    box(2.2, 3.0, 1.6, 0.55, "flux / microwave drive", "#c6e5b1")
    arrow(1.35, 2.0, 2.65, 2.0, "gSF")
    arrow(3.35, 2.0, 4.7, 2.0, "kappa")
    arrow(6.1, 2.0, 7.0, 2.0, "J(omega)")
    ax.annotate("", xy=(3.0, 2.35), xytext=(3.0, 3.0), arrowprops={"arrowstyle": "->", "linestyle": "--"})
    ax.text(1.0, 0.7, "output qubit", ha="center", fontsize=9)
    ax.text(3.0, 0.7, "Floquet buffer", ha="center", fontsize=9)
    ax.text(5.4, 0.7, "reaction coordinate", ha="center", fontsize=9)
    ax.text(7.85, 0.7, "structured bath", ha="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        make_path_integral_memory_figure(OUTPUT_DIR / "path_integral_memory_kernel.png"),
        make_lamb_shift_figure(OUTPUT_DIR / "structured_bath_lamb_shift.png"),
        make_qed_implementation_figure(OUTPUT_DIR / "qed_buffer_implementation.png"),
    ]
    for path in paths:
        print("Saved theory figure:", path)


if __name__ == "__main__":
    main()
