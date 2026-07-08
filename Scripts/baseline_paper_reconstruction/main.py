import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / ".matplotlib"))

from parameters import (
    LogicRangeParameters,
    NotGateParameters,
    VirtualQubitParameters,
    beta1_sweep,
    epsilon1_values,
    fig2_beta1_sweep,
    majority_input_grid,
    nor_input_grid,
)
from src.io_utils import save_rows_csv
from src.logic_gates import generate_majority_volume, generate_nor_surface, generate_xor_surface
from src.not_gate import generate_not_gate_curves, save_curves_csv
from src.tradeoff import generate_not_tradeoff
from src.virtual_qubit import generate_virtual_temperature_curve
from visualization.figure_plots import (
    plot_fig2_virtual_temperature,
    plot_fig3b_not_transfer,
    plot_fig3c_tradeoff,
    plot_fig6_nor_surface,
    plot_fig7_majority_slices,
    plot_fig8_xor_surface,
)
from visualization.not_gate_plots import plot_transfer_curves


def main():
    not_params = NotGateParameters()
    virtual_params = VirtualQubitParameters()
    logic_params = LogicRangeParameters()
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig2_rows = generate_virtual_temperature_curve(
        beta1_values=fig2_beta1_sweep(),
        params=virtual_params,
    )
    fig2_csv = save_rows_csv(
        fig2_rows,
        output_dir / "fig2_virtual_temperature_regimes.csv",
        ["beta1", "beta_v", "regime"],
    )
    fig2_plot = plot_fig2_virtual_temperature(
        fig2_rows,
        output_dir / "fig2_virtual_temperature_regimes.png",
        beta0=virtual_params.beta0,
        engine_threshold=(virtual_params.epsilon0 / virtual_params.epsilon1)
        * virtual_params.beta0,
    )

    not_curves = generate_not_gate_curves(
        beta1_values=beta1_sweep(),
        epsilon1_list=epsilon1_values(),
        params=not_params,
    )

    not_csv = save_curves_csv(not_curves, output_dir / "not_gate_transfer_curves.csv")
    not_plot = plot_transfer_curves(
        not_curves,
        output_dir / "not_gate_transfer_curves.png",
        beta_hot=not_params.beta_hot,
        beta_cold=not_params.beta_cold,
    )
    fig3b_plot = plot_fig3b_not_transfer(
        not_curves,
        output_dir / "fig3b_not_transfer_curve.png",
        beta_hot=not_params.beta_hot,
        beta_cold=not_params.beta_cold,
    )
    tradeoff_rows = generate_not_tradeoff(epsilon1_values(), not_params)
    tradeoff_csv = save_rows_csv(
        tradeoff_rows,
        output_dir / "fig3c_error_dissipation_tradeoff.csv",
        [
            "epsilon1",
            "average_error",
            "dissipation_proxy",
            "beta_out_hot_input",
            "beta_out_cold_input",
        ],
    )
    tradeoff_plot = plot_fig3c_tradeoff(
        tradeoff_rows,
        output_dir / "fig3c_error_dissipation_tradeoff.png",
    )

    beta1_grid, beta2_grid = nor_input_grid()
    nor_rows = generate_nor_surface(beta1_grid, beta2_grid, logic_params)
    nor_csv = save_rows_csv(
        nor_rows,
        output_dir / "fig6_nor_response_surface.csv",
        ["beta1", "beta2", "beta_v", "beta_z_infinity"],
    )
    nor_plot = plot_fig6_nor_surface(
        nor_rows,
        output_dir / "fig6_nor_response_surface.png",
        beta_hot=logic_params.beta_hot,
        beta_cold=logic_params.beta_cold,
    )

    beta1_grid_3d, beta2_grid_3d, beta3_grid_3d = majority_input_grid()
    majority_rows = generate_majority_volume(beta1_grid_3d, beta2_grid_3d, beta3_grid_3d, logic_params)
    majority_csv = save_rows_csv(
        majority_rows,
        output_dir / "fig7_majority_response.csv",
        ["beta1", "beta2", "beta3", "beta_v", "beta_z_infinity"],
    )
    majority_plot = plot_fig7_majority_slices(
        majority_rows,
        output_dir / "fig7_majority_response_slices.png",
        beta_hot=logic_params.beta_hot,
        beta_cold=logic_params.beta_cold,
    )

    xor_rows = generate_xor_surface(beta1_grid, beta2_grid, logic_params)
    xor_csv = save_rows_csv(
        xor_rows,
        output_dir / "fig8_xor_network_response.csv",
        ["beta1", "beta2", "nand_beta_z", "or_beta_z", "xor_beta_z"],
    )
    xor_plot = plot_fig8_xor_surface(
        xor_rows,
        output_dir / "fig8_xor_network_response.png",
        beta_hot=logic_params.beta_hot,
        beta_cold=logic_params.beta_cold,
    )

    print("Baseline paper reconstruction complete.")
    print(f"Saved Fig. 2 CSV: {fig2_csv}")
    print(f"Saved Fig. 2 plot: {fig2_plot}")
    print(f"Saved NOT CSV: {not_csv}")
    print(f"Saved NOT plot: {not_plot}")
    print(f"Saved Fig. 3B plot: {fig3b_plot}")
    print(f"Saved Fig. 3C CSV: {tradeoff_csv}")
    print(f"Saved Fig. 3C plot: {tradeoff_plot}")
    print(f"Saved Fig. 6 CSV: {nor_csv}")
    print(f"Saved Fig. 6 plot: {nor_plot}")
    print(f"Saved Fig. 7 CSV: {majority_csv}")
    print(f"Saved Fig. 7 plot: {majority_plot}")
    print(f"Saved Fig. 8 CSV: {xor_csv}")
    print(f"Saved Fig. 8 plot: {xor_plot}")
    print("Models: virtual-temperature regimes, NOT, trade-off proxy, NOR, majority, XOR network.")


if __name__ == "__main__":
    main()
