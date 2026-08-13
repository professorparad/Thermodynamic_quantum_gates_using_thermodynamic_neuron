import csv
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


def _read_csv(name):
    with (OUTPUT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row, key):
    return float(row[key])


def main():
    checks = []

    graph_rows = _read_csv("graph_layout_metrics.csv")
    assert len(graph_rows) == 90
    assert all(int(row["factor_treewidth"]) == 1 for row in graph_rows)
    assert all(int(row["primal_treewidth_upper_bound"]) == 2 for row in graph_rows)
    for architecture in ("direct", "passive_buffer", "floquet_buffer"):
        for chain_length in range(1, 11):
            selected = [
                row for row in graph_rows
                if row["architecture"] == architecture
                and int(row["chain_length"]) == chain_length
            ]
            by_order = {row["ordering"]: row for row in selected}
            assert _number(by_order["graph_optimized_mps"], "cutwidth") <= _number(
                by_order["interleaved_mps"], "cutwidth"
            )
            assert int(float(by_order["factor_tree_ttn"]["cutwidth"])) == 1
    checks.append("graph: 90 layouts, factor treewidth 1, optimized path no worse than interleaving")

    state_rows = _read_csv("exact_state_tn_metrics.csv")
    assert len(state_rows) == 252
    for architecture in ("direct", "passive_buffer", "floquet_buffer"):
        selected = [
            row for row in state_rows
            if row["architecture"] == architecture and int(row["chain_length"]) == 4
        ]
        maxima = {
            method: max(
                int(row["max_required_bond_dimension"])
                for row in selected
                if row["method"] == method
            )
            for method in ("interleaved_mps", "graph_optimized_mps", "factor_tree_ttn")
        }
        assert maxima["factor_tree_ttn"] <= maxima["graph_optimized_mps"]
        assert maxima["graph_optimized_mps"] <= maxima["interleaved_mps"]
    checks.append("state: 252 exact Schmidt audits, TTN <= optimized MPS <= interleaved MPS at L=4")

    compression_rows = _read_csv("mps_compression_metrics.csv")
    assert len(compression_rows) == 72
    for architecture in ("direct", "passive_buffer", "floquet_buffer"):
        selected = sorted(
            (
                row for row in compression_rows
                if row["architecture"] == architecture
                and int(row["chain_length"]) == 4
                and row["ordering"] == "graph_optimized_mps"
            ),
            key=lambda row: int(row["compression_max_bond"]),
        )
        fidelities = [_number(row, "fidelity") for row in selected]
        assert fidelities == sorted(fidelities)
        assert fidelities[-1] > 0.9999
    checks.append("compression: Quimb fidelity increases with bond cap and exceeds 0.9999 at chi=8")

    figures = ["tensor_network_graph_topologies.png", "mps_ttn_graph_scaling.png"]
    for figure in figures:
        path = OUTPUT_DIR / figure
        assert path.exists() and path.stat().st_size > 10_000, figure
    checks.append("figures: topology and scaling plots are present and nonempty")

    report_path = OUTPUT_DIR / "tensor_network_validation_report.txt"
    report_path.write_text(
        "Tensor-network generated-output acceptance checks\n"
        "=================================================\n\n"
        + "\n".join(f"PASS: {check}" for check in checks)
        + "\n",
        encoding="utf-8",
    )
    print(report_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
