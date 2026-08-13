import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NON_MARKOVIAN_OUTPUTS = ROOT / "Scripts" / "non_markovian_extension" / "outputs"
FLOQUET_OUTPUTS = ROOT / "Scripts" / "floquet_buffer_extension" / "outputs"
AUDIT_OUTPUTS = ROOT / "Scripts" / "thermodynamic_audit" / "outputs"
FULL_NOT_OUTPUTS = ROOT / "Scripts" / "full_floquet_not_gate" / "outputs"
PHASE_V_OUTPUTS = ROOT / "Scripts" / "phase_v_structured_backend" / "outputs"
OUTPUT_DIR = ROOT / "Scripts" / "project_decision" / "outputs"


def _read_csv(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_text(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _as_float(row, key, default=0.0):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def convergence_passes():
    reports = {
        "dt": _read_text(NON_MARKOVIAN_OUTPUTS / "dt_scan_report.txt"),
        "memory": _read_text(NON_MARKOVIAN_OUTPUTS / "memory_scan_report.txt"),
    }
    return {
        name: text.startswith("Phase 3 convergence report") and "PASS:" in text
        for name, text in reports.items()
    }


def floquet_summary():
    rows = _read_csv(FLOQUET_OUTPUTS / "floquet_buffer_summary.csv")
    by_architecture = {row["architecture"]: row for row in rows}
    direct = by_architecture.get("direct", {})
    buffered = by_architecture.get("floquet_buffer", {})
    direct_distance = _as_float(direct, "final_trace_distance")
    buffered_distance = _as_float(buffered, "final_trace_distance")
    return {
        "direct_trace_distance": direct_distance,
        "buffered_trace_distance": buffered_distance,
        "gain": buffered_distance - direct_distance,
        "work": _as_float(buffered, "integrated_drive_work"),
        "available": bool(rows),
    }


def non_markovian_summary():
    rows = _read_csv(NON_MARKOVIAN_OUTPUTS / "single_qubit_research_summary.csv")
    if not rows:
        return {"available": False, "max_trace_deviation": 0.0, "max_hermiticity_error": 0.0}
    return {
        "available": True,
        "max_trace_deviation": max(_as_float(row, "max_trace_deviation") for row in rows),
        "max_hermiticity_error": max(_as_float(row, "max_hermiticity_error") for row in rows),
        "regimes": len(rows),
    }


def sweep_summary():
    rows = _read_csv(FLOQUET_OUTPUTS / "floquet_parameter_sweep.csv")
    if not rows:
        return {"available": False, "positive_rows": 0, "best_gain": 0.0}
    positive = [row for row in rows if _as_float(row, "trace_distance_gain") > 0.0]
    best_gain = max((_as_float(row, "trace_distance_gain") for row in rows), default=0.0)
    return {
        "available": True,
        "positive_rows": len(positive),
        "total_rows": len(rows),
        "best_gain": best_gain,
    }


def ablation_summary():
    rows = _read_csv(FLOQUET_OUTPUTS / "floquet_ablation_study.csv")
    if not rows:
        return {
            "available": False,
            "positive_rows": 0,
            "total_rows": 0,
            "best_gain": 0.0,
            "best_case": "",
            "no_drive_gain": 0.0,
        }
    positive = [row for row in rows if _as_float(row, "trace_distance_gain") > 0.0]
    best = max(rows, key=lambda row: _as_float(row, "trace_distance_gain"))
    no_drive = next((row for row in rows if row.get("ablation") == "no_periodic_drive"), {})
    return {
        "available": True,
        "positive_rows": len(positive),
        "total_rows": len(rows),
        "best_gain": _as_float(best, "trace_distance_gain"),
        "best_case": best.get("ablation", ""),
        "no_drive_gain": _as_float(no_drive, "trace_distance_gain"),
    }


def audit_summary():
    rows = _read_csv(AUDIT_OUTPUTS / "thermodynamic_audit_table.csv")
    if not rows:
        return {"available": False, "unsafe_rows": 0, "provisional_rows": 0}
    return {
        "available": True,
        "unsafe_rows": sum(row.get("status") == "unsafe-beyond-baseline" for row in rows),
        "provisional_rows": sum(row.get("status") == "provisional" for row in rows),
    }


def full_not_summary():
    rows = _read_csv(FULL_NOT_OUTPUTS / "full_not_gate_summary.csv")
    if not rows:
        return {"available": False, "passing_architectures": 0, "total_architectures": 0}
    passing = [
        row
        for row in rows
        if _as_float(row, "truth_table_accuracy") >= 1.0
    ]
    return {
        "available": True,
        "passing_architectures": len(passing),
        "total_architectures": len(rows),
        "buffered_accuracy": max(
            (
                _as_float(row, "truth_table_accuracy")
                for row in rows
                if row.get("architecture") == "floquet_buffered_three_qubit"
            ),
            default=0.0,
        ),
        "buffered_trace_distance": max(
            (
                _as_float(row, "final_trace_distance")
                for row in rows
                if row.get("architecture") == "floquet_buffered_three_qubit"
            ),
            default=0.0,
        ),
    }


def phase_v_summary():
    rows = _read_csv(PHASE_V_OUTPUTS / "phase_v_quick_summary.csv")
    if not rows:
        return {"available": False, "passing_architectures": 0, "total_architectures": 0}
    passing = [row for row in rows if _as_float(row, "truth_table_accuracy") >= 1.0]
    return {
        "available": True,
        "passing_architectures": len(passing),
        "total_architectures": len(rows),
        "best_accuracy": max((_as_float(row, "truth_table_accuracy") for row in rows), default=0.0),
        "max_trace_deviation": max((_as_float(row, "max_trace_deviation") for row in rows), default=0.0),
        "max_hermiticity_error": max((_as_float(row, "max_hermiticity_error") for row in rows), default=0.0),
    }


def decision():
    convergence = convergence_passes()
    floquet = floquet_summary()
    non_markovian = non_markovian_summary()
    sweep = sweep_summary()
    ablation = ablation_summary()
    audit = audit_summary()
    full_not = full_not_summary()
    phase_v = phase_v_summary()

    numerical_health = (
        non_markovian["available"]
        and non_markovian["max_trace_deviation"] < 1.0e-3
        and non_markovian["max_hermiticity_error"] < 1.0e-5
        and all(convergence.values())
    )
    buffer_has_signal = floquet["available"] and floquet["gain"] > 0.0
    sweep_has_signal = (not sweep["available"]) or sweep["positive_rows"] > 0
    ablation_has_signal = (not ablation["available"]) or ablation["positive_rows"] > 0
    audit_available = audit["available"]
    full_not_passes = (
        full_not["available"]
        and full_not["buffered_accuracy"] >= 1.0
    )

    if numerical_health and buffer_has_signal and sweep_has_signal and ablation_has_signal and full_not_passes:
        verdict = "GKSL-FULL-LADDER-PASS"
        next_step = (
            "Use the passing three-qubit GKSL truth-table surrogate as the thesis prototype, "
            "then replace the output bath by OQuPy/PT-MPO or HEOM for the structured-bath backend."
        )
    elif numerical_health and buffer_has_signal and sweep_has_signal and ablation_has_signal:
        verdict = "RISKY-GO"
        next_step = (
            "Attempt a minimal thermodynamic NOT-style prototype, but keep the first run "
            "single-output and heavily instrumented. Keep thermodynamic claims aligned with "
            "the audit table."
        )
    elif numerical_health:
        verdict = "HOLD-BUFFER"
        next_step = (
            "Do not start the full gate yet. Widen Floquet drive/coupling sweeps until the "
            "buffer has a repeatable distinguishability advantage."
        )
    else:
        verdict = "NO-GO"
        next_step = (
            "Fix convergence and numerical-health diagnostics before building a full gate."
        )

    return {
        "verdict": verdict,
        "next_step": next_step,
        "convergence": convergence,
        "floquet": floquet,
        "non_markovian": non_markovian,
        "sweep": sweep,
        "ablation": ablation,
        "audit": audit,
        "audit_available": audit_available,
        "full_not": full_not,
        "phase_v": phase_v,
    }


def write_report(output_path):
    result = decision()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("MSc thermodynamic Floquet-buffer project decision report\n")
        handle.write("========================================================\n\n")
        handle.write(f"Verdict: {result['verdict']}\n")
        handle.write(f"Next step: {result['next_step']}\n\n")

        handle.write("Numerical health:\n")
        handle.write(f"- dt convergence pass: {result['convergence'].get('dt', False)}\n")
        handle.write(
            f"- memory convergence pass: {result['convergence'].get('memory', False)}\n"
        )
        nm = result["non_markovian"]
        handle.write(f"- non-Markovian summary available: {nm['available']}\n")
        handle.write(f"- max trace deviation: {nm['max_trace_deviation']:.3e}\n")
        handle.write(f"- max Hermiticity error: {nm['max_hermiticity_error']:.3e}\n\n")

        fl = result["floquet"]
        handle.write("Floquet bridge:\n")
        handle.write(f"- summary available: {fl['available']}\n")
        handle.write(f"- direct final trace distance: {fl['direct_trace_distance']:.6f}\n")
        handle.write(f"- buffered final trace distance: {fl['buffered_trace_distance']:.6f}\n")
        handle.write(f"- trace-distance gain: {fl['gain']:.6f}\n")
        handle.write(f"- integrated drive-work proxy: {fl['work']:.6f}\n\n")

        sw = result["sweep"]
        handle.write("Floquet sweep:\n")
        handle.write(f"- sweep available: {sw['available']}\n")
        handle.write(f"- positive rows: {sw.get('positive_rows', 0)} / {sw.get('total_rows', 0)}\n")
        handle.write(f"- best trace-distance gain: {sw.get('best_gain', 0.0):.6f}\n\n")

        ab = result["ablation"]
        handle.write("Floquet ablation:\n")
        handle.write(f"- ablation available: {ab['available']}\n")
        handle.write(f"- positive rows: {ab.get('positive_rows', 0)} / {ab.get('total_rows', 0)}\n")
        handle.write(f"- best ablation case: {ab.get('best_case', '')}\n")
        handle.write(f"- best ablation gain: {ab.get('best_gain', 0.0):.6f}\n")
        handle.write(f"- no-drive gain: {ab.get('no_drive_gain', 0.0):.6f}\n\n")

        audit = result["audit"]
        handle.write("Thermodynamic audit:\n")
        handle.write(f"- audit available: {audit['available']}\n")
        handle.write(f"- provisional rows: {audit.get('provisional_rows', 0)}\n")
        handle.write(f"- unsafe-beyond-baseline rows: {audit.get('unsafe_rows', 0)}\n\n")

        fn = result["full_not"]
        handle.write("Full three-qubit NOT surrogate:\n")
        handle.write(f"- summary available: {fn['available']}\n")
        handle.write(
            f"- passing architectures: {fn.get('passing_architectures', 0)} / "
            f"{fn.get('total_architectures', 0)}\n"
        )
        handle.write(f"- buffered truth-table accuracy: {fn.get('buffered_accuracy', 0.0):.3f}\n")
        handle.write(
            f"- buffered final trace distance: {fn.get('buffered_trace_distance', 0.0):.6f}\n\n"
        )

        pv = result["phase_v"]
        handle.write("Phase V PT-MPO/TEMPO structured backend:\n")
        handle.write(f"- summary available: {pv['available']}\n")
        handle.write(
            f"- passing architectures: {pv.get('passing_architectures', 0)} / "
            f"{pv.get('total_architectures', 0)}\n"
        )
        handle.write(f"- best truth-table accuracy: {pv.get('best_accuracy', 0.0):.3f}\n")
        handle.write(f"- max trace deviation: {pv.get('max_trace_deviation', 0.0):.3e}\n")
        handle.write(f"- max Hermiticity error: {pv.get('max_hermiticity_error', 0.0):.3e}\n\n")

        handle.write("Programming caveat:\n")
        handle.write(
            "Do not jump directly to a full three-qubit multi-bath gate unless this report "
            "is at least RISKY-GO. Full heat/work claims need a separate thermodynamic audit "
            "because strong coupling, memory, and drive work make reduced-system currents "
            "ambiguous.\n"
        )
    return output_path


def main():
    path = write_report(OUTPUT_DIR / "phase2_phase3_decision_report.txt")
    print("Project decision report complete.")
    print(f"Saved report: {path}")


if __name__ == "__main__":
    main()
