import pandas as pd 
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

def _data_path(filename):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / Path(filename).name

def create_dataset(results):
    df = pd.DataFrame(results)
    if "rho_ss" in df:
        df = df.drop(columns=["rho_ss"])
    return df 

def save_dataset(results, filename="thermodynamic_neuron_data.csv"):
    df = create_dataset(results)
    output_path = _data_path(filename)
    df.to_csv(output_path, index=False)
    df.attrs["saved_to"] = str(output_path)
    return df 

def create_structured_datasets(results):
    df = create_dataset(results).copy()
    df.insert(0, "point_id", range(len(df)))

    threshold = df["threshold"] if "threshold" in df else df["beta_out"].mean()
    beta_out_range = df["beta_out"].max() - df["beta_out"].min()
    sigma_abs = df["sigma"].abs()
    sum_J = df["sum_J"] if "sum_J" in df else df["J0"] + df["J1"] + df["Jz"]
    final_error = df.get("final_error", pd.Series([0.0] * len(df)))
    convergence_steps = df.get("convergence_steps", pd.Series([0] * len(df)))
    margin = df.get("margin", (df["beta_out"] - threshold).abs())
    confidence = df.get("confidence_score", margin / (beta_out_range or 1.0))
    logic_output = df.get("logic_output", (df["beta_out"] > threshold).astype(int))

    datasets = {
        "00_Master_GKSL_Dataset.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
            "beta_v": df["beta_v"],
            "J0": df["J0"],
            "J1": df["J1"],
            "Jz": df["Jz"],
            "sigma": df["sigma"],
            "sum_J": sum_J,
            "energy_conserved": sum_J.abs() < 1e-12,
            "second_law_valid": df.get("second_law_valid", df["sigma"] >= -1e-10),
            "beta_out": df["beta_out"],
            "convergence_steps": convergence_steps,
            "final_error": final_error,
            "threshold": threshold,
            "margin": margin,
            "logic_output": logic_output,
        }),
        "01_Input_Parameters.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
        }),
        "02_Collector_Metrics.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
            "beta_v": df["beta_v"],
            "J0": df["J0"],
            "J1": df["J1"],
            "Jz": df["Jz"],
            "Jz_abs": df["Jz"].abs(),
        }),
        "03_Thermodynamic_Consistency.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
            "sigma": df["sigma"],
            "sigma_abs": sigma_abs,
            "sum_J": sum_J,
            "sum_J_abs": sum_J.abs(),
            "energy_conserved": sum_J.abs() < 1e-12,
            "second_law_valid": df.get("second_law_valid", df["sigma"] >= -1e-10),
        }),
        "04_Output_Dynamics.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
            "beta_out": df["beta_out"],
            "beta_out_range": beta_out_range,
            "convergence_steps": convergence_steps,
            "final_error": final_error,
            "error_satisfied": final_error < 1e-3,
        }),
        "05_Logic_Performance.csv": pd.DataFrame({
            "point_id": df["point_id"],
            "beta1": df["beta1"],
            "beta_out": df["beta_out"],
            "beta_threshold": threshold,
            "margin_from_threshold": margin,
            "confidence_score": confidence,
            "logic_output": logic_output,
            "logic_output_bit": logic_output.astype(int),
        }),
    }

    return datasets

def save_structured_datasets(results):
    datasets = create_structured_datasets(results)
    saved_paths = {}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, df in datasets.items():
        output_path = DATA_DIR / filename
        df.to_csv(output_path, index=False)
        saved_paths[filename] = str(output_path)
    return saved_paths
