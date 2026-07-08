from pathlib import Path


def save_rows_csv(rows, output_path, headers):
    """Save a list of dict rows to CSV without depending on pandas."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        for row in rows:
            values = []
            for key in headers:
                value = row[key]
                if isinstance(value, float):
                    values.append(f"{value:.12g}")
                else:
                    values.append(str(value))
            handle.write(",".join(values) + "\n")
    return output_path

