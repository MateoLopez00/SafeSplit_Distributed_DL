import pandas as pd
from IPython.display import display

from utils.experiment_result import ExperimentResult, ensure_experiment_result


def show_rows(
    rows: list[dict],
    max_rows: int = 10,
) -> None:
    if not rows:
        print("<No rows to display>")
        return

    df = pd.DataFrame(rows[:max_rows]).astype(str)

    with pd.option_context(
        "display.max_colwidth", None,
        "display.max_columns", None,
        "display.width", None,
    ):
        display(df)
    print(f"<Showed {min(max_rows, len(rows))} of {len(rows)} rows>")


def result_to_row(result: ExperimentResult | dict[str, object]) -> dict[str, object]:
    typed_result = ensure_experiment_result(result)
    run_cfg = typed_result.config
    return {
        "preset": run_cfg["preset"],
        "arch": run_cfg["arch"],
        "defense": run_cfg["defense"],
        "backdoor": run_cfg["backdoor"],
        "iid_rate": run_cfg["iid_rate"],
        "rounds": run_cfg["num_rounds"],
        "clients": run_cfg["num_clients"],
        "malicious": run_cfg["num_malicious"],
        "final_MA": round(float(typed_result.final_MA), 4),
        "final_BA": round(float(typed_result.final_BA), 4),
        "results_path": typed_result.results_path,
    }
