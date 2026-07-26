import matplotlib.pyplot as plt
from pathlib import Path

def show_plot(y_test, test_predictions, model_comparison, horizon, figures_path, show=True, filename=None):
    best_model_name = model_comparison.loc[0, "Model"]
    plt.figure(figsize=(14, 6))
    plt.plot(
        y_test.index,
        y_test,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        y_test.index,
        test_predictions,
        label=best_model_name
    )
    plt.xlabel("Forecast origin")
    plt.ylabel(f"{horizon}-month-ahead price")
    plt.title("Test predictions")
    plt.legend()
    plt.tight_layout()

    figures_dir = Path(figures_path)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        filename = f"test_predictions_h{horizon}.png"
    save_path = figures_dir / filename

    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return save_path