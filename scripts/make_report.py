from zeroqkernel.reports.export import export_artifacts


def main() -> None:
    export_artifacts("results/runs/latest", "results/runs/latest/export")


if __name__ == "__main__":
    main()
