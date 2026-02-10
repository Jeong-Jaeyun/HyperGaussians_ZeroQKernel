from zeroqkernel.core.utils import load_yaml
from zeroqkernel.datasets.splits import build_split_strategy


def main() -> None:
    cfg = load_yaml("configs/experiment/zeroday_attack_holdout.yaml")
    strategy = build_split_strategy(cfg["split"])
    print(f"Prepared split strategy: {strategy.__class__.__name__}")


if __name__ == "__main__":
    main()
