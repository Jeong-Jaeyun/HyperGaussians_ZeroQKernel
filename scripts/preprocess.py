from zeroqkernel.core.utils import load_yaml


def main() -> None:
    cfg = load_yaml("configs/dataset/cicids2017.yaml")
    print(f"Preprocess scaffold config loaded for dataset: {cfg['name']}")


if __name__ == "__main__":
    main()
