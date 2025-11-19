import argparse
from pathlib import Path

import numpy as np


def generate_prices(num_stocks: int, num_steps: int, seed: int | None = 42) -> np.ndarray:
    """
    Generate a simple price matrix, shape = (num_steps, num_stocks).
    Each stock follows an independent random walk, suitable for MA / EMA tests.
    """
    rng = np.random.default_rng(seed)

    # 每只股票一个独立随机游走
    # 起始价 100，单步波动 ~ N(0, 1)
    steps = rng.normal(loc=0.0, scale=1.0, size=(num_steps, num_stocks))
    prices = 100.0 + np.cumsum(steps, axis=0)
    return prices


def save_csv(prices: np.ndarray, out_path: Path) -> None:
    """
    Save price matrix to CSV.
    Rows: time steps; columns: stocks; column names: s0, s1, ...
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_steps, num_stocks = prices.shape
    header = ",".join([f"s{i}" for i in range(num_stocks)])

    # np.savetxt 会自动按行输出
    np.savetxt(out_path, prices, delimiter=",", header=header, comments="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CSV data for MA / EMA benchmarking.")
    parser.add_argument("--stocks", type=int, default=3000, help="Number of stocks (columns), default 3000")
    parser.add_argument("--steps", type=int, default=5000, help="Number of time steps (rows), default 5000")
    parser.add_argument(
        "--output",
        type=str,
        default="data/prices_3000x5000.csv",
        help="Output CSV path, default data/prices_3000x5000.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed, default 42")

    args = parser.parse_args()

    prices = generate_prices(args.stocks, args.steps, seed=args.seed)
    out_path = Path(args.output)
    save_csv(prices, out_path)

    print(f"Generated CSV: {out_path}  shape=({args.steps}, {args.stocks})")


if __name__ == "__main__":
    main()
