import argparse
from src.core.solver import Solver

def main():
    parser = argparse.ArgumentParser(description="Run tasks using the Solver module.")
    parser.add_argument(
        "--task",
        type=str,
        choices=["train", "predict"],
        required=True,
        help="Specify the task to run: 'train' or 'predict'."
    )

    args = parser.parse_args()
    solver = Solver()

    if args.task == "train":
        solver.find_best_model()
        solver.false_analysis()
    elif args.task == "predict":
        solver.test_best_model()

if __name__ == "__main__":
    main()