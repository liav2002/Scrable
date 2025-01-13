from src.core.solver import Solver


def main():
    solver = Solver()
    solver.find_best_model()
    solver.false_analysis()


if __name__ == "__main__":
    main()
