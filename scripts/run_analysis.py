import sys

from lesion_similarity_analysis.core import SLICE_SIZE, evaluate_slices


def main():
    data_path = "/data2/data1_backup/david/FXS/FXScoh3/LesionEvaluation/FXScoh3"
    for hippoPart in SLICE_SIZE:
        print(f"Evaluating {hippoPart}")
        evaluate_slices(data_path, hippoPart, sigma=50,
                        output_path="../PROCESSED_DATA")


if __name__ == "__main__":
    sys.exit(main())
