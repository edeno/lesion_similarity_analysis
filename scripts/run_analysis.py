import sys

from lesion_similarity_analysis.core import SLICE_SIZE, evaluate_slices


def main():
    data_path = "/stelmo/david/FXS/coh4/lesionEvaluation/FXScoh4/Cropped/"
    for LorR, hippoPart in SLICE_SIZE:
        print(f"Evaluating {hippoPart}, {LorR}")
        evaluate_slices(data_path, hippoPart, LorR, sigma=50,
                        output_path="../PROCESSED_DATA")


if __name__ == "__main__":
    sys.exit(main())