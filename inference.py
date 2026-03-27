# inference.py (root level)
import argparse
from src.inference.infer import run_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sustainable urban layout from sketch")
    parser.add_argument('--sketch', type=str, required=True, help='Path to raw JPG/PNG sketch')
    parser.add_argument('--green', type=float, default=0.65, help='Green intensity (0-1)')
    parser.add_argument('--density', type=float, default=0.75, help='Building density (0-1)')
    parser.add_argument('--checkpoint', type=str, default='outputs/checkpoints/G_final.pth')
    parser.add_argument('--use_canny', action='store_true', default=True, help='Apply Canny edge cleaning')

    args = parser.parse_args()

    print(f"Generating layout from: {args.sketch}")
    run_inference(
        sketch_path=args.sketch,
        green_intensity=args.green,
        density=args.density,
        checkpoint_path=args.checkpoint,
        use_canny=args.use_canny
    )