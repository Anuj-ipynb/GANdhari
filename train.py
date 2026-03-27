# train.py
import torch
import yaml
import argparse
import os
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tiny', action='store_true', help='Use tiny mode (ngf=32) for low VRAM')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file {args.config} not found!")
        return

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.tiny:
        config['ngf'] = 32
        config['ndf'] = 32
        print("Running in Tiny Mode (ngf=32) - Optimized for GTX 1050 Ti")

    print(f"Starting training | Epochs: {config.get('epochs', 100)}")
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()