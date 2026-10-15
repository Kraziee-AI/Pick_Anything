"""Create a four-channel PPO initialization from the proven RGB policy."""

import argparse
from grounded_pick_cli import configure_caches

configure_caches()

from pick_anything.grounded_pick.checkpoints import initialize_from_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/grounded_pick/initializations")
    parser.add_argument("--image-history", type=int, choices=[1, 3], default=3)
    args = parser.parse_args()
    print(initialize_from_rgb(args.checkpoint, args.output_dir, args.image_history))
