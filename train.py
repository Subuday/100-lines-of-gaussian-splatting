import random
import numpy as np
import torch
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_path")
    parser.add_argument("--model_path", default="output/")
    parser.add_argument("--sh_degree", default=3)

    parser.add_argument("--feature_lr", default=1e-4)

    parser.add_argument("--iterations", default=30_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument('--feature_lr', type=float, default=0.0025,
                        help='Spherical harmonics features learning rate, 0.0025 by default.')
    parser.add_argument('--opacity_lr', type=float, default=0.05,
                        help='Opacity learning rate, 0.05 by default.')
    parser.add_argument('--scaling_lr', type=float, default=0.005,
                        help='Scaling learning rate, 0.005 by default.')
    parser.add_argument('--rotation_lr', type=float, default=0.001,
                        help='Rotation learning rate, 0.001 by default.')
    parser.add_argument('--position_lr_init', type=float, default=0.00016,
                        help='Initial 3D position learning rate, 0.00016 by default.')
    parser.add_argument('--position_lr_final', type=float, default=0.0000016,
                        help='Final 3D position learning rate, 0.0000016 by default.')
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01,
                        help='Position learning rate multiplier (cf. Plenoxels), 0.01 by default.')
    parser.add_argument('--position_lr_max_steps', type=int, default=30_000,
                        help='Number of steps (from 0) where position learning rate goes from initial to final. '
                             '30_000 by default.')
    parser.add_argument('--densification_interval', type=int, default=100,
                        help='How frequently to densify, 100 (every 100 iterations) by default.')
    parser.add_argument('--densify_from_iter', type=int, default=500,
                        help='Iteration where densification starts, 500 by default.')
    parser.add_argument('--densify_until_iter', type=int, default=15000,
                        help='Iteration where densification stops, 15_000 by default.')
    parser.add_argument('--densify_grad_threshold', type=float, default=0.0002,
                        help='Limit that decides if points should be densified based on 2D position gradient, '
                             '0.0002 by default.')
    parser.add_argument('--opacity_reset_interval', type=int, default=3_000,
                        help='How frequently to reset opacity, 3_000 by default..')
    parser.add_argument('--percent_dense', type=float, default=0.01,
                        help='Percentage of scene extent (0--1) a point must exceed to be forcibly densified, '
                             '0.01 by default.')
    parser.add_argument('--lambda_dssim', type=float, default=0.2,
                        help='Influence of SSIM on total loss from 0 to 1, 0.2 by default.')

    parser.add_argument("--debug", default=True, action="store_true", help="Enables debug mode if you experience "
                                                                           "errors. If the rasterizer fails, "
                                                                           "a dump file is created that you may "
                                                                           "forward to us in an issue so we can take "
                                                                           "a look.")
    parser.add_argument("--debug_from", help="Debugging is slow. You may specify an iteration (starting from 0) after "
                                             "which the above debugging becomes active.")
    args = parser.parse_args()

    args.save_iterations.append(args.iterations)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
