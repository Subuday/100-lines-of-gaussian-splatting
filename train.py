import os
import uuid
import numpy as np
import torch
import random
from argparse import ArgumentParser
from random import randint
from model import GaussianModel
from params import TrainingParams, ModelParams
from render import render
from scene import Scene
from utils.loss_utils import l1_loss, ssim


def prepare_output_dir(params):
    if not params.model_path:
        params.model_path = os.path.join("./output/", str(uuid.uuid4()))
    os.makedirs(params.model_path, exist_ok=True)


def train(training_params, model_params, device):
    prepare_output_dir(training_params)

    model = GaussianModel(model_params, device=device)
    scene = Scene(model, training_params, model_params)

    cameras = None
    for i in range(1, training_params.iterations + 1):
        if not cameras:
            cameras = scene.train_cameras.copy()
        camera = cameras.pop(randint(0, len(cameras) - 1))

        render_res = render(camera, model, debug=training_params.debug, device=device)

        rendered_image = render_res["image"]
        original_image = camera.original_image.to(device)
        loss = (1.0 - model_params.lambda_dssim) * l1_loss(rendered_image, original_image) + \
               model_params.lambda_dssim * (1.0 - ssim(rendered_image, original_image))
        loss.backward()
        print("Iteration: {}, Loss: {}".format(i, loss.item()))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--source_path")
    parser.add_argument("--images", default="images",
                        help="Alternative subdirectory for COLMAP images (images by default).")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--iterations", default=30_000)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--sh_degree", default=3)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train(TrainingParams(args), ModelParams(args), device)
