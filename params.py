class TrainingParams:

    def __init__(self, args):
        self.source_path = args.source_path
        self.images = args.images
        self.model_path = args.model_path
        self.iterations = args.iterations
        self.test_iterations = args.test_iterations
        self.save_iterations = args.save_iterations
        self.checkpoint_iterations = args.checkpoint_iterations
        self.debug = args.debug
        self.debug_from = args.debug_from


class ModelParams:

    def __init__(self, args):
        self.sh_degree = args.sh_degree
        self.feature_lr = args.feature_lr
        self.opacity_lr = args.opacity_lr
        self.scaling_lr = args.scaling_lr
        self.rotation_lr = args.rotation_lr
        self.position_lr_init = args.position_lr_init
        self.position_lr_final = args.position_lr_final
        self.position_lr_delay_mult = args.position_lr_delay_mult
        self.position_lr_max_steps = args.position_lr_max_steps
        self.densification_interval = args.densification_interval
        self.densify_from_iter = args.densify_from_iter
        self.densify_until_iter = args.densify_until_iter
        self.densify_grad_threshold = args.densify_grad_threshold
        self.opacity_reset_interval = args.opacity_reset_interval
        self.percent_dense = args.percent_dense
        self.lambda_dssim = args.lambda_dssim