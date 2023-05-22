def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # * base configures
    parser.add_argument("--basedir", 
                        type=str, 
                        default="dataset")
    parser.add_argument("--datatype", 
                        type=str, 
                        default="pokemon")
    parser.add_argument("--device", 
                        type=str, 
                        default="cuda")
    parser.add_argument("--gpu_id", 
                        type=int, 
                        default=0)
    parser.add_argument("--exp_name", 
                        type=str, 
                        default="debug")
    # * diffusion configures
    parser.add_argument("--img_size", 
                        type=int, 
                        default=128,
                        help="Height and Width of Images")
    parser.add_argument("--diffuse_step", 
                        type=int, 
                        default=1000)
    parser.add_argument("--ema_beta", 
                        type=float, 
                        default=0.995)
    # * train configures
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=16)
    parser.add_argument("--lr_init", 
                        type=int, 
                        default=1e-4)
    parser.add_argument("--max_warmup_step", 
                        type=int, 
                        default=500)
    parser.add_argument("--max_epoch", 
                        type=int, 
                        default=50)
    parser.add_argument("--save_epoch", 
                        type=int, 
                        default=10)
    parser.add_argument("--eval_epoch", 
                        type=int, 
                        default=10)
    parser.add_argument("--N_eval", 
                        type=int, 
                        default=10)
    args = parser.parse_args()

    # * update device with id
    if args.device == "cuda":
        args.device = f"{args.device}:{args.gpu_id}"
    return args
