import os


def define_args(args):
    # Get the path of the current directory
    dirname = os.path.dirname(os.path.realpath(__file__))

    args.add_argment("--model_dir", type=str, default=os.path.join(dirname, "model"))
    args.add_argment("--model_name", type=str, default="best_8s.engine")
    args.add_argment("--verbose", type=bool, default=False)

    args.add_argument("--wait", type=int, default=0, help="wait time")
    args.add_argument("--toggle_key", type=str, default="y", help="toggle key")

    args.add_argument("--Kp", type=float, default=0.35, help="Kp")  # proporcional to distance 0.4 nimble 0.1 slack
    args.add_argument("--Ki", type=float, default=0.02, help="Ki")  # integral accumulator 0.04 explosive 0.01 composed
    args.add_argument("--Kd", type=float, default=0.3, help="Kd")  # derivative absorber 0.4 stiff 0.1 soft

    return args
