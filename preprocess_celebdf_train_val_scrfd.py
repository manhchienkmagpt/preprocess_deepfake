from utils import build_arg_parser, run_preprocess_celebdf_train_val


def main() -> None:
    parser = build_arg_parser("Preprocess CelebDF-v2 train/val videos with SCRFD face detection")
    parser.add_argument(
        "--test_list",
        type=str,
        default=None,
        help="CelebDF test list txt to exclude from train/val. Defaults to input_root/List_of_testing_videos.txt",
    )
    args = parser.parse_args()

    run_preprocess_celebdf_train_val(
        input_root=args.input_root,
        output_root=args.output_root,
        scrfd_model=args.scrfd_model,
        img_size=args.img_size,
        seed=args.seed,
        test_list=args.test_list,
    )


if __name__ == "__main__":
    main()
