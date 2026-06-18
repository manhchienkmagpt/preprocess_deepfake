from utils import build_arg_parser, run_preprocess


def main() -> None:
    parser = build_arg_parser("Preprocess CelebDF-v2 videos with SCRFD face detection")
    args = parser.parse_args()

    run_preprocess(
        dataset_name="celebdf",
        input_root=args.input_root,
        output_root=args.output_root,
        scrfd_model=args.scrfd_model,
        img_size=args.img_size,
        seed=args.seed,
        real_dirs=("Celeb-real", "YouTube-real"),
        fake_dirs=("Celeb-synthesis",),
        frames_per_real_video=128,
        frames_per_fake_video=21,
    )


if __name__ == "__main__":
    main()
