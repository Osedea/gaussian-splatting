from argparse import ArgumentParser
from pathlib import Path

import cv2


def main(video_filename: Path, output_path: Path, k: int = 10):
    if not video_filename.exists():
        raise FileNotFoundError(f"Invalid video filename {video_filename}.")

    output_path.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(video_filename.as_posix())

    current_frame = 0
    while True:
        ret, frame = cam.read()

        if not ret:
            break

        if not current_frame % k == 0:
            current_frame += 1
            continue

        output_filename = output_path / f"{str(current_frame).zfill(6)}.jpg"
        cv2.imwrite(output_filename.as_posix(), frame)
        print(f"Writing {output_filename}.")

        current_frame += 1


if __name__ == "__main__":
    parser = ArgumentParser("Extract frames from video file.")
    parser.add_argument("-v", "--video", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("-k", default=1, type=int)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.video.parent / str(args.k)
    args.output = args.output / "input"
    args.output.mkdir(exist_ok=True, parents=True)

    main(video_filename=args.video, output_path=args.output, k=args.k)
