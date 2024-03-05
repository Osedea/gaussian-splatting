import cv2
from pathlib import Path


_DATA_PATH = Path("data")


def main(
    video_filename: Path,
    output_path: Path,
    k: int = 10
):
    if not video_filename.exists():
        raise FileNotFoundError(f"Invalid video filename {video_filename}.")

    output_path.mkdir(parents=True, exist_ok=True)

    cam = cv2.VideoCapture(video_filename.as_posix())

    current_frame = 0
    while(True):
        ret, frame = cam.read()

        if not ret:
            break

        if not current_frame % k == 0:
            current_frame += 1
            continue

        output_filename = output_path / f"{current_frame}.jpg"
        cv2.imwrite(output_filename.as_posix(), frame)
        print(f"Writing {output_filename}.")

        current_frame += 1


if __name__ == "__main__":
    main(
        video_filename=(_DATA_PATH / "test.mp4"),
        output_path=(_DATA_PATH / "output"),
        k=1
    )
