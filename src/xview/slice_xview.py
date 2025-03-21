import fire
from sahi.scripts.slice_coco import slice

MAX_WORKERS = 20
IGNORE_NEGATIVE_SAMPLES = False


def slice_xview(
    image_dir: str, dataset_json_path: str, output_dir: str, slice_size: int, overlap_ratio: float
):
    slice(
        image_dir=image_dir,
        dataset_json_path=dataset_json_path,
        output_dir=output_dir,
        slice_size=slice_size,
        overlap_ratio=overlap_ratio,
    )


if __name__ == "__main__":
    fire.Fire(slice_xview)
