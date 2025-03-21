"""
trainvalsplit.py is a script that splits an MS COCO formatted dataset into train and val
partitions. For sample usage, run from command line:

Example:
    python trainvalsplit.py --help
"""

# Standard Library imports:
import argparse
import sys
import subprocess
from pathlib import Path

# h4dlib imports:
# import _import_helper  # pylint: disable=unused-import # noqa: F401
# PROJ_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
try:
    PROJ_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
except subprocess.CalledProcessError:
    print("Error: Not inside a Git repository.")
    PROJ_ROOT = None

print("PROJ ROOOOOOT: ", PROJ_ROOT)
h4dlib_path = (Path(PROJ_ROOT) / "src/utilities/h4dlib").resolve()
print(h4dlib_path)
assert h4dlib_path.exists()
if str(h4dlib_path) not in sys.path:
    sys.path.append(str(h4dlib_path))


from h4dlib.data.cocohelpers import CocoClassDistHelper, CocoJsonBuilder, split

# Used to check the results of the split--all classes in both splits
# should have at least this many annotations:
_CLASS_COUNT_THRESHOLD = 0


def create_split(
    input_json: Path,
    output_path: Path,
    output_json_name: str,
    seed: int,
    test_size: float = 0.2,
) -> CocoClassDistHelper:
    """
    Creates train/val split for the coco-formatted dataset defined by input_json.

    params:
        input_json: full path or Path object to coco-formatted input json file.
        output_path: full path or Path object to directory where outputted json will be
        saved. output_json_name:
    """
    coco = CocoClassDistHelper(input_json)
    train_img_ids, val_img_ids = split(
        coco.img_ids, test_size=test_size, random_state=seed
    )
    train_counts, train_percents = coco.get_class_dist(train_img_ids)
    val_counts, val_percents = coco.get_class_dist(val_img_ids)

    # Generate coco-formatted json's for train and val:
    def generate_coco_json(coco, split_type, img_ids):
        coco_builder = CocoJsonBuilder(
            coco.cats,
            dest_path=output_path,
            dest_name=output_json_name.format(split_type),
        )
        for idx, img_id in enumerate(img_ids):
            coco_builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
        coco_builder.save()

    generate_coco_json(coco, "train", train_img_ids)
    generate_coco_json(coco, "val", val_img_ids)
    return coco


def verify_output(
    original_coco: CocoClassDistHelper, output_path: Path, output_json_name: str
) -> None:
    """
    Verify that the outputted json's for the train/val split can be loaded, and
    have correct number of annotations, and minimum count for each class meets
    our threshold.
    """

    def verify_split_part(output_json_name, split_part):
        json_path = output_path / output_json_name.format(split_part)
        print(f"Checking if we can load json via coco api:{json_path}...")
        coco = CocoClassDistHelper(json_path)
        counts, _ = coco.get_class_dist()
        assert min(counts.values()) >= _CLASS_COUNT_THRESHOLD, (
            f"min class count ({min(counts.values())}) is "
            + f"lower than threshold of {_CLASS_COUNT_THRESHOLD}"
        )
        print(f"{split_part} class counts: ", counts)
        return coco

    train_coco = verify_split_part(output_json_name, "train")
    val_coco = verify_split_part(output_json_name, "val")
    assert len(original_coco.imgs) == len(train_coco.imgs) + len(
        val_coco.imgs
    ), "Num Images in original data should equal sum of imgs in splits."
    assert len(original_coco.anns) == len(train_coco.anns) + len(
        val_coco.anns
    ), "Num annotations in original data should equal sum of those in splits."


def main(args: argparse.Namespace):
    """
    Creates train/val split and verifies output.
    params:
        opt: command line options (there are none right now)
        output_json_name: format-string of output file names, with a '{}'
            style placeholder where split type will be inserted.
    """
    input_json = Path(args.input_json).resolve()
    assert input_json.exists(), str(input_json)
    assert input_json.is_file(), str(input_json)

    output_path = Path(args.output_dir).resolve()
    assert output_path.is_dir(), str(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    output_json_name = input_json.stem.replace("_full", "") + "_{}.json"
    original_coco = create_split(
        input_json, output_path, output_json_name, args.seed, args.val_split_size
    )
    verify_output(original_coco, output_path, output_json_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_split_size", type=float, default=0.2)
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. Use split_search.py to find a seed that generates a good split",
    )
    parser.add_argument("--input_json", type=Path, help="Input json path")
    parser.add_argument("--output_dir", type=Path, help="Path to output json")
    args = parser.parse_args()
    main(args)
