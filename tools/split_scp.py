from pathlib import Path


def main(scp_file_path):
    scp_file_path = Path(scp_file_path).resolve()
    assert scp_file_path.exists(), f"{scp_file_path} does not exist"

    lines = open(scp_file_path.as_posix(), "r").readlines()

    output = []
    for line in lines:
        line = line.strip()
        file_paths = line.split(" ")
        output_file_path = file_paths[1]
        output_file_path = Path(output_file_path).resolve()
        output.append(output_file_path.as_posix())

    # replace the **.scp to **.0.scp
    output_scp_file_path = scp_file_path.parent / f"{scp_file_path.stem}.0.scp"

    with open(output_scp_file_path.as_posix(), "w") as f:
        for line in output:
            # print(line)
            f.write(f"{line}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        epilog="Example: python tools/split_scp.py --scp_file_path ./data/train/wav.scp"
    )
    parser.add_argument("--scp_file_path", type=str, required=True)
    args = parser.parse_args()

    main(args.scp_file_path)
