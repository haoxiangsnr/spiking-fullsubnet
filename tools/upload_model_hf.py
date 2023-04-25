import argparse
from pathlib import Path

from huggingface_hub import HfApi, login


def get_repo_id(model_folder_path):
    """Convert config path to task flag.

    dns_icassp_2023/bsrnn => dns_icassp_2023_bsrnn

    """
    print(f"Convert model path to repo id: {model_folder_path.as_posix()}")

    model_name = model_folder_path.name
    data_name = model_folder_path.parent.name

    return f"haoxiangsnr/{data_name}_{model_name}"


def setup():
    login(token="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")


def main(args):
    setup()
    api = HfApi()

    model_folder_path = Path(args.model_folder_path).absolute()
    assert model_folder_path.exists(), f"{model_folder_path} does not exist"

    repo_id = get_repo_id(model_folder_path)
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True).repo_id

    # Upload README.md
    if (model_folder_path / "README.md").exists():
        api.upload_file(
            repo_id=repo_id,
            file_path=(model_folder_path / "README.md").as_posix(),
            file_name="README.md",
        )

    api.upload_folder(
        folder_path=model_folder_path.as_posix(),
        repo_id=repo_id,
        ignore_patterns="*.tar",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_folder_path", type=str, default="bsrnn")
    args = parser.parse_args()
    main(args)
