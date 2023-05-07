import argparse
import textwrap
from pathlib import Path

from huggingface_hub import HfApi, login


def get_repo_id(model_folder_path):
    """recipes/dns_icassp_2023/bsrnn => dns_icassp_2023_bsrnn"""
    model_name = model_folder_path.name
    data_name = model_folder_path.parent.name
    repo_id = f"haoxiangsnr/{data_name}_{model_name}"

    print(f"Converted model path {model_folder_path.as_posix()} to repo id {repo_id}")

    return f"{data_name}_{model_name}"


def main(args):
    # Login to HuggingFace Hub
    login(token="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")

    # Get HuggingFace API
    api = HfApi()

    model_folder_path = Path(args.model_folder_path).absolute()
    assert model_folder_path.exists(), f"{model_folder_path} does not exist"

    repo_url = api.create_repo(
        repo_id=get_repo_id(model_folder_path),
        exist_ok=True,
        private=True,
    )
    repo_id = repo_url.repo_id
    print(f"Created repo {repo_id}")

    # Upload README.md
    if (model_folder_path / "README.md").exists():
        print(f"Uploading README.md to {repo_id}")
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=(model_folder_path / "README.md"),
            path_in_repo="README.md",
        )

    # Upload model files and tensorboard logs
    print(f"Uploading model files to {repo_id}")
    api.upload_folder(
        folder_path=model_folder_path,
        repo_id=repo_id,
        ignore_patterns="*.tar",
        repo_type="model",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=textwrap.dedent(
            """\
            python tools/upload_model_hf.py -m recipes/dns_icassp_2023/bsrnn
            """
        )
    )
    parser.add_argument("-m", "--model_folder_path", type=str, default="bsrnn")
    args = parser.parse_args()
    main(args)
