"""Upload the merged dualyam_combined LeRobot dataset to the HF Hub.

Creates (if needed) the dataset repo jellyho/phase1_combined and uploads the
whole folder with upload_large_folder (resumable, multi-commit; good for ~49GB).
"""

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "jellyho/phase1_combined"
ROOT = Path("/home/yonsei_jell/dualyam_combined")


def main():
    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(ROOT),
    )
    print(f"DONE -> https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
