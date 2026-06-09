"""Upload the annotated insert-mouse-battery dataset to the HF Hub as a proper
LeRobot v3.0 dataset (data/videos + LeRobot card + v3.0 codebase tag).

Lesson from the combined dataset: a raw folder upload is NOT enough — the Hub
needs the LeRobot dataset card (README with tags/configs) AND the v3.0 tag, or
LeRobotDataset(repo_id) fails with RevisionNotFoundError.
"""

import pathlib

from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.datasets.utils import create_lerobot_dataset_card

REPO = "jellyho/insert-mouse-battery_annotated"
ROOT = pathlib.Path("/NHNHOME/WORKSPACE/0526040008_A/jellyho/insert-mouse-battery_annotated")


def main():
    api = HfApi()
    api.create_repo(repo_id=REPO, repo_type="dataset", exist_ok=True, private=False)

    # ~184GB → resumable multi-commit upload.
    api.upload_large_folder(repo_id=REPO, repo_type="dataset", folder_path=str(ROOT))

    # LeRobot card from the CURRENT info.json (now includes rl_token/base_action).
    meta = LeRobotDatasetMetadata(REPO, root=ROOT)
    card = create_lerobot_dataset_card(tags=None, dataset_info=meta.info, license="apache-2.0", repo_id=REPO)
    card.push_to_hub(repo_id=REPO, repo_type="dataset")

    # Codebase version tag (the thing LeRobotDataset loads by).
    try:
        api.delete_tag(REPO, tag=CODEBASE_VERSION, repo_type="dataset")
    except Exception:
        pass
    api.create_tag(REPO, tag=CODEBASE_VERSION, repo_type="dataset")
    print(f"DONE -> https://huggingface.co/datasets/{REPO}")


if __name__ == "__main__":
    main()
