"""Push the locally-built seal_mini dataset to HF (private).
Run yourself:  uv run python .diag/push_mini.py
(Separate from the build because the build reads gwanwoo13's private lustre data;
this step sends the derived subset to an external service — your call.)
"""
from huggingface_hub import HfApi, create_repo
OUT = "/lustre/jellyho/seal_mini"
REPO = "jellyho/seal-water-bottle-cap_mini30"
create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
HfApi().upload_folder(folder_path=OUT, repo_id=REPO, repo_type="dataset",
                      commit_message="seal_v3 mini: 10 success + 10 failure + 10 intervention")
print(f"pushed -> https://huggingface.co/datasets/{REPO}")
