"""
push_to_space.py -- upload the freshly trained artifacts to a Hugging Face Space.
Run by the GitHub Action after train_and_build_app.py. Needs env vars HF_TOKEN, HF_SPACE.
The Space auto-rebuilds when these files change, so the live app refreshes itself.
"""
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
space = os.environ["HF_SPACE"]

for f in ["wc_model.joblib", "retro_predictor.html"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f, repo_id=space, repo_type="space",
                    commit_message="daily auto-update")
    print("pushed", f)
print("done ->", space)
