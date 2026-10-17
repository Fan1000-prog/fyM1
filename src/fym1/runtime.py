"""Notebook runtime differences (Colab vs Kaggle) in one place.

Notebooks need the HF token before they can download this package from the private checkpoint repo, so
their first cell must carry a verbatim copy of `hf_token`; everything else imports it from here.
"""

import os

KAGGLE_WORKING = "/kaggle/working"


def hf_token() -> str:
    """HF token from Colab secrets, else Kaggle secrets, else the HF_TOKEN env var. Never print it."""
    try:
        from google.colab import userdata
    except ImportError:
        pass
    else:
        return userdata.get("HF_TOKEN")
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        pass
    else:
        return UserSecretsClient().get_secret("HF_TOKEN")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("no HF_TOKEN in Colab secrets, Kaggle secrets or the environment")
    return token


def work_dir() -> str:
    """Writable scratch directory: /kaggle/working on Kaggle, /content on Colab."""
    return KAGGLE_WORKING if os.path.isdir(KAGGLE_WORKING) else "/content"
