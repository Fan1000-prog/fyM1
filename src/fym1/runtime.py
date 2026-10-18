"""Notebook runtime differences (Colab vs Kaggle) in one place.

Notebooks need the HF token before they can download this package from the private checkpoint repo, so
their bootstrap cell carries a verbatim copy of `hf_token` and its helpers (`MISSING_TOKEN`, `on_kaggle`,
`_secret`; tests/test_notebook_runtime.py checks the copy); everything else imports from here.
Kaggle is checked first: its GPU image is built from Colab's, so `google.colab` may import there too.
"""

import os
import tempfile

KAGGLE_WORKING = "/kaggle/working"
MISSING_TOKEN = ("no HF_TOKEN found: on Kaggle attach it under Add-ons → Secrets; on Colab add it in the "
                 "Secrets pane with notebook access; elsewhere set the HF_TOKEN environment variable")


def on_kaggle() -> bool:
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return True
    try:
        import kaggle_secrets  # noqa: F401  (only installed in Kaggle kernels)
    except ImportError:
        return False
    return True


def _secret(get) -> str | None:
    try:
        return get("HF_TOKEN")
    except Exception:  # missing or not attached: the caller raises one clear error; never echo details
        return None


def hf_token() -> str:
    """HF token from Kaggle secrets, else Colab secrets, else the HF_TOKEN env var. Never print it."""
    if on_kaggle():
        from kaggle_secrets import UserSecretsClient

        token = _secret(UserSecretsClient().get_secret)
    else:
        try:
            from google.colab import userdata
        except ImportError:
            token = os.environ.get("HF_TOKEN")
        else:
            token = _secret(userdata.get)
    if not token:
        raise RuntimeError(MISSING_TOKEN)
    return token


def output_dir() -> str:
    """Notebook-only: where wanted outputs go. On Kaggle, /kaggle/working is saved with the run (~20 GB cap)
    and is what `kaggle kernels output` downloads, so keep it for small results only."""
    return KAGGLE_WORKING if on_kaggle() else "/content"


def scratch_dir() -> str:
    """Notebook-only: large throwaway files (model downloads, checkpoints, merged models); never saved as
    Kaggle output."""
    if not on_kaggle():
        return "/content"
    path = os.path.join(tempfile.gettempdir(), "fym1")
    os.makedirs(path, exist_ok=True)
    return path
