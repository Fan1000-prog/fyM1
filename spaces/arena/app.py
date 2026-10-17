"""fyM1 arena: raters see one prompt and two unlabelled answers (base vs fyM1) and pick the more natural one.

Votes are appended to a local JSONL file that CommitScheduler pushes to the private dataset VOTES_REPO.
Env: ARENA_KEY (Space secret, signs session ids), HF_TOKEN (Space secret with write access to VOTES_REPO),
PAIRS_FILE (default pairs.jsonl), VOTES_REPO (default Fan1000-Prog/fyM1-votes; empty = keep votes local,
for development).
"""

import json
import os
import random
import tempfile
import threading
import uuid
from pathlib import Path

import gradio as gr
from huggingface_hub import CommitScheduler, HfApi
from votes import VoteBook, is_valid_session_id, load_pairs, new_session_id, pick_pair

# OWNER REVIEW: Malagasy below is a draft by the builder; the native-speaker owner fixes the wording.
INSTRUCTIONS = """\
## Iza no mamaly tsara kokoa?

Misy fanontaniana iray sy valiny roa (**A** sy **B**). Vakio izy roa, dia safidio izay **toa resaka olona \
tena izy kokoa**: teny malagasy fampiasa isan'andro. Mety ny mampifangaro teny frantsay, ny soratra \
fohy amin'ny SMS, ny "slang" ary ny emoji.

- **A** na **B**: io no tsara kokoa.
- **Mitovy**: tsy misy tsara kokoa noho ny iray.
- **Diso**: asio marika ny valiny misy hevitra diso na tsy marina, na inona na inona safidinao.

Tsy mila miditra (login). Tsy aseho hoe iza no nanoratra ny valiny tsirairay. Misaotra betsaka!
"""
UI_TEXT = {
    "prompt": "Fanontaniana",
    "same": "Mitovy",
    "wrong": "Diso (raha misy)",
    "saved": "Voaray, misaotra! Ity ny manaraka.",
    "done": "Vita daholo! Misaotra betsaka 🙏",
    "reload": "Misy olana kely: avereno alaina ny pejy (refresh).",
}

APP_DIR = Path(__file__).parent
ARENA_KEY = os.environ.get("ARENA_KEY", "")
if not ARENA_KEY:
    raise RuntimeError("Set the ARENA_KEY secret in the Space settings (any long random string)")
PAIRS = load_pairs(APP_DIR / os.environ.get("PAIRS_FILE", "pairs.jsonl"))
VOTES_REPO = os.environ.get("VOTES_REPO", "Fan1000-Prog/fyM1-votes")
VOTES_DIR = Path(tempfile.gettempdir()) / "fym1-votes"
VOTES_DIR.mkdir(exist_ok=True)
# One file per process, so a restarted Space never overwrites votes it already pushed.
VOTES_FILE = VOTES_DIR / f"votes-{uuid.uuid4().hex}.jsonl"


def start_vote_upload() -> threading.Lock:
    """Start pushing VOTES_DIR to the private dataset; returns the lock that guards writes to it."""
    if not VOTES_REPO:
        print(f"VOTES_REPO is empty: votes stay in {VOTES_DIR} and are not uploaded")
        return threading.Lock()
    scheduler = CommitScheduler(
        repo_id=VOTES_REPO,
        repo_type="dataset",
        folder_path=VOTES_DIR,
        path_in_repo="votes",
        private=True,
        every=2,  # minutes
    )
    if not HfApi().repo_info(VOTES_REPO, repo_type="dataset").private:
        raise RuntimeError(f"{VOTES_REPO} is public; make it private before collecting votes")
    return scheduler.lock


write_lock = start_vote_upload()
# Remembers votes since this process started; a session can vote at most once per pair.
vote_book = VoteBook(ARENA_KEY.encode(), max_votes_per_session=len(PAIRS))
rng = random.Random()


def save_vote(pair_id: str | None, choice: str, session_id: str, wrong: list[str]) -> bool:
    """Append the vote unless the vote book rejects it; True when it was saved."""
    if pair_id is None:  # nothing on screen: every pair already rated
        return False
    with write_lock:
        vote = vote_book.record(pair_id, choice, session_id, wrong=wrong)
        if vote is None:
            return False
        with VOTES_FILE.open("a", encoding="utf-8") as votes_file:
            votes_file.write(json.dumps(vote, ensure_ascii=False) + "\n")
    return True


def next_pair(session_id: str, status: str):
    """Screen values for the next unrated pair: pair id, prompt, answer A, answer B, cleared flags, status."""
    pair = pick_pair(PAIRS, session_id, vote_book.seen, rng)
    if pair is None:
        return None, "", "", "", [], UI_TEXT["done"]
    return pair["pair_id"], pair["prompt"], pair["a"], pair["b"], [], status


def start_session(session_id):
    if not is_valid_session_id(session_id, vote_book.key):  # first visit, or an id this server did not sign
        session_id = new_session_id(vote_book.key)
    return session_id, *next_pair(session_id, "")


def vote_for(choice: str):
    def handle(session_id, pair_id: str | None, wrong: list[str]):
        if not is_valid_session_id(session_id, vote_book.key):  # sent by the browser; never trust it
            raise gr.Error(UI_TEXT["reload"])
        status = UI_TEXT["saved"] if save_vote(pair_id, choice, session_id, wrong) else ""
        return next_pair(session_id, status)

    return handle


with gr.Blocks(title="fyM1 arena") as demo:
    # Kept in the browser so a page refresh keeps the same id (and cannot re-vote the same pair).
    # The id is signed by the server; ARENA_KEY as the storage secret lets it survive Space restarts.
    session_id = gr.BrowserState(None, storage_key="fym1-arena-session", secret=ARENA_KEY)
    pair_id = gr.State(None)

    gr.Markdown(INSTRUCTIONS)
    prompt = gr.Textbox(label=UI_TEXT["prompt"], interactive=False)
    with gr.Row():
        answer_a = gr.Textbox(label="A", interactive=False, lines=6)
        answer_b = gr.Textbox(label="B", interactive=False, lines=6)
    wrong = gr.CheckboxGroup(["A", "B"], label=UI_TEXT["wrong"])
    with gr.Row():
        buttons = {
            "A": gr.Button("A", variant="primary"),
            "same": gr.Button(UI_TEXT["same"]),
            "B": gr.Button("B", variant="primary"),
        }
    status = gr.Markdown()

    screen = [pair_id, prompt, answer_a, answer_b, wrong, status]
    # Private: votes come from the page only, not from scripted API calls.
    demo.load(start_session, inputs=session_id, outputs=[session_id, *screen], api_visibility="private")
    for choice, button in buttons.items():
        button.click(
            vote_for(choice), inputs=[session_id, pair_id, wrong], outputs=screen, api_visibility="private"
        )

if __name__ == "__main__":
    demo.launch()
