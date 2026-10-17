"""is_malagasy tests. All sentences are made up for the test; none come from real chats."""

import pytest

from fym1.eval import lang
from fym1.eval.lang import MALAGASY_LABELS, is_malagasy


@pytest.fixture(scope="session", autouse=True)
def glotlid_loaded():
    """Load GlotLID once per test session (first run downloads ~1.7 GB, then cached)."""
    return lang.glotlid()


@pytest.mark.parametrize(
    "text",
    [
        # Code-mixed everyday Malagasy (French verbs, texting spelling, emoji) counts.
        "Mety be zany, lancer-na fotsiny😂",
        "De ahoana?",
        "De aona ry zalahy?",
        "Mbola miasa ve ialahy?",
        "Ok fa aza adino ny rdv rahampitso",
        "Sali ahn",
        "Salama!",
        "Misaotra betsaka!",
        "haha zany ve",
        "Tsy haiko fa mety ihany",
        "Aiza ianao zao? Efa tonga ve?",
        "Je suis fatigué be zao, mody aho",
        "On se voit demain ve? Tsy afaka aho anio",
        "Mandeha any amin'ny réunion aho dia miverina",
        "Tena nice ilay film 👌",
        "Manao ahoana ianao?\nTsara ve ny weekend?",
        "Inona no vaovao?",
        "Andao hisakafo, tsy maintsy manana vola kely aloha",
    ],
)
def test_everyday_malagasy_is_malagasy(text):
    assert is_malagasy(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # Full French or English with no Malagasy is drift.
        "Ok merci, à demain",
        "Sure, I can help with that.",
        "Hello there",
        "Bonjour, comment ça va ?",
        "Je suis fatigué, je rentre à la maison.",
        "I've been busy all week, sorry for the late reply.",
        "Eh bien, on verra demain.",
        "That's a zany idea, but I love it and I think we should try it out this weekend with everyone.",
        "",
        "   ",
        "😂😂😂",
    ],
)
def test_french_english_or_empty_is_not_malagasy(text):
    assert is_malagasy(text) is False


def test_malagasy_labels_exist_in_model():
    model_labels = {label.removeprefix("__label__") for label in lang.glotlid().get_labels()}
    assert MALAGASY_LABELS <= model_labels
