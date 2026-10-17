"""Language check: does a reply count as (everyday) Malagasy?

A reply only "drifts" when it is full French or English with no Malagasy in it.
Code-mixed lines ("Mety be zany, lancer-na fotsiny😂") must pass, so GlotLID alone
is not enough: it misreads short texting lines ("De ahoana?" -> Romanian). A reply is
Malagasy when GlotLID says so OR enough of its words are common Malagasy words.
"""

import re
import unicodedata
from functools import lru_cache

import fasttext
from huggingface_hub import hf_hub_download

GLOTLID_REPO = "cis-lmu/glotlid"
GLOTLID_REVISION = "85cd6716494360367b75f642b5bc78667605d0b4"  # pinned for rerunnable numbers

# Plateau Malagasy plus the Malagasy dialect labels GlotLID uses for short texts.
MALAGASY_LABELS = frozenset({"plt_Latn", "xmv_Latn", "tdx_Latn", "skg_Latn"})

# Common Malagasy words, including texting forms ("zany", "aona", "ahn", "zah").
# Words that are also English/French words or common names are left out on purpose
# ("be", "no", "any", "sa", "lay", "marina", "e", "eh", "na", "ho", "mba", "aloha",
# "tonga", "tia", "ko", "vita", "mila", "koa"), so a French or English sentence never
# passes on a look-alike word.
MALAGASY_WORDS = frozenset(
    """
    ny ilay ve fa dia ka ary sy amin amina aminy amin'ny tamin'ny an'ny kosa satria raha
    rehefa nefa kanefa sady fotsiny ihany avy eto ety ao eo izao zao izany zany ireo io
    ity iny iry ireto inona iza aiza oviana ahoana aona nahoana firy ohatrinona tsy tsia
    eny ia mbola efa tena loatra kely tsara ratsy mety tokony afaka tiako fantatro haiko
    hoy aho ahy ianao anao ianareo anareo izy azy izahay izaho zah isika antsika ialahy
    ry zalahy ahn rahampitso omaly anio androany rahariva alina maraina hariva misaotra
    betsaka veloma salama manao mandeha miasa mipetraka mahita mihinana misotro matory
    mody miverina mankany avia andao andeha vao vaovao mihitsy sahady foana tsotra olona
    namana trano sakafo vola fotoana zavatra resaka ndray indray angamba toa hafa
    rehetra daholo misy tsisy aza adino nao bé maintsy manana hisakafo azafady tompoko
    manahoana mandria fahizay
    """.split()
)

# Share of words that must be Malagasy when GlotLID disagrees. Low on purpose:
# one Malagasy word in a short French line is code-mixing, but one look-alike
# word in a long English paragraph is not.
MIN_MALAGASY_WORD_SHARE = 0.1

# Runs of letters; an apostrophe inside keeps "amin'ny" / "i've" whole. Hyphens split "lancer-na".
_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


@lru_cache(maxsize=1)
def glotlid():
    """Load the GlotLID fastText model once (downloaded and cached by huggingface_hub).

    Returns fasttext's model object; its class is private, so it is not annotated.
    """
    path = hf_hub_download(GLOTLID_REPO, "model.bin", revision=GLOTLID_REVISION)
    return fasttext.load_model(path)


def _glotlid_label(text: str) -> str:
    labels, _ = glotlid().predict(" ".join(text.split()))  # fastText rejects newlines
    return labels[0].removeprefix("__label__")


# All-caps words this short are acronyms ("NY", "IO", "IA"), not Malagasy words.
_MAX_ACRONYM_LEN = 3


def _counts_as_malagasy(word: str) -> bool:
    if word.isupper() and len(word) <= _MAX_ACRONYM_LEN:
        return False
    return word.lower() in MALAGASY_WORDS


def _malagasy_word_share(words: list[str]) -> float:
    return sum(_counts_as_malagasy(word) for word in words) / len(words)


def is_malagasy(text: str) -> bool:
    """True if the text is Malagasy, code-mixed with French/English or not.

    Text with no words at all (empty, emoji only) is not Malagasy.
    """
    text = unicodedata.normalize("NFC", text).replace("\u2019", "'")
    words = _WORD.findall(text)
    if not words:
        return False
    if _glotlid_label(text) in MALAGASY_LABELS:
        return True
    return _malagasy_word_share(words) >= MIN_MALAGASY_WORD_SHARE
