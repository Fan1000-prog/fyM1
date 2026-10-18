import pytest

from fym1.data.anonymize import (
    COMMON_WORDS,
    FIRST_NAMES,
    SURNAMES,
    Anonymizer,
    anonymize,
    assign_fake_names,
    fold,
)

NAMES = {"Hanta Rakoto": "Soa Rabe", "Tojo Randria": "Hasina Ravelo"}


@pytest.mark.parametrize("written", ["Hanta", "hant", "HANTA", "Hantaaa"])
def test_anonymize_replaces_name_in_body(written):
    assert anonymize(f"Salama ry {written}!", NAMES) == "Salama ry Soa!"


def test_anonymize_replaces_full_name_once():
    assert anonymize("Hoy i Hanta  Rakoto", NAMES) == "Hoy i Soa Rabe"


@pytest.mark.parametrize("written", ["rakoto", "rakot"])  # "rakot": a 5-letter prefix
def test_anonymize_replaces_surname_alone(written):
    assert anonymize(f"ry {written}", NAMES) == "ry Rabe"


def test_anonymize_keeps_short_prefixes_and_other_words():
    # "Nir" is under the 4-letter prefix floor; "tsara" matches nobody
    assert anonymize("Nir tsara", NAMES) == "Nir tsara"


def test_common_words_never_prefix_match_names():
    names = {"Mbolatiana Raharison": "Soa Rabe", "Volamena Faratiana": "Hasina Ravelo"}
    text = "Mbola tsy vita raha ny volana farany"

    assert anonymize(text, names) == text


def test_exact_name_part_is_replaced_even_if_common_and_counted():
    anonymizer = Anonymizer({"Vola Rakoto": "Soa Rabe"})

    assert anonymizer("Tsy misy vola") == "Tsy misy Soa"
    assert anonymizer.common_word_replacements == 1


@pytest.mark.parametrize("first", ["Jo", "Fy", "Soa"])
def test_short_name_parts_are_replaced(first):
    names = {f"{first} Rakotobe": "Hasina Ravelo"}

    assert anonymize(f"Aiza i {first}? {first.lower()}!", names) == "Aiza i Hasina? Hasina!"


def test_short_function_words_are_protected():
    names = {"Ny Aina Rabe": "Hasina Ravelo"}

    assert anonymize("ny trano an'i Aina", names) == "ny trano an'i Ravelo"


@pytest.mark.parametrize("written", ["Hélène", "helene", "HELENE", "Hélè", "Hélène"])
def test_anonymize_ignores_accents(written):
    names = {"Hélène Rakoto": "Soa Rabe"}

    assert anonymize(f"Salama {written}", names) == "Salama Soa"


def test_anonymize_full_name_ignores_accents():
    assert anonymize("helene rakoto", {"Hélène Rakoto": "Soa Rabe"}) == "Soa Rabe"


@pytest.mark.parametrize(
    "phone",
    [
        "034 12 345 67",
        "034 123 45 67",
        "034 12 34 567",
        "020 22 345 67",
        "0341234567",
        "034.12.345.67",
        "+261341234567",
        "+261 34 12 345 67",
        "(+261) 34 12 345 67",
        "+261 (0)34 12 345 67",
        "00261 34 12 345 67",
        "+33 6 12 34 56 78",
    ],
)
def test_anonymize_phone(phone):
    assert anonymize(f"Antsoy amin'ny {phone} rahampitso", NAMES) == "Antsoy amin'ny <TEL> rahampitso"


def test_anonymize_phone_next_to_other_numbers():
    assert anonymize("034 12 345 67 5 olona", NAMES) == "<TEL> 5 olona"


@pytest.mark.parametrize("text", ["Ny 2026-10-17 dia 15 000 Ar", "1 500 000 Ar", "ora 18:30", "01-10-2026"])
def test_anonymize_keeps_dates_and_prices(text):
    assert anonymize(text, NAMES) == text


def test_anonymize_email_and_url():
    text = "Alefaso any amin'ny hanta.r@example.mg na jereo https://example.mg/x?y=1 sy www.example.com"
    assert anonymize(text, NAMES) == "Alefaso any amin'ny <EMAIL> na jereo <URL> sy <URL>"


@pytest.mark.parametrize(
    "link", ["facebook.com/hanta.rakoto", "fb.com/x", "fb.me/x", "m.me/hanta", "bit.ly/3abc", "example.mg"]
)
def test_anonymize_schemeless_links(link):
    assert anonymize(f"jereo {link} fa tsara", NAMES) == "jereo <URL> fa tsara"


@pytest.mark.parametrize("address", ["Lot II M 85 Ter", "lot IVG 12 Bis", "Lot 67 HA", "LOT VK 15"])
def test_anonymize_lot_address(address):
    assert anonymize(f"Any {address} ihany aho", NAMES) == "Any <ADDRESS> ihany aho"


def test_anonymize_keeps_lot_without_number():
    assert anonymize("un lot de chaussures", NAMES) == "un lot de chaussures"


def test_anonymize_handle_becomes_consistent_fake_name():
    first = anonymize("Jereo @rabe_99 sy @rabe_99", NAMES)
    name = first.split()[1]

    assert name in FIRST_NAMES
    assert first == f"Jereo {name} sy {name}"


def test_anonymize_mention_of_participant_uses_their_fake_name():
    assert anonymize("@Hanta jereo ity", NAMES) == "Soa jereo ity"


def test_fold_strips_accents_and_case():
    assert fold("HÉLÈNE") == fold("helene") == "helene"


def test_common_words_come_from_wikipedia_list():
    assert {"ny", "mbola", "raha", "vola", "volana", "farany"} <= set(COMMON_WORDS)


def test_assign_fake_names_is_distinct_and_avoids_real_names():
    real = ["Soa Rabe", "Hanta Rakoto", "Tojo Randria"]
    fakes = assign_fake_names(real)

    assert set(fakes) == set(real)
    assert len(set(fakes.values())) == 3
    assert not {"soa", "rabe"} & {fold(w) for fake in fakes.values() for w in fake.split()}


def test_assign_fake_names_is_stable():
    assert assign_fake_names(["A Be", "B Be"]) == assign_fake_names(["B Be", "A Be"])


def test_assign_fake_names_survives_contacts_covering_the_pool():
    real = [f"{first} {last}" for first, last in zip(FIRST_NAMES, SURNAMES * 3)] + list(SURNAMES)

    fakes = assign_fake_names(real)

    assert len(set(fakes.values())) == len(real)
    assert all(fakes.values())
    real_words = {fold(w) for name in real for w in name.split()}
    assert not real_words & {fold(w) for fake in fakes.values() for w in fake.split()}


def test_assign_fake_names_beyond_pool_size_stays_distinct():
    real = [f"Olona{i} X" for i in range(len(FIRST_NAMES) * len(SURNAMES) + 5)]

    assert len(set(assign_fake_names(real).values())) == len(real)
