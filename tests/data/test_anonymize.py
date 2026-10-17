import pytest

from fym1.data.anonymize import FAKE_NAMES, anonymize, assign_fake_names

NAMES = {"Nirina Rakoto": "Soa", "Tojo Randria": "Hasina"}


@pytest.mark.parametrize("written", ["Nirina", "nirin", "NIRINA", "Nirinaaa"])
def test_anonymize_replaces_name_in_body(written):
    assert anonymize(f"Salama ry {written}!", NAMES) == "Salama ry Soa!"


def test_anonymize_replaces_full_name_once():
    assert anonymize("Hoy i Nirina Rakoto", NAMES) == "Hoy i Soa"


def test_anonymize_replaces_surname_alone():
    assert anonymize("ry rakoto", NAMES) == "ry Soa"


def test_anonymize_keeps_short_prefixes_and_other_words():
    # "Nir" is under the 4-char prefix floor; "tsara" matches nobody
    assert anonymize("Nir tsara", NAMES) == "Nir tsara"


@pytest.mark.parametrize(
    "phone", ["034 12 345 67", "+261341234567", "+261 34 12 345 67", "0341234567", "+33 6 12 34 56 78"]
)
def test_anonymize_phone(phone):
    assert anonymize(f"Antsoy amin'ny {phone} rahampitso", NAMES) == "Antsoy amin'ny <TEL> rahampitso"


def test_anonymize_keeps_dates_and_prices():
    assert anonymize("Ny 2026-10-17 dia 15 000 Ar", NAMES) == "Ny 2026-10-17 dia 15 000 Ar"


def test_anonymize_email_and_url():
    text = "Alefaso any amin'ny nirina.r@example.mg na jereo https://example.mg/x?y=1 sy www.example.com"
    assert anonymize(text, NAMES) == "Alefaso any amin'ny <EMAIL> na jereo <URL> sy <URL>"


def test_anonymize_handle_becomes_consistent_fake_name():
    first = anonymize("Jereo @rabe_99 sy @rabe_99", NAMES)
    name = first.split()[1]

    assert name in FAKE_NAMES
    assert first == f"Jereo {name} sy {name}"


def test_anonymize_mention_of_participant_uses_their_fake_name():
    assert anonymize("@Nirina jereo ity", NAMES) == "Soa jereo ity"


def test_assign_fake_names_is_distinct_and_avoids_real_names():
    real = ["Soa Rabe", "Nirina Rakoto", "Tojo Randria"]
    fakes = assign_fake_names(real)

    assert set(fakes) == set(real)
    assert len(set(fakes.values())) == 3
    assert "Soa" not in fakes.values()


def test_assign_fake_names_is_stable():
    assert assign_fake_names(["A Be", "B Be"]) == assign_fake_names(["B Be", "A Be"])
