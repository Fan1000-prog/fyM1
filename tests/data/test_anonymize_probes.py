"""The PR #3 reviewers' probe inputs. Known limits are strict xfails, so fixing one is noticed."""

import pytest

from fym1.data.anonymize import Anonymizer, assign_fake_names, compare_key

PEOPLE = ["Nirina Rakotomalala", "Soa Randria", "Haja Andrianasolo", "Fy Rakoto", "Jo Be", "Hélène Rabe"]
REAL_PARTS = {compare_key(w) for name in PEOPLE for w in name.split()}


def run(text: str, people=PEOPLE) -> str:
    return Anonymizer(assign_fake_names(people))(text)


def leaked_parts(text: str) -> set[str]:
    return REAL_PARTS & {compare_key(w) for w in text.replace("-", " ").split()}


@pytest.mark.parametrize(
    "text",
    [
        "NIRINA", "RAKOTOMALALA", "SOA", "NIRINAKO", "Nirinako", "Nirinany", "an'i Nirina", "ry Soa",
        "ry Soa a", "Soako", "Soako a", "Soanay", "Fyko", "Soa ô", "SOAAA", "Soaaa", "Fyyy", "Niriiina",
        "Nirinaaaa", "Nirinà", "nirin", "Niri", "Rakotomalalaaa", "Andrianasolo", "Andrianasolo-ko",
        "Salama helene", "Salama Hélène", "Salama Hélène", "Soa ve?", "Fy ve", "fy be",
    ],
)  # fmt: skip
def test_participant_names_do_not_leak(text):
    assert not leaked_parts(run(text))


@pytest.mark.parametrize(
    "text", ["@Haja Rabe", "@Haja Andrianasolo", "@Hery Rakotobe", "@Hery", "miaraka amin'i @Fidy Ravelojaona"]
)
def test_display_name_mentions_become_fake_full_names(text):
    out = run(text)

    assert "@" not in out
    assert not {"haja", "andrianasolo", "hery", "rakotobe", "fidy", "ravelojaona", "rabe"} & {
        compare_key(w) for w in out.split()
    }


def test_mention_of_participant_full_name_uses_their_fake():
    names = assign_fake_names(PEOPLE)

    assert Anonymizer(names)("@Haja Andrianasolo salama") == names["Haja Andrianasolo"] + " salama"


def test_mention_stops_at_lowercase_words():
    out = run("@Hery Rakotobe salama tompoko")

    assert out.endswith(" salama tompoko")


@pytest.mark.parametrize(
    "text",
    [
        "034 123 45 67", "034 12 34 567", "032.12.345.67", "033-12-345-67", "020 22 345 67", "0202234567",
        "(+261) 34 12 345 67", "+261 (0)34 12 345 67", "00261 34 12 345 67", "34 12 345 67", "O34 12 345 67",
        "o34 12 345 67", "034/12/345/67", "034  12  345  67", "+261-34-12-345-67", "261341234567",
        "+33 6 12 34 56 78", "06 12 34 56 78", "+1 (555) 123-4567", "034 12 - 345 67", "tel: 034.12.345.67",
    ],
)  # fmt: skip
def test_phone_formats_become_tel(text):
    out = run(text)

    assert "<TEL>" in out
    assert not any(c.isdigit() for c in out)


@pytest.mark.parametrize(
    "text", ["034 12 345 67 / 033 11 222 33", "0341234567 0331122233", "034 12 345 67."]
)
def test_several_phones(text):
    assert not any(c.isdigit() for c in run(text))


@pytest.mark.parametrize(
    "text",
    [
        "15 000 Ar", "15.000 Ar", "1 500 000 Ar", "Ar 25000", "2026-10-17", "ny 12/10/2026 amin'ny 14h30",
        "100 000 000", "32 000 000 Ar", "35 000 Ar", "345 000 000 Ar", "39 500 000", "ny 03/10/2026 amin'ny 14h30",
    ],
)  # fmt: skip
def test_prices_and_dates_are_kept(text):
    assert run(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("whatsapp: 0341234567", "whatsapp: <TEL>"),
        ("WA 034 12 345 67", "WA <TEL>"),
        ("viber +261 34 12 345 67", "viber <TEL>"),
        ("ig: @nirina.rk", "ig: <HANDLE>"),
        ("insta nirina_rk", "insta <HANDLE>"),
        ("ig: nirina.rk", "ig: <HANDLE>"),
        ("snap: nirinark", "snap: <HANDLE>"),
        ("ig: hery.rakotobe", "ig: <HANDLE>"),
        ("insta hery_rakotobe", "insta <HANDLE>"),
        ("tg: @hery99", "tg: <HANDLE>"),
        ("snap ve?", "snap ve?"),
    ],
)
def test_labelled_handles(text, expected):
    assert run(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "tiktok.com/@nirina", "t.me/nirina", "instagram.com/nirina.rk", "linktr.ee/nirina", "nirina.site/x",
        "wa.me/+261341234567", "linktr.ee/hery.rakotobe", "HTTPS://Exämple.mg/ñ",
    ],
)  # fmt: skip
def test_links_become_url(text):
    assert run(text) == "<URL>"


@pytest.mark.parametrize(
    "text",
    [
        "nirina+tag@gmail.com", "nirina.rk+fy@yahoo.fr", "nirina @ gmail . com", "nirina[at]gmail.com",
        "nirina@gmail", "NIRINA@GMAIL.COM", "hery.rakotobe[at]gmail.com", "hery (at) yahoo (dot) fr",
    ],
)  # fmt: skip
def test_emails_become_email(text):
    assert run(text) == "<EMAIL>"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Lot II M 85 Ter Ankadifotsy", "<ADDRESS> Ankadifotsy"),
        ("lot IVG 12 bis", "<ADDRESS>"),
        ("Lot 12", "<ADDRESS>"),
        ("ao amin'ny Lot VA 12", "ao amin'ny <ADDRESS>"),
        ("ny lot", "ny lot"),
        ("lotery", "lotery"),
    ],
)
def test_lot_addresses(text, expected):
    assert run(text) == expected


@pytest.mark.parametrize("text", ["andriana", "ny trano", "Mbola tsy vita raha ny volana farany"])
def test_common_words_survive(text):
    assert run(text) == text


@pytest.mark.parametrize(
    "people",
    [["Soa", "Soa Rabe"], ["Rabe", "Rabe Rabe"], ["Hasina Rabe", "Soa Hasina", "Rabe Soa"], ["A"],
     ["Jean-Pierre Rakoto"], ["Fara", "Faralahy"]],
)  # fmt: skip
def test_assign_fake_names_handles_overlapping_contacts(people):
    fakes = assign_fake_names(people)

    assert len(set(fakes.values())) == len(people)
    real = {compare_key(w) for name in people for w in name.replace("-", " ").split()}
    assert not real & {compare_key(w) for fake in fakes.values() for w in fake.split()}


@pytest.mark.parametrize("text", ["Facebook user ve", "user", "ilay user"])
def test_deleted_account_names_are_not_names(text):
    assert run(text, people=["Facebook User", "Utilisateur Facebook", "Nirina Rakoto"]) == text


@pytest.mark.parametrize("text", ["Jean-Pierre", "jean pierre", "Jean", "Pierre"])
def test_hyphenated_first_name(text):
    out = run(text, people=["Jean-Pierre Rakoto"])

    assert not {"jean", "pierre"} & {compare_key(w) for w in out.replace("-", " ").split()}


KNOWN_LIMITS = [
    "Ni-ri-na",  # spelled out with separators
    "N i r i n a",
    "N1r1na",  # digits for letters
    "zero trois quatre douze",  # phone spelled in words
    "aotra telo efatra roa",
    "034 douze 345 67",
    "034 l2 345 67",  # letter l for digit 1
    "Cité 67Ha porte 12",  # addresses other than "Lot …"
    "Villa 23 Ivandry",
]


@pytest.mark.xfail(strict=True, reason="known limit, see docs/anonymisation.md; owner review is the backstop")
@pytest.mark.parametrize("text", KNOWN_LIMITS)
def test_known_limits(text):
    out = run(text)
    assert out != text and not leaked_parts(out)
