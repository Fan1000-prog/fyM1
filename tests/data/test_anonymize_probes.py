"""The PR #3 reviewers' probe inputs. Known limits are strict xfails, so fixing one is noticed."""

import pytest

from fym1.data.anonymize import FIRST_NAMES, SURNAMES, Anonymizer, assign_fake_names, compare_key

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
    "tel34 12 345 67",  # glued mobile without the leading 0
    "telO34 12 345 67",  # glued, letter O typed for the 0
]


@pytest.mark.xfail(strict=True, reason="known limit, see docs/anonymisation.md; owner review is the backstop")
@pytest.mark.parametrize("text", KNOWN_LIMITS)
def test_known_limits(text):
    out = run(text)
    assert out != text and not leaked_parts(out)


# --- final review (PR #3 round 3) -------------------------------------------------------

FINAL_PEOPLE = ["Nirina Rakotomalala", "Fara Randria", "Tojo Rasolofonirina", "Mialy Razafindrakoto",
                "Hery Andriamanana", "Fan Rabearison"]  # fmt: skip
FINAL_NAMES = assign_fake_names(FINAL_PEOPLE)

EVERYDAY_LINES = """De aona ry zalahy?
Mety be zany, lancer-na fotsiny😂
Mbola miasa ve ialahy?
Ok fa aza adino ny rdv rahampitso
Hoatrinona ilay kiraro? 15 000 Ar ve?
2 500 Ar ny mofo gasy androany
Ho avy aho amin'ny 3h
Amin'ny 14h30 ny fivoriana, ny 12/10/2026
Ny 17 oktobra 2026 no hanao fety
Mila trano, t.s.a fa ataovy haingana
Mividy voatabia, tongolo, sakay, etc.
Tsara be ny volana farany
Efa vita ve ny devoir-nao?
Lafo be ny vary, 3 000 Ar ny kapoaka
Mandeha any Antsirabe izahay rahampitso maraina
Taiza ianao omaly hariva?
Misaotra betsaka tompoko 🙏
Tsy haiko loatra fa andramo ihany
Aiza ny lakile? Tadiavo ao ambony latabatra
Ny bus 194 dia mandalo eo Analakely
Rdv @15h eo Ankorondrano
1 000 000 000 Fmg taloha io
Ny score dia 3-0, faly be aho!
Nahazo 18/20 aho tamin'ny maths
Mila 2 kg sy 500 g aho
Ny laharana 5, 6, 7, 8, 9 no tafita
01/02/2026 03/04 no daty roa
Mety tsara.Be ny fitiavako anao
Hitako tao amin'ny fb 2 andro lasa
Ig: tsy mampiasa aho
Alefaso amin'ny whatsapp 5mn aloha
Andao hijery film amin'ny 20h
Ny famerenana amin'ny 1er novembre
Efa nandoa 50.000 Ar aho tamin'ny MVola
Tel aloha fa tsy afaka manoratra aho
Mandehana mody ianareo fa efa alina
Ok fa aza adino, A.S.A.P azafady
Ny pp no tsy mety
Mahafinaritra ny fety, misaotra!
Haha tena mampihomehy ilay video
Vao 7h ny ora dia efa tonga aho
Ny RN7 dia mbola tapaka
Nisy 3 olona no maty, mampalahelo
Tsy afaka aho fa marary ny zanako
Andao isika hiara-hisakafo
lol mety be, à demain
1 000 000 000 Ar
5 000 000 000 Ar
Ar 1 000 000 000
10 000 000 000
Mety.Be
mety.be ny
ok.Me
farany""".splitlines()


@pytest.mark.parametrize("text", EVERYDAY_LINES)
def test_everyday_lines_are_not_redacted(text):
    assert Anonymizer(FINAL_NAMES)(text) == text


@pytest.mark.parametrize(
    "text, kept",
    [
        ("Tonga ve ny colis? Lot 3 no nalefa", "<ADDRESS>"),  # leans private: Lot + number
        ("Ref: 123456789 ilay transaction", "<TEL>"),  # leans private: long number
        ("Sambatra ianao ry Mialy", None),
        ("Nirina sy Fara no tonga voalohany", None),
    ],
)
def test_lines_with_private_bits(text, kept):
    out = Anonymizer(FINAL_NAMES)(text)
    real = {compare_key(w) for name in FINAL_PEOPLE for w in name.split()}

    assert not real & {compare_key(w) for w in out.split()}
    assert kept is None or kept in out


def fake_first(person):
    return FINAL_NAMES[person].split()[0]


@pytest.mark.parametrize(
    "text, expected",
    [
        # owner's call (round 4): every capitalised word after "@" goes, even "Aiza"
        ("@Nirina Rakotomalala Aiza ianao?", FINAL_NAMES["Nirina Rakotomalala"] + " ianao?"),
        ("@Nirina Mbola tsy vita", fake_first("Nirina Rakotomalala") + " tsy vita"),
        ("@Nirina mbola tsy vita", fake_first("Nirina Rakotomalala") + " mbola tsy vita"),
        ("@Nirina Rakotomalala", FINAL_NAMES["Nirina Rakotomalala"]),
        ("@Fara Ok ve?", fake_first("Fara Randria") + " ve?"),
        ("@Fara Randria Tojo ve?", FINAL_NAMES["Fara Randria"] + " ve?"),
    ],
)
def test_mention_replaces_every_capitalised_word(text, expected):
    assert Anonymizer(FINAL_NAMES)(text) == expected


PRIV_NAMES = assign_fake_names(["Nirina Rakotomalala", "Fara Randria", "Jean-Pierre Rakoto"])


@pytest.mark.parametrize(
    "text, leaked, rest",
    [
        ("@Jean-Pierre Rakoto salut", {"jean", "pierre", "rakoto"}, " salut"),
        ("@Marie-Claire Rasoanaivo salut", {"marie", "claire", "rasoanaivo"}, " salut"),
        ("@Hery-Zo Andriamanana", {"hery", "zo", "andriamanana"}, ""),
        ("@haja rabe ok", {"haja", "rabe"}, " ok"),
    ],
)
def test_hyphenated_and_lowercase_mentions(text, leaked, rest):
    out = Anonymizer(PRIV_NAMES)(text)

    assert not leaked & {compare_key(p) for w in out.split() for p in w.split("-")}
    assert out.endswith(rest)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("(034) 12 345 67", "<TEL>"),
        ("(034)12 345 67", "<TEL>"),
        ("Antsoy aho (034 12 345 67)", "Antsoy aho (<TEL>)"),
        ("Tel(034 12 345 67)", "Tel(<TEL>)"),
        ("[034 12 345 67]", "[<TEL>]"),
        ("num:0341234567", "num:<TEL>"),
        ("laharana0341234567", "laharana<TEL>"),
        ("0341234567📞", "<TEL>📞"),
        ("034,12,345,67", "<TEL>"),
        ("034_12_345_67", "<TEL>"),
        ("+261 34 12 345 67", "<TEL>"),
    ],
)
def test_phones_in_brackets_and_glued(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


@pytest.mark.parametrize(
    "text, leaked",
    [
        ("Nirina-nao", "nirina"), ("Nirinantsika", "nirinantsika"), ("NIRINAKO", "nirinako"),
        ("ry Nirinaaa ô", "nirina"), ("Fara a!", "fara"), ("Faraaa", "fara"), ("Faralahy", "faralahy"),
    ],
)  # fmt: skip
def test_name_forms_do_not_leak(text, leaked):
    out = Anonymizer(PRIV_NAMES)(text)

    assert leaked not in {compare_key(p) for w in out.split() for p in w.split("-")}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Marie-Claire.Rasoanaivo@gmail.com", "<EMAIL>"),
        ("https://www.facebook.com/profile.php?id=100012345678", "<URL>"),
    ],
)
def test_email_and_profile_link(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


def test_hashed_fake_avoids_real_names():
    from fym1.data.anonymize import FIRST_NAMES

    names = assign_fake_names([f"{first} Rakotobe" for first in FIRST_NAMES[1:]])

    assert Anonymizer(names)("@Zafindraibe") == FIRST_NAMES[0]


@pytest.mark.parametrize("text", ["@Haja Rabe Aiza ianao?", "@Marie-Claire Rasoanaivo Aiza ianao?"])
def test_unknown_mention_becomes_one_fake_name(text):
    out = Anonymizer(FINAL_NAMES)(text)
    name = out.removesuffix(" ianao?")

    assert out.endswith(" ianao?")
    assert len(name.split()) == 2 and all(w[0].isupper() for w in name.split())
    assert not {"haja", "rabe", "aiza", "marie", "claire", "rasoanaivo"} & {compare_key(w) for w in name.split()}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("laharana034 12 345 67", "laharana<TEL>"),
        ("tel034.12.345.67", "tel<TEL>"),
        ("num034-12-345-67 io", "num<TEL> io"),
        ("Tel:020 22 345 67", "Tel:<TEL>"),
    ],
)
def test_glued_phones_with_gaps(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


@pytest.mark.parametrize("text", ["Taona 2020-03-12 io", "ny 2023 12 345 olona"])
def test_glued_phone_rule_skips_years(text):
    assert Anonymizer(PRIV_NAMES)(text) == text


# --- leak fix: mention echo, glued +261 phones --------------------------------------------------------

NOBODY = Anonymizer({})


@pytest.mark.parametrize("first", FIRST_NAMES)
def test_hashed_mention_fake_never_repeats_the_mentioned_first_name(first):
    out = NOBODY(f"@{first} tonga ve?")

    assert compare_key(out.split()[0]) != compare_key(first)
    assert out.endswith(" tonga ve?")


@pytest.mark.parametrize("first, last", [(f, s) for f in FIRST_NAMES for s in SURNAMES])
def test_hashed_mention_fake_never_repeats_mentioned_words(first, last):
    out = NOBODY(f"@{first} {last}")

    assert not {compare_key(first), compare_key(last)} & {compare_key(w) for w in out.split()}


@pytest.mark.parametrize("first", FIRST_NAMES)
def test_hashed_handle_fake_never_repeats_the_handle(first):
    out = NOBODY(f"@{first.lower()}.rk salama")

    assert compare_key(out.split()[0]) != compare_key(first)


def test_exact_mention_of_participant_first_name_with_unknown_surname():
    out = Anonymizer(FINAL_NAMES)("@Nirina Rabearisoa")

    assert out == fake_first("Nirina Rakotomalala")


@pytest.mark.parametrize(
    "text, expected",
    [
        ("WhatsApp+261341234567", "WhatsApp<TEL>"),
        ("wa+261341234567", "wa<TEL>"),
        ("laharana+261341234567", "laharana<TEL>"),
        ("laharana261341234567", "laharana<TEL>"),
        ("laharana261 34 12 345 67", "laharana<TEL>"),
        ("tel00261341234567", "tel<TEL>"),
        ("tel+2610341234567", "tel<TEL>"),
        ("tel2610341234567", "tel<TEL>"),
        ("tel(034) 12 345 67", "tel<TEL>"),
        ("tel+261 (0)34 123 45 67", "tel<TEL>"),
        ("Tel(+261)34 123 45 67", "Tel<TEL>"),
        ("tel(00261)34 123 45 67", "tel<TEL>"),
        ("numero261 034 123 45 67", "numero<TEL>"),
        ("ato261.034.123.45.67", "ato<TEL>"),
        ("tel+261 3 4 1 2 3 4 5 6 7", "tel<TEL>"),
        ("tel (089) 123 45 67", "tel <TEL>"),
        ("tel (069) 1234567", "tel <TEL>"),
        ("tel(044) 12 345 67", "tel<TEL>"),
    ],
)
def test_glued_international_phones(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "20:30", "Amin'ny 20:30 isika", "1 000 000 Ar", "261 300 000 Ar", "1 261 340 000 Ar", "solo 34 olona",
        "Manao 300 000 000 Ar ilay tany", "Mila vola ho 300 000 000 Ar", "valo 320 000 000",
        "Tonga tao 2 345 678 90", "ho 2025 12 345 olona", "Ar350 000 000", "MGA320 000 000", "Fmg 320 000 000",
        "Tamin'ny 7h35 1 500 000 Ar", "Nandoa 8h32 2 500 000 Ar", "tamin'ny 14h36 2 400 000 Ar",
        "ilay320 000 000", "io200000000",
        ") (010) 2171 2h02 1925 25/01/2019", "valo 20h02 592 10 212 590", "0h03 50,676,597", "(041) 10283 13h38",
    ],
)
def test_times_and_amounts_stay(text):
    assert Anonymizer(PRIV_NAMES)(text) == text


def test_mention_across_a_non_breaking_space():
    out = NOBODY("@Fara\u00a0Rabe salama")

    assert not {"fara", "rabe"} & {compare_key(w) for w in out.replace("\u00a0", " ").split()}
    assert out.endswith(" salama")


@pytest.mark.parametrize("prefix", ["its", "xx", "i_", "mr", "the", "real"])
@pytest.mark.parametrize("first", FIRST_NAMES)
def test_hashed_handle_fake_never_repeats_a_name_inside_the_handle(prefix, first):
    out = NOBODY(f"@{prefix}{first.lower()} salama")

    assert compare_key(out.split()[0]) != compare_key(first)


@pytest.mark.parametrize("gap", ["\u00a0", "\u202f", "\u2002", "\u2009", "\u3000", "\t"])
def test_mention_across_unicode_spaces(gap):
    out = NOBODY(f"@Fara{gap}Rabe salama")

    assert "Rabe" not in out
    assert out.endswith(" salama")


@pytest.mark.parametrize("brk", ["\n", "\r\n", "\u2028", "\u2029", "\x85"])
def test_mention_stops_at_a_line_break(brk):
    out = NOBODY(f"@Fara{brk}Rabe salama")

    assert out.endswith(f"{brk}Rabe salama")


@pytest.mark.parametrize("split", ["{a}_{b}", "{a}.{b}", "its{a}_{b}", "{a}2{b}", "{a}_{a2}"])
@pytest.mark.parametrize("first", FIRST_NAMES)
def test_hashed_handle_fake_never_rebuilds_a_split_name(split, first):
    key = first.lower()
    out = NOBODY("@" + split.format(a=key[:2], b=key[2:], a2=key[1:]) + " salama")  # {a}_{a2}: "lo_ova"

    assert compare_key(out.split()[0]) != compare_key(first)


# --- #16 review notes ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("+261 038 27 2000/5", "<TEL>"),
        ("Antsoy +261 038 27 2000/5 aho", "Antsoy <TEL> aho"),
        ("00261 038 27 2000/5", "<TEL>"),
        ("(+261) 38 27 2000/5", "<TEL>"),
        ("261 038 27 2000/5", "<TEL>"),
    ],
)
def test_country_code_phone_wins_over_date_filter(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


@pytest.mark.parametrize(
    "text", ["01/02/2026 03/04 no daty roa", "ny 03/10/2026 amin'ny 14h30", "Taona 2020-03-12 io", "2h03 50 olona"]
)
def test_dates_still_kept_after_country_code_rule(text):
    assert Anonymizer(PRIV_NAMES)(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Antsoy 038 27 2000/5 aho", "Antsoy <TEL> aho"),
        ("034 12 2019-5", "<TEL>"),
        ("0 34 12 2019/5", "<TEL>"),
        ("034/12/2019/5", "<TEL>"),
        ("(034) 12 2000/5", "<TEL>"),
        ("laharana038 27 2000/5", "laharana<TEL>"),
        ("34 12 2019/5", "<TEL>"),
    ],
)
def test_local_phone_with_a_year_like_group_is_a_phone(text, expected):
    assert Anonymizer(PRIV_NAMES)(text) == expected


@pytest.mark.parametrize("text", ["01/02/2026 03/04", "03/04/2026 12 30", "2020-03-12 034", "17/10/2026 1430"])
def test_full_dates_next_to_numbers_are_kept(text):
    assert Anonymizer(PRIV_NAMES)(text) == text
