from collections import Counter

from fym1.data.leftovers import leftover_names, read_blocklist, render_leftover_sheet


def test_leftover_names_counts_surviving_capitalised_tokens():
    texts = ["salama Koto, aiza i Bema?", "Koto ve? <TEL>", "Mbola tsy vita", "Soa sy Bema"]

    assert leftover_names(texts, fake_names=["Soa Rabe"]) == Counter({"Koto": 2, "Bema": 2})


def test_leftover_names_skips_common_words_placeholders_and_short_tokens():
    texts = ["Ny trano <EMAIL> <URL> <ADDRESS> <HANDLE> <TEL> A Mbola Tsara RN7"]

    assert leftover_names(texts, fake_names=[]) == Counter()


def test_leftover_sheet_lists_tokens_and_counts_only():
    html = render_leftover_sheet(Counter({"Koto": 3, "<b>Bad</b>": 1}))

    assert html.count('type="checkbox"') == 2
    assert 'value="Koto"' in html and ">3<" in html
    assert "<b>Bad</b>" not in html and "&lt;b&gt;Bad&lt;/b&gt;" in html
    assert "blocklist.txt" in html


def test_leftover_sheet_most_common_first():
    html = render_leftover_sheet(Counter({"Rare": 1, "Koto": 5}))

    assert html.index("Koto") < html.index("Rare")


def test_read_blocklist_ignores_blank_lines(tmp_path):
    path = tmp_path / "blocklist.txt"
    path.write_text("Koto\n\n  Bema  \n")

    assert read_blocklist(path) == ["Koto", "Bema"]
