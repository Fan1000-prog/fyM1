from fym1.data.review_sheet import render_review_sheet, sample_for_review


def record(i, content="Salama"):
    return {
        "id": f"id{i}",
        "messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "Eny"}],
    }


def test_review_sheet_has_one_checkbox_per_conversation():
    html = render_review_sheet([record(1), record(2)])

    assert html.count('type="checkbox"') == 2
    assert 'value="id1"' in html and 'value="id2"' in html


def test_review_sheet_escapes_text():
    html = render_review_sheet([record(1, "Antsoy <TEL> <script>x</script>")])

    assert "&lt;TEL&gt;" in html
    assert "<script>x</script>" not in html


def test_review_sheet_exports_approved_ids():
    html = render_review_sheet([record(1)])

    assert "approved_ids.txt" in html


def test_sample_for_review_is_seeded_and_capped():
    records = [record(i) for i in range(300)]

    sample = sample_for_review(records, n=100, seed=0)

    assert len(sample) == 100
    assert sample == sample_for_review(records, n=100, seed=0)
    assert len(sample_for_review(records[:5], n=100, seed=0)) == 5
