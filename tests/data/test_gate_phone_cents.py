"""Security finding #19: cents must not hide a phone number from the gate. Synthetic numbers only."""

import pytest

from fym1.data.gate import has_long_number


@pytest.mark.parametrize(
    "text",
    [
        "034 12 345 67,00 Ar",  # Malagasy mobile, spaced, cents attached
        "0341234567,00 Ar",  # with trunk 0
        "341234567,50 Ar",  # without trunk 0
        "341 234 567,00 Ar",  # grouped in thousands but not ending 000
        "391 234 567.00 €",
        "3412345678,00 Ar",  # 10 digits starting 3[2-9]
        "261341234567,00 Ar",  # country code
        "+261 34 12 345 67,00 Ar",
        "341234567,5 euros",
    ],
)
def test_phone_with_cents_is_a_long_number(text):
    assert has_long_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "320 000 000,00 Ar",  # starts 3[2-9] but grouped thousands ending 000
        "350.000.000,50 Ariary",
        "1 250 000,50 €",  # 7 digits: unchanged
        "123 456 789,00 €",  # 9 digits not starting 3[2-9]: cents still mark a price
    ],
)
def test_real_prices_with_cents_still_pass(text):
    assert not has_long_number(text)
