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
        "202212345,00 Ar",  # 9 digits starting 2: cents never make a price
        "612 345 678,00 €",  # French mobile without its 0, grouped
        "712 345 678,50 euros",
        "123 456 789,00 €",  # 9 digits not ending 000: cents never make a price
        "6123456,78 €",  # 7 digits + cents = 9 digits in the run
        "61234567,8 €",  # 8 digits + cents = 9
        "320 000 000,00 Ar",  # 3[2-9] mobile shape once the cents are removed, even ending 000
        "350.000.000,50 Ariary",
        "341 234 000,00 Ar",
        "341 200 000,5 €",
        "341.234.000,00 Ar",
    ],
)
def test_phone_with_cents_is_a_long_number(text):
    assert has_long_number(text)


@pytest.mark.parametrize(
    "text",
    [
        "150 000 000,00 Ar",  # 9 digits grouped ending 000, not a phone shape
        "1 250 000,50 €",  # 7 digits: unchanged
        "12 345 000,50 €",  # 10 digits with cents, grouped ending 000
    ],
)
def test_real_prices_with_cents_still_pass(text):
    assert not has_long_number(text)
