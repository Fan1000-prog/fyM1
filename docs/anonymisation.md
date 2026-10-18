# Anonymisation (scripts/build_chats.py)

Runs on the owner's laptop before anything leaves it. Code: `src/fym1/data/anonymize.py`.
Tests: `tests/data/test_anonymize.py`, `tests/data/test_anonymize_probes.py`, `tests/data/test_anonymize_dates.py`,
`tests/data/test_anonymize_marks.py`.

## What is replaced

| Found in a message | Becomes |
|---|---|
| Participant names, in full or any part, any case, accents or stretched letters ("NIRINA", "helene", "Niriiina") | the person's fake name ("First Last"; a part -> fake first name or surname) |
| A part with a possessive ending ("Soako", "Fyntsika") | fake + the ending |
| A word sharing a 4+ letter prefix with a name part ("nirin"), unless it is a common Malagasy word | fake |
| Nicknames set through Messenger ("set the nickname for … to …", "a défini le surnom de … sur …") | the person's fake first name |
| `@mention`: "@" plus every capitalised word after it (up to 3 words, hyphens allowed), or an `@handle` | one fake name: a participant's own fake, otherwise a made-up one that never equals or sits inside the mention, its letters joined up ("@mia_ly" never becomes "Mialy"). A capitalised word after the name goes too ("@Nirina Aiza ianao?" -> "Lova ianao?"); the owner chose that over leaking a surname |
| Labelled handles: `ig:`, `insta`, `snap:`, `tiktok`, `tg`/`telegram`, `twitter`, `fb`, `viber`, `wa`/`whatsapp` (not numbers or everyday words: "fb 2 andro") | `<HANDLE>` |
| Phone numbers: Malagasy (`034 12 345 67` in any grouping, `(034) 12 …`, `020 …`, `+261 (0)34 …`, without the 0 in any grouping: `34 12 345 67`, `34 123 45 67`, `34 1234567`, `341234567`, glued to a word: `laharana034 12 345 67`, `wa+261341234567`, `wa00261…`, `tel(034) …`, `tel+261 (0)34 …`, or with a letter O for the 0: `WhatsAppO328585520`, `numéroo32.78.833.34`), other international, any of these with unicode dashes (`034–12–345–67`) or no-break spaces; not amounts or dates | `<TEL>` |
| Emails, including `x [at] y . com` | `<EMAIL>` |
| Links, with or without `http`, any `host.tld/path`; a bare lower-case `host.com` | `<URL>` |
| Plot addresses `Lot II M 85 Ter` | `<ADDRESS>` |

Common Malagasy words (top 3,000 from Malagasy Wikipedia, `src/fym1/data/mg_common_words.txt`)
are never wiped by a prefix match. If a contact's name *is* a common word ("Vola"), that word is
replaced everywhere; `build_chats.py` reports how often (`common_word_replacements`), and how many
nickname events it could not read (`unharvested_nickname_events`).

## Dates next to numbers

Real dates and clock times (`14:30`, `14h30`) are found in the whole message before phones are: `dd/mm/yyyy`, `dd-mm-yyyy`, `dd.mm.yyyy`,
`dd/mm/yy`, `dd/mm`, `yyyy-mm-dd` (day 1-31, month 1-12), and ranges of them (`15/03-20/03/2026`). A phone
is never read out of a date, so "15/03/2026 1 500 Ar", "le 15/03/2026 14h30" and "15/03/2026 14 30" stay
as written, and "15/03/2026 034 12 345 67" keeps the date and masks the phone. A date glued to more
digits ("34/12/2019/5", "038 27 2000/5") is not a date. Phones fail closed around dates: every phone
shape that touches a date or time is masked unless all its digits are in the date or time, plus clear
numbers next to it: a time typed with a space (`14 30`), a price in clean 3-digit groups (`1 500`,
`25.000`), a year, a round number (`2500000`), or one short number (`150`). Several may follow each other
("2026 150 olona"), but only one short number, not next to a price, and no other number with a time. So
"12/03 41 23 45 67" (a phone whose `03` sits in a date) -> "12/<TEL>", and "15/03/2026 1 500 34 12" is
masked: digits that run past a clean price are not part of it. A phone that starts with `+`, `00`, `261`
or `02x`/`03x` (also `03-24-…`) outside a date still wins when a date shape sits inside it ("034 01/02/26 7",
"03-24-21-05-98"), even with a stray extra digit ("034 12 01/02/26", "261 32 01/11/12 12"). A date or time
right before a phone is never read as an amount, so "15/03/2026 0324-21-05-98" masks the whole phone. A
number before a phone counts as an amount only if the digits after it are clean 3-digit groups with the
amount's own separator, up to its end ("1 261 340 000 19/04" stays; "Ar 25 000 000 261-341-26-06-08" is masked). Digit groups glued on by a mark
after a phone go with it, and a number right before a phone may be masked with it ("12 00 037-44-352-52"
-> "12 <TEL>"): masking a bit too much beats leaving part of a number.

| Rule | Kept | Masked |
|---|---|---|
| A date or time is never part of a phone | `15/03/2026 1 500 Ar`, `15/03/2026 14 30` | `15/03/2026 034 12 345 67` -> `15/03/2026 <TEL>` |
| A strong start (`+`, `00`, `261`, `03x`, `03-x`) wins over a date shape inside | `03 45 15/03/2026` | `034 12 01/02/26` -> `<TEL>` |
| Any other phone shape touching a date is masked unless only dates and clear numbers | `15/03/2026 14 30`, `15/03/2026 2026 150 olona` | `12/03 41 23 45 67` -> `12/<TEL>`, `15/03/2026 1 500 34 12` -> `15/<TEL>` |
| A whole phone stops before a date after it | `034 12 345 67 15/03/2026` -> `<TEL> 15/03/2026` (date kept) | the phone |
| An amount before a phone: only clean 3-digit groups, same separator, nothing but dates after it | `1 261 340 000 19/04`, `1 020 000 15/03/2026` | `olona 3 261 382-08-11-65` -> `olona 3 <TEL>`, `5 034 01/02/26 7` -> `5 <TEL>` |
| Glued to a word, a pair-style `03` may stand apart (two letters before) | `2h03 50 12 34 56` | `laharana03 41 23 45 67` -> `laharana<TEL>` |
| Digits glued on by a mark after a phone go with it | | `05 58 0326-59-58-41` -> `<TEL>` |

Checked by a generated corpus of 5,000 everyday lines (dates, prices, times, quantities): none gets a
`<TEL>` (before this rule, 1,568 did). A generated corpus of 2,000 phones in every Malagasy layout (pairs,
3-2-3-2, `+261`/`261`/`00261`, separators space . - / or none) after a date, time, price or word: none
leaks a digit. Two non-date numbers side by side ("08 18 150 000") are still
one number to the phone rule; that is not a date problem.

## Unicode dashes, no-break spaces, a letter O glued to a word

Phones and dates are matched on a same-length copy where `‐ ‑ ‒ – — ― −` read as `-` and no-break spaces
(`\u00a0`, `\u202f`) as a space; the masks are then applied to the original text. So `034–12–345–67` and
`0327—35—17—59` are masked, `15–03–2026` stays a date, and `2\u202f000\u202f000\u202f000 Ar` stays an amount.

A letter O for the 0 right after a word (`contactO345 40 77 66`, `numO3 74 78 29 68`) is masked only before
`3` + `2-9` or `2` + `0`, with exactly 9 digits after it, and not when those digits read as an amount
(`manao320 000 000`, `io200000000`). It never wins over a date the way a real `0` can. Checked by a generated
corpus of 20,000 everyday lines with a word ending in o/O right before a number ("Hono 3 kilao", "photo 2",
"Toamasina 25", glued amounts): none gets a `<TEL>`.

## Leftover names: the owner's second pass

After anonymising, `build_chats.py` writes `leftover_names.html` next to the review sheet: every
capitalised word that survived and is not a common Malagasy word, with how often it appears —
words and counts only, no sentences. The owner ticks the real names, saves `blocklist.txt`, and
runs `build_chats.py` again with `--blocklist blocklist.txt`; those words are then treated as names
everywhere (same rules: case, accents, endings, stretched letters).

When unsure, the rules lean private: "Lot 3" becomes `<ADDRESS>`, a 9+ digit reference number `<TEL>`.

## Known limits — the owner's review sheet is the backstop

- **Third-party names typed as plain text** (someone not in the chat, no `@`) are not detected
  automatically; that needs named-entity recognition. Capitalised ones show up on the leftover
  sheet for the owner to block; lower-case ones rely on the review sheet.
- Names spelled out or disguised: "Ni-ri-na", "N i r i n a", "N1r1na".
- Phone numbers written in words ("aotra telo efatra…", "034 douze …") or with letters for digits ("l2").
- A phone whose first digits make a full date and whose rest reads as a price ("18/03/80 559 874",
  "18/03/80 559000"): kept as a date and a price.
- Phone numbers glued to a word without the leading 0 ("tel34 12 345 67"): a word's own last letters
  can't be told apart from a number's start. With a letter O glued to a word, an amount-shaped number
  ("contactO320 000 000") is kept as an amount.
- Addresses other than "Lot …" ("Cité 67Ha porte 12", "Villa 23 Ivandry"), and place names.
- Nicknames used in chat but never set through Messenger.

These are pinned as `xfail` tests in `tests/data/test_anonymize_probes.py`.

## Over-masking we accept

Losing a number is cheap; leaking one is not. A mobile without the 0 is masked in any grouping
(`3[2-9]` + 7 digits), unless it reads as a price in clean 3-digit groups ("35 000, 3 500", "32 500 000 Ar",
"35.000 2015"). So a count before a price ("34 2 500 000") or two short numbers after a date
("15/03/2026 17 52 485 kg") can become `<TEL>`.
