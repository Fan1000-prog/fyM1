# Anonymisation (scripts/build_chats.py)

Runs on the owner's laptop before anything leaves it. Code: `src/fym1/data/anonymize.py`.
Tests: `tests/data/test_anonymize.py`, `tests/data/test_anonymize_probes.py`, `tests/data/test_anonymize_dates.py`,
`tests/data/test_anonymize_marks.py`.

## What is replaced

| Found in a message | Becomes |
|---|---|
| Participant names, in full or any part, any case, accents or stretched letters ("HANTA", "helene", "Haaanta") | the person's fake name ("First Last"; a part -> fake first name or surname) |
| A part with a possessive ending ("Soako", "Fyntsika") | fake + the ending |
| A word sharing a 4+ letter prefix with a name part ("hant"), unless it is a common Malagasy word | fake |
| Nicknames set through Messenger ("set the nickname for … to …", "a défini le surnom de … sur …") | the person's fake first name |
| `@mention`: "@" plus every capitalised word after it (up to 3 words, hyphens allowed), or an `@handle` | one fake name: a participant's own fake, otherwise a made-up one that never equals or sits inside the mention, its letters joined up ("@mia_ly" never becomes "Mialy"). A capitalised word after the name goes too ("@Hanta Aiza ianao?" -> "Lova ianao?"); the owner chose that over leaking a surname |
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

## Strict gate: nobody reads the text

After anonymising, `build_chats.py` runs a strict automatic gate (`src/fym1/data/gate.py`,
tests `tests/data/test_gate.py`). It drops a whole conversation if any turn still holds:

| Reason | Caught | Kept |
|---|---|---|
| `long_number` | 7+ digits in one run across any non-letter characters (every kind of space, `. , / \ - _ ( ) + ; * \| :`, unicode dashes) (`12 34 567`, `(034) 12 345 67`, `15/03/2026`) | a price: clean 3-digit groups or no separator, then `Ar`, `Ariary`, `fmg`, `€`, `euro(s)`, `eur` (`1 500 000 Ar`), and not phone-shaped: no `+` before it, digits not starting `0` or `261`, not `3[2-9]` + 7 digits (`0341234567 Ar`, `341234567 Ar` are long numbers). A run of 9-10 digits is a phone's length (Malagasy landlines `2x`, French mobiles `[67]` without the 0, ...), so it is a price only with grouped thousands ending `000` (`150 000 000 Ar`) or cents (`123 456 789,00 €`); `202 212 345 Ar`, `612 345 678 €`, `150000000 Ar` are long numbers |
| `contact` | an email, a link with a path or query, an `@` before a word or a placeholder (`@njaka`, `@ njaka`, `njaka@ <URL>`), `at`/`arobase`/`(at)`/`[at]` before a placeholder or a mail host (`njaka at gmail`), a private-use mark from a failed replacement | `<EMAIL>`, `<URL>`, a bare `host.com`, `@ 3 ora` |
| `too_many_unknown` | more than 30% of the conversation's words would be replaced as names (below) | 30% or less |
| `hidden_char` | an invisible or combining character still there after normalising (below): a spacing mark (Mc) inside a name, a Hangul filler, a format character not right after a letter (`a \u200dHanta`), bidi marks | emoji variation selectors, ZWJ between emoji, keycap `U+20E3` after a digit, `#` or `*`, subdivision-flag tags after 🏴, accents NFC composes |

The gate fails closed: replacing or dropping too much is fine, a leak is not. The anonymiser and the
gate both NFKC-normalise text first, so full-width (`＠`, `Ｒ`) and circled (`Ⓡ`) characters are read as
plain ones (kept text comes out normalised: no-break spaces become plain spaces). Combining marks left
loose after that (strike-through, underline "fancy text") and zero-width spaces are dropped, and so are
format characters (ZWJ, ZWNJ, soft hyphen, U+2062), enclosing marks (U+20DD, U+20E3) and variation
selectors right after a letter; after an emoji, digit, `#` or `*` they stay, so emoji families, `❤️`
and keycaps `1️⃣` are kept. Accents NFKC composes ("é", "ô", "à", "ñ") stay; a mark with no precomposed
letter is dropped, so `n̈` (Northern Malagasy spelling) becomes `n` and `m̀` becomes `m`. Words are letter runs
split at digits, underscores, punctuation and a lower-to-upper case change, so "Rakoto2", "2Rakoto",
"jean_r" and "salutRakoto" are checked; only the exact placeholders (`<TEL>`, `<EMAIL>`, `<URL>`,
`<HANDLE>`, `<ADDRESS>`) are skipped. A hyphenated `Ra-be` is one word, and so is a run of 3+ single
letters joined by 1-3 non-letter, non-digit characters ("r a k o t o", "R.a.k.o.t.o", "R_a_k_o_t_o",
"J,E,A,N", "R . a . k . o . t . o", "R🔥a🔥k🔥o🔥t🔥o"): it is checked whole.

Every capitalised word (2+ letters) that is not a fake the anonymiser actually wrote (participants'
fakes and @mention fakes, reported by `Anonymizer.mention_fakes`) or a currency word is cleared or
replaced. Any other pool name ("Mialy" where nobody got that fake) is replaced like an unknown word.

- **Cleared at a sentence start** if a known-word list has it (`src/fym1/data/known_words.py`),
  checked in this order; the first list that has it gets the count (`cleared_<list>`):
  1. `mg_common`: the anonymiser's common Malagasy words that `mg_wiki` also has, plus short texting words.
  2. `mg_wiki` (`mg_wiki_words.txt`): words Malagasy Wikipedia types in lower case 3+ times.
  3. `mg_corpus` (`mg_corpus_words.txt`): the 50,000 words Malagasy web text (HPLT 2.0) types in
     lower case most often, 3+ times, and capitalises mid-sentence at most a quarter as often as in
     lower case (stricter than `mg_wiki`: web text types given names such as "riana" or "nambinina" in
     lower case too). It adds everyday and texting spellings Wikipedia lacks. Never built from our chats.
  4. `known_names` (`mg_known_names.txt`): a short hand list of places, brands, religious, family,
     calendar and title words and texting words that are not person names (Tana, Mvola, Jesoa, Neny,
     Dada, Zoky, Zoma, Alatsinainy, Desambra, Krismasy, Rtoa, Dr, Andao, ...).
  5. `fr`, `en` (`fr_words.txt`, `en_words.txt`): the 30,000 most frequent French and English words from `wordfreq`.
- **Cleared mid-sentence or after a person marker** (`i`, `ry`, `ra`, `n'i`, `an'i`, `amin'i`,
  `tamin'i`) only by the hand list `known_names` ("any Tana", "i Neny"). "tonga ny Fiara",
  "hoy Bonjour" are replaced.
- **`mg_corpus` alone never clears a name position** (lead decision, corpus-only name slots): after a
  person marker, after `@`, or capitalised mid-sentence, a word only `mg_corpus` knows is replaced
  unless another list (`mg_common`, `mg_wiki`, `fr`, `en`, `known_names`) has it too. Web text types
  given names in lower case, so its lower-case count is weak evidence exactly where names stand.
  Elsewhere (lower case in running text, a sentence start) `mg_corpus` alone still clears: asking a
  second list everywhere would bring back the over-masking `mg_corpus` was added to fix.
- **Never cleared**: a person name (`PERSON_NAMES`): `first_names.txt`, the fake-name pool, and
  `name_deny_list.txt`, common Malagasy given names that are also words (Hery, Soa, Vola, Mamy,
  Lalao, Fetra, Rivo, Tiana, Fara, Noro, Voahangy, ...). No corpus-wide evidence clears a word: a
  word typed in lower case elsewhere does not clear it.
- **Replaced** otherwise, by a fake first name from the anonymiser's pool: the same fake for the
  same word within one conversation, never a word already in that conversation. Counted as
  `tokens_replaced` (kept conversations only), with `fakes_per_kept_median` / `fakes_per_kept_max`.
  A conversation with more than 5 replaced words is dropped (`too_many_fakes`, lead decision
  AUTO-DECISIONS 22): trained on chats full of fakes, the model would sprinkle random names into
  replies. More than 30% replaced is dropped too (`too_many_unknown`). The fake counts also go to
  `results/data_stats.json`.

Also replaced, the same way:

- **A pool name that was already in the raw conversation** ("Rabe no nandeha", "RABE2" before
  anonymising): there it is a real person, not our fake, so it is replaced by a different fake, word
  or not. `build_records` finds these per conversation and passes them to the gate.
- **Lower-case words no list knows** (fail closed; name lists can never hold every name). Anywhere,
  a lower-case word that no known-word list knows is replaced like an unknown capitalised word:
  "salut tsanta", "rakoto", "salut sandrine", usernames such as "sandrinerakoto", and letters with
  digits inside ("h4nt4"). Person names are never known, so `mg_first_names.txt` ("mialy") and
  `first_names.txt` ("sandrine") names are always replaced. Kept: words a list knows ("trano", "lol",
  "mdr"), and names that are also everyday words (`name_deny_list.txt`, `wordlike_first_names.txt`,
  the fallback "olona": "mila vola", "soa ve", "une rose", "tia"). Lists miss some everyday words too
  (colloquial Malagasy, some French and English words Wikipedia mostly capitalises): those are
  replaced as well, which is over-masking, and can push a conversation over a drop line.
  After a person marker or an `@`, a person name or a word no list but `mg_corpus` knows ("ry hasina",
  "ry hery"; "ry namana", "i dada" stay). Any `ra-` hyphenated word, and a first name hyphen-joined to another
  ("jean-pierre", "marie-claire").

Words match folded (lower case, no accents). Stretched letters are collapsed only for real
stretching, a letter typed 3+ times ("Bonjouuur" -> "bonjour"), and never into a person name: double
letters are spelling ("Anne" does not match "âne"), and "Mariiie" does not match "mariée".

`scripts/build_known_words.py` (needs `uv sync --group data`) builds `first_names.txt`, `mg_wiki`,
`mg_corpus`, `fr` and `en` (`--corpus-only` rebuilds `mg_corpus` alone). It drops from the last four every person name (`first_names.txt`, the fake-name pool, the
deny list) and any word Wikipedia writes capitalised mid-sentence 3+ times and more than 3x as often as
in lower case in Malagasy Wikipedia (French and English Wikipedia capitalise everyday words in titles,
so they would drop "Bonjour"). The floor of 3 matches the lower-case floor for a word; it was 10, which
let rarer names through.
`wordfreq` and the common-word list ignore case, so without this "marie", "sandrine" or "kevin" would
clear a person's name.
`first_names.txt` is every French first name given 1,000+ times since 1900 (English ones
common in France included), minus first names Malagasy Wikipedia types in lower case 150+ times
(everyday words: "tia", "mila", "amin") and the hand list. The Wikipedia revision and the INSEE file's
hash are pinned in the script, and so is the HPLT revision. `mg_corpus` streams the two HPLT parquet
files one row group at a time (nothing saved) and cuts links, bare domains, emails and @handles before
counting, since web addresses hold names in lower case ("fb.com/rakoto.jean"). Every list file starts with a `#` source line.
It also builds `wordlike_first_names.txt`: INSEE first names that wordfreq's top 30k
French/English or Malagasy Wikipedia has, and that the same language's Wikipedia types in lower case
3+ times and does not mostly capitalise (the 3x name filter). French and English Wikipedia are read
as the first 100,000 articles of the pinned revision. So "rose", "claire", "aime", "tia" are words,
while "pierre", "max", "france", "dina" stay names (Wikipedia capitalises them far more often).

Counts only (never text or words) go to `data_gate.json` next to `--out`; `--gate-results
results/data_gate.json` copies them into the repo. `--dry-run` prints the same counts and writes nothing.
The 100-sample review sheet (`--sheet`) and the leftover-words list (`--leftover-sheet`) are optional and
off by default; `--blocklist` still replaces listed words everywhere.

### Word list sources and licences

- `mg_wiki_words.txt`, and the name filter: derived from Malagasy Wikipedia (`wikimedia/wikipedia`,
  `20231101.mg`), text under CC BY-SA 4.0, by Wikipedia contributors. Only a word list is kept.
- `mg_corpus_words.txt`: derived from HPLT 2.0 cleaned (`HPLT/HPLT2.0_cleaned`, `plt_Latn`), released
  under CC0 1.0 (web text crawled from Internet Archive and Common Crawl). Only a word list is kept.
- `first_names.txt`: derived from the INSEE "fichier des prénoms" (édition 2022, national file),
  Licence Ouverte / Etalab 2.0. Only folded names over the threshold are kept.
- `wordlike_first_names.txt`: INSEE names (Licence Ouverte) filtered with wordfreq and French, English
  and Malagasy Wikipedia case counts (CC BY-SA 4.0). Only a word list is kept.
- `fr_words.txt`, `en_words.txt`: derived from `wordfreq` 3.1 by Robyn Speer (code Apache-2.0, word
  frequency data CC BY-SA 4.0). `wordfreq` is a build-time dependency only (`data` group).

When unsure, the rules lean private: "Lot 3" becomes `<ADDRESS>`, a 9+ digit reference number `<TEL>`.

## Known limits — the strict gate is the backstop

- **Third-party names typed as plain text** (someone not in the chat, no `@`) are not detected
  automatically; that needs named-entity recognition. The strict gate replaces them, capitalised or
  lower-case, unless a known-word list has them: a name that is also an ordinary word can pass.
- **A name that is also a word** passes only at a sentence start, and only if it is in no name list:
  a Malagasy given name missing from `name_deny_list.txt`, a name that is also an everyday Malagasy
  word ("Tia", "Mila"), or an English-only name rare in France. Mid-sentence it is always replaced
  unless the hand list has it. In lower case and not after a marker, a name that is also a word
  stays ("mila vola", "soa ve", "une rose").
- Capitalised ordinary words in no list (rare words, typos, places), any capitalised word
  mid-sentence that the hand list lacks, and sentence-start words that are also names ("Faly be aho")
  are replaced by a fake name: over-masking, not a leak.
- A pool name that is also an everyday word ("Soa", "Vola", "Mamy") typed in lower case and not after
  a name particle ("hitako mamy") stays.
- Names spelled out in pieces of two or more letters: "Han-ta". (The gate reads 3+ single letters
  joined by 1-3 non-letter, non-digit characters as one word, so "H a n t a", "R.a.k.o.t.o" and
  "R_a_k_o_t_o" are replaced; it
  drops loose strike-through/underline marks, so "R̶a̶k̶o̶t̶o̶" is too. "H4nt4", "Rak0to" are replaced:
  letters with digits inside are always a name.)
- Names spelled in regional-indicator letters ("🇷🇦🇰🇴🇹🇴"): NFKC does not fold them to letters, so
  the gate sees emoji, not a word.
- A phone split over two turns ("nomerao 034 12", then "345 67 io"): the gate checks one turn at a time.
- A phone split by a word inside one turn ("034 12 dia 345 67"): a word ends the digit run.
- Digits joined by any non-letter characters count as one number, so a list of times or numbered
  lines ("14:30-15:30") can drop a conversation: over-dropping, not a leak.
- Phone numbers written in words, Malagasy or French ("aotra telo efatra…", "034 douze …"), or with
  letters for digits ("l2").
- A phone whose first digits make a full date and whose rest reads as a price ("18/03/80 559 874",
  "18/03/80 559000"): kept as a date and a price.
- Phone numbers glued to a word without the leading 0 ("tel34 12 345 67"): a word's own last letters
  can't be told apart from a number's start. With a letter O glued to a word, an amount-shaped number
  ("contactO320 000 000") is kept as an amount.
- Addresses other than "Lot …" ("Cité 67Ha porte 12", "Villa 23 Ivandry"), and place names.
- Nicknames used in chat but never set through Messenger.

From the gate-hardening PR on, reviewers judge the gate against real-world family-chat risk. An evasion
that needs someone to type a name or number in a deliberately obscure way, and that the anonymiser
cannot meet in a real export, goes in this list; it is not a merge blocker.

These are pinned as `xfail` tests in `tests/data/test_anonymize_probes.py`.

## Public release snapshot

The public repo is one fresh commit built by a private release tool (it does not ship). It reads the export's
sender names at run time, removes them from the three hand lists, and fails if any shipped file or path still
holds one, a private id or a secret. Words are read as the strict gate reads them (`fym1.data.gate.tokens`: case
changes, spaced-out letters), after decoding `\x`/`\u`/`\U`, HTML and percent escapes, with hidden characters
taken both as a break and as nothing. Only known text types are read, and a PDF with its metadata; any other file
fails. The public tree holds no archives, so any archive (zip, gz, tar, tgz, 7z, rar, whl, jar, bz2, xz and the
like) is refused before anything is written, found by its name or by its bytes: a file that starts like an
archive or a compressed stream, a tar, or a file with a zip record anywhere (a zip glued to a PDF or a text file).
A name of 4+ letters is a hit anywhere inside a word that no known-word list has, unless the release
owner clears that exact word in that exact file by hand, with a reason; a path is never cleared. Nothing is
written to the public folder unless every check passes.

The lists `scripts/build_known_words.py` writes do not ship: it builds them minus the hand lists' names, so
rebuilding them from the shipped (stripped) hand lists and diffing would show the removed names. Build them
before using the gate: `uv sync --group data && uv run python scripts/build_known_words.py` (downloads Wikipedia,
INSEE and HPLT). Until then the snapshot's tests that need them are skipped.

Known limits of the release scan:

- A name typed under 3 letters is not searched (counted as skipped in the report).
- A name inside a known word ("certainly" for a name "Ertain") passes.
- In a PDF only the text layer, the document info and the XMP metadata are read. Annotations, bookmarks, form
  fields and objects no page uses are not read, and neither is text drawn as an image.
- The shipped hand lists are shorter by the entries removed; a doc that quotes a list's size would show how
  many went, not which.

## Over-masking we accept

Losing a number is cheap; leaking one is not. A mobile without the 0 is masked in any grouping
(`3[2-9]` + 7 digits), unless it reads as a price in clean 3-digit groups ("35 000, 3 500", "32 500 000 Ar",
"35.000 2015"). So a count before a price ("34 2 500 000") or two short numbers after a date
("15/03/2026 17 52 485 kg") can become `<TEL>`.
