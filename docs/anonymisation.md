# Anonymisation (scripts/build_chats.py)

Runs on the owner's laptop before anything leaves it. Code: `src/fym1/data/anonymize.py`.
Tests: `tests/data/test_anonymize.py`, `tests/data/test_anonymize_probes.py`.

## What is replaced

| Found in a message | Becomes |
|---|---|
| Participant names, in full or any part, any case, accents or stretched letters ("NIRINA", "helene", "Niriiina") | the person's fake name ("First Last"; a part -> fake first name or surname) |
| A part with a possessive ending ("Soako", "Fyntsika") | fake + the ending |
| A word sharing a 4+ letter prefix with a name part ("nirin"), unless it is a common Malagasy word | fake |
| Nicknames set through Messenger ("set the nickname for … to …", "a défini le surnom de … sur …") | the person's fake first name |
| `@mention`: "@" plus every capitalised word after it (up to 3 words, hyphens allowed), or an `@handle` | one fake name: a participant's own fake, otherwise a made-up one that never equals or sits inside the mention, its letters joined up ("@mia_ly" never becomes "Mialy"). A capitalised word after the name goes too ("@Nirina Aiza ianao?" -> "Lova ianao?"); the owner chose that over leaking a surname |
| Labelled handles: `ig:`, `insta`, `snap:`, `tiktok`, `tg`/`telegram`, `twitter`, `fb`, `viber`, `wa`/`whatsapp` (not numbers or everyday words: "fb 2 andro") | `<HANDLE>` |
| Phone numbers: Malagasy (`034 12 345 67` in any grouping, `(034) 12 …`, `020 …`, `+261 (0)34 …`, `34 12 345 67`, glued to a word in the same forms: `laharana034 12 345 67`, `wa+261341234567`, `tel(034) …`, `tel34 12 345 67`), other international; not amounts or dates | `<TEL>` |
| Emails, including `x [at] y . com` | `<EMAIL>` |
| Links, with or without `http`, any `host.tld/path`; a bare lower-case `host.com` | `<URL>` |
| Plot addresses `Lot II M 85 Ter` | `<ADDRESS>` |

Common Malagasy words (top 3,000 from Malagasy Wikipedia, `src/fym1/data/mg_common_words.txt`)
are never wiped by a prefix match. If a contact's name *is* a common word ("Vola"), that word is
replaced everywhere; `build_chats.py` reports how often (`common_word_replacements`), and how many
nickname events it could not read (`unharvested_nickname_events`).

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
- Addresses other than "Lot …" ("Cité 67Ha porte 12", "Villa 23 Ivandry"), and place names.
- Nicknames used in chat but never set through Messenger.

These are pinned as `xfail` tests in `tests/data/test_anonymize_probes.py`.
