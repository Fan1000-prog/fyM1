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
| `@mention` of a display name (up to 3 capitalised words) or an `@handle` | a fake name; a participant's mention gets their own fake |
| Labelled handles: `ig:`, `insta`, `snap:`, `tiktok`, `tg`/`telegram`, `twitter`, `fb`, `viber`, `wa`/`whatsapp` | `<HANDLE>` |
| Phone numbers: Malagasy (`034 12 345 67` in any grouping, `020 …`, `+261 (0)34 …`, `34 12 345 67`), other international | `<TEL>` |
| Emails, including `x [at] y . com` | `<EMAIL>` |
| Links, with or without `http`, any `host.tld/path` | `<URL>` |
| Plot addresses `Lot II M 85 Ter` | `<ADDRESS>` |

Common Malagasy words (top 3,000 from Malagasy Wikipedia, `src/fym1/data/mg_common_words.txt`)
are never wiped by a prefix match. If a contact's name *is* a common word ("Vola"), that word is
replaced everywhere; `build_chats.py` reports how often (`common_word_replacements`), and how many
nickname events it could not read (`unharvested_nickname_events`).

## Known limits — the owner's review sheet is the backstop

- **Third-party names typed as plain text** (someone not in the chat, no `@`) are not detected.
  Catching them needs named-entity recognition, which v0.1 does not have.
- Names spelled out or disguised: "Ni-ri-na", "N i r i n a", "N1r1na".
- Phone numbers written in words ("aotra telo efatra…", "034 douze …") or with letters for digits ("l2").
- Addresses other than "Lot …" ("Cité 67Ha porte 12", "Villa 23 Ivandry"), and place names.
- Nicknames used in chat but never set through Messenger.

These are pinned as `xfail` tests in `tests/data/test_anonymize_probes.py`.
