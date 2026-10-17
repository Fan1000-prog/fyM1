"""Made-up text for the trimming tests. Never real chat text."""

# Synthetic everyday-Malagasy lines with French mixing (the text we keep).
KEEP_CORPUS = [
    "De aona ry zalahy?",
    "Mety be zany, lancer-na fotsiny 😂",
    "Mbola miasa ve ialahy?",
    "Ok fa aza adino ny rdv rahampitso",
    "Tsy maninona fa avy aty aho rehefa vita ny cours",
    "Merci be ry namana, tena tsara be ilay soirée",
    "Aiza ianao izao? Efa tonga any amin'ny gare aho",
    "Mandeha any an-tsena izahay rahampitso maraina",
]

# Lines the trimmed tokenizer never saw, including a character outside the keep corpus.
HELD_OUT = [
    "Manao ahoana ianao androany?",
    "Aza miandry ahy fa mbola misy réunion aho",
    "Tonga soa ry zalahy! 🎉",
    "Ô, tsara be ilay sary nalefanao",
]

# Extra languages so the trained vocabulary has many tokens the keep corpus never uses.
OTHER_LANGUAGES = [
    "Guten Morgen, wie geht es dir heute?",
    "Der schnelle braune Fuchs springt über den faulen Hund",
    "Dziękuję bardzo, do zobaczenia jutro",
    "Hyvää huomenta, mitä kuuluu tänään?",
    "Köszönöm szépen, viszontlátásra holnap",
    "Det regnar mycket i Stockholm idag",
]

TRAIN_TEXT = (KEEP_CORPUS + HELD_OUT + OTHER_LANGUAGES) * 20
