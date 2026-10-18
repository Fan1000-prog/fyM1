from fym1.data.messenger import Message, Thread
from fym1.data.segment import to_conversations

MIN = 60_000


def thread(participants, lines):
    """lines: (sender, minute, text)"""
    return Thread(
        title="t",
        participants=participants,
        messages=[Message(sender=s, text=t, ts_ms=m * MIN) for s, m, t in lines],
    )


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


def test_to_conversations_merges_and_splits():
    chat = thread(
        ["Ana", "Bema"],
        [
            ("Ana", 0, "Salama"),
            ("Ana", 1, "De aona?"),
            ("Bema", 2, "Tsara"),
            ("Ana", 3, "Mety"),
            # gap over 60 min starts a new conversation
            ("Bema", 70, "Mbola eo ve?"),
            ("Ana", 71, "Eny"),
        ],
    )

    conversations = to_conversations(chat, owner="Ana")

    # 1:1 thread: each side once as assistant (participant order), then per session
    assert conversations == [
        [user("Tsara"), assistant("Mety")],
        [user("Mbola eo ve?"), assistant("Eny")],
        [user("Salama\nDe aona?"), assistant("Tsara")],
    ]


def test_to_conversations_gap_exactly_at_limit_does_not_split():
    chat = thread(["Ana", "Bema"], [("Ana", 0, "Salama"), ("Bema", 60, "Salama")])

    assert to_conversations(chat, owner="Ana") == [[user("Salama"), assistant("Salama")]]


def test_group_thread_owner_is_assistant():
    chat = thread(
        ["Ana", "Bema", "Koto"],
        [
            ("Bema", 0, "Iza no ho avy?"),
            ("Koto", 1, "Izaho"),
            ("Ana", 2, "Izaho koa"),
            ("Bema", 3, "Tsara"),
        ],
    )

    conversations = to_conversations(chat, owner="Ana")

    # other members merge into one nameless user turn; trailing user turn dropped
    assert conversations == [[user("Iza no ho avy?\nIzaho"), assistant("Izaho koa")]]


def test_to_conversations_caps_turns():
    lines = [("Ana" if i % 2 else "Bema", i, f"m{i}") for i in range(10)]
    chat = thread(["Ana", "Bema", "Koto"], lines)

    conversations = to_conversations(chat, owner="Ana", max_turns=4)

    assert [len(c) for c in conversations] == [4, 4, 2]
    assert all(c[0]["role"] == "user" and c[-1]["role"] == "assistant" for c in conversations)


def test_to_conversations_drops_one_sided_sessions():
    chat = thread(["Ana", "Bema"], [("Ana", 0, "Allo?"), ("Ana", 1, "Allo??")])

    assert to_conversations(chat, owner="Ana") == []


def test_every_sender_takes_a_turn_as_assistant_in_a_group():
    chat = thread(
        ["Ana", "Bema", "Koto"],
        [("Ana", 0, "Salama"), ("Bema", 1, "Eny"), ("Koto", 2, "Inona?"), ("Ana", 3, "Tsy misy")],
    )

    every = to_conversations(chat, owner=None)

    assert every == [c for name in chat.participants for c in to_conversations(chat, owner=name)]
    assert len(every) == 3


def test_every_sender_in_a_one_to_one_thread_is_each_side_once():
    chat = thread(["Ana", "Bema"], [("Ana", 0, "Salama"), ("Bema", 1, "Eny"), ("Ana", 2, "Veloma")])

    assert to_conversations(chat, owner=None) == to_conversations(chat, owner="Ana")
