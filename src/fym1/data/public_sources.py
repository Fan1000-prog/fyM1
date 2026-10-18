"""Pinned public text sources, shared by the word-list builder and the vocabulary trim.

Each pin is a dataset revision on the Hugging Face Hub, so a rebuild reads exactly the same text.
"""

WIKIPEDIA_DATASET = "wikimedia/wikipedia"
WIKIPEDIA_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"  # 2024-01-09; configs 20231101.<lang>, CC BY-SA 4.0
HPLT_DATASET = "HPLT/HPLT2.0_cleaned"  # Malagasy web text, CC0 1.0
HPLT_REVISION = "d1324a5283f762ee62c2a5c81de08fc6450ea540"
HPLT_FILES = ("plt_Latn/train-00000-of-00002.parquet", "plt_Latn/train-00001-of-00002.parquet")
