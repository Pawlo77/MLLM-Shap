"""Code to generate a word cloud image from a text file using a question mark mask."""

import os

import numpy as np
from PIL import Image
from wordcloud import WordCloud

ROOT_DIR: str = os.path.dirname(__file__)
FIGURES_DIR: str = os.path.join(ROOT_DIR, "figures")

def main() -> None:
    """Generate a word cloud image from a text file using a question mark mask."""
    text = open(os.path.join(ROOT_DIR, "words.txt")).read()
    mask = np.array(Image.open(os.path.join(FIGURES_DIR, "question_mark.png")))

    wc = WordCloud(
        background_color="white",
        max_words=4000,
        mask=mask,
        scale=4,
        margin=1,
        min_font_size=1,
        random_state=42,
        collocations=False,
    )

    wc.generate(text)
    wc.to_file(os.path.join(FIGURES_DIR, "cloud.png"))

if __name__ == "__main__":
    main()
