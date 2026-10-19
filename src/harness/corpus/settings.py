"""Corpus build parameters (see plan: 20 fields x 6 theses, 5 exemplars + 1 holdout per field)."""
import re

FIELDS_TARGET = 20
PER_FIELD = 6
MIN_YEAR = 2015
PAGES = (60, 450)
MIN_CHARS_PER_PAGE = 1200
CANDIDATES_MAX = 40
PASS_TARGET = 8
REQUIRED_TYPES = ["introduction", "literature_review", "methodology", "findings", "discussion", "conclusion"]
MIN_TYPES_FOUND = 3  # topic-titled body chapters all count as "findings", so 3 real types is a full thesis
MIN_WORDS_PER_TYPE = 800
USER_AGENT = "AcdyON-Scholar-corpus/1.0 (mailto:support@acdyon.io)"

DISSERTATION_FILTER = f"type:dissertation,open_access.is_oa:true,from_publication_date:{MIN_YEAR}-01-01,language:en"
WORK_SELECT = ("id,doi,title,publication_year,authorships,open_access,best_oa_location,locations,cited_by_count,"
               "primary_topic,is_retracted,abstract_inverted_index")


def slugify(text: str, max_len: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")[:max_len].strip("-") or "x"
