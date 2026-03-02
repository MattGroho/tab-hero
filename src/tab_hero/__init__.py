from importlib.metadata import metadata, version

_metadata = metadata("tab-hero")

__version__ = version("tab-hero")

# Author-email field is comma-separated, split into individual entries
_author_email = _metadata.get("Author-email") or ""
__authors__ = [author.strip() for author in _author_email.split(",") if author.strip()]

