from functools import lru_cache

from .base import Connector
from .mock_feed import MockFeedConnector


@lru_cache(maxsize=8)
def get_connector(name: str) -> Connector:
    if name in ("mock", "playstore"):
        return MockFeedConnector()
    if name == "reddit":
        from .reddit import RedditConnector

        return RedditConnector()
    raise ValueError(f"Unknown connector: {name}")


__all__ = ["Connector", "MockFeedConnector", "get_connector"]
