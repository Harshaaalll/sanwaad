"""Reddit connector (asyncpraw).

Read access needs only a script-type app; posting needs a logged-in account.
`SANWAAD_ALLOW_POSTING` gates writes and defaults to off, so a misconfigured
run can embarrass you in a log file but not on somebody's actual thread.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from loguru import logger

from ..models import AuthorMeta, Channel, Complaint
from .base import Connector


class RedditConnector(Connector):
    name = "reddit"

    def __init__(self, subreddits: str | None = None, query: str | None = None):
        self.subreddits = subreddits or os.getenv("REDDIT_SUBREDDITS", "india+IndiaInvestments")
        self.query = query or os.getenv("REDDIT_QUERY", "upi OR wallet OR refund")
        self._client = None

    def _reddit(self):
        if self._client is None:
            import asyncpraw

            self._client = asyncpraw.Reddit(
                client_id=os.environ["REDDIT_CLIENT_ID"],
                client_secret=os.environ["REDDIT_CLIENT_SECRET"],
                user_agent=os.getenv("REDDIT_USER_AGENT", "sanwaad/0.1 by u/sanwaad_bot"),
                username=os.getenv("REDDIT_USERNAME"),
                password=os.getenv("REDDIT_PASSWORD"),
            )
        return self._client

    async def fetch(self, limit: int = 20) -> list[Complaint]:
        reddit = self._reddit()
        subreddit = await reddit.subreddit(self.subreddits)
        out: list[Complaint] = []
        async for post in subreddit.search(self.query, sort="new", limit=limit):
            body = (post.selftext or "").strip()
            out.append(Complaint(
                external_id=post.id,
                channel=Channel.REDDIT,
                author=str(post.author) if post.author else "[deleted]",
                text=f"{post.title}\n\n{body}".strip(),
                url=f"https://reddit.com{post.permalink}",
                created_at=datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat(),
                author_meta=await self._author_meta(post),
            ))
        return out

    @staticmethod
    async def _author_meta(post) -> AuthorMeta:
        """What Reddit will tell us about the poster.

        Best-effort by design. Fetching an author costs a request and can fail
        on deleted or suspended accounts, and the judge treats every missing
        field as carrying no weight rather than as a negative — so losing this
        degrades the verdict's confidence, never its safety.
        """
        author = getattr(post, "author", None)
        if author is None:
            return AuthorMeta()
        try:
            await author.load()
            created = getattr(author, "created_utc", None)
            age_days = None
            if created:
                age_days = int(
                    (datetime.now(timezone.utc)
                     - datetime.fromtimestamp(created, timezone.utc)).days
                )
            return AuthorMeta(
                account_age_days=age_days,
                karma=(getattr(author, "link_karma", 0) or 0)
                + (getattr(author, "comment_karma", 0) or 0),
                verified=bool(getattr(author, "verified", False)),
            )
        except Exception as exc:
            logger.debug(f"[reddit] could not load author for {post.id}: {exc}")
            return AuthorMeta()

    async def reply(self, external_id: str, text: str) -> dict:
        if os.getenv("SANWAAD_ALLOW_POSTING", "").lower() not in ("1", "true", "yes"):
            logger.warning(f"[reddit] posting disabled; would have replied to {external_id}")
            return {"reply_id": None, "url": None, "dry_run": True,
                    "posted_at": datetime.now(timezone.utc).isoformat()}
        reddit = self._reddit()
        submission = await reddit.submission(id=external_id)
        comment = await submission.reply(text)
        return {"reply_id": comment.id,
                "url": f"https://reddit.com{comment.permalink}",
                "dry_run": False,
                "posted_at": datetime.now(timezone.utc).isoformat()}

    async def close(self):
        if self._client is not None:
            await self._client.close()
