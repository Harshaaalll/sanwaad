"""Connector contract.

A connector is deliberately thin: fetch normalised complaints, post a reply.
Everything else — triage, policy, escalation — is channel-agnostic and lives
in the graph. Adding Instagram later should mean writing one class, not
touching the state machine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Complaint


class Connector(ABC):
    name: str

    @abstractmethod
    async def fetch(self, limit: int = 20) -> list[Complaint]:
        """Pull recent inbound items, newest first."""

    @abstractmethod
    async def reply(self, external_id: str, text: str) -> dict:
        """Post a public reply. Returns a receipt dict (id, url, posted_at)."""
