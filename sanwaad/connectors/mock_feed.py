"""A seeded feed of realistic complaints.

Written to exercise the paths that matter rather than to look tidy: Devanagari
and Latin-script Hinglish, a regulatory mention that must bypass auto-reply, a
sub-₹100 grumble that should sail through unattended, a fraud report, and a
compliment that must cost one triage call and stop.

Three later additions exist for the agents added after the first cut, and each
is here because it is a case the earlier pipeline handled *wrongly*:

- a fresh throwaway account posting brand-level outrage with nothing concrete
  in it, which the old pipeline would have drafted a careful grounded reply to;
- a large account making the same accusation, which must still be answered —
  reach outranks suspicion when the audience is real;
- six different people reporting one payment outage inside twelve minutes,
  which the old pipeline saw as six unrelated tickets because nothing was
  looking across cases.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from ..config import DATA_DIR
from ..models import AuthorMeta, Channel, Complaint
from .base import Connector

# (author, text, minutes_ago, author_meta)
_SEED = [
    ("u/rohit_mhrs",
     "NimbusPay se ₹4,500 transfer kiya tha 6 din pehle, paise beneficiary ko "
     "nahi mile aur mere account me bhi wapas nahi aaye. Support pe 3 baar likha, "
     "koi jawab nahi. Ye paisa mera hai ya aapka?",
     140, {"account_age_days": 1180, "karma": 3400, "post_count": 210}),

    ("u/anita_k92",
     "मेरा वॉलेट अचानक फ्रीज कर दिया गया, ₹18,000 अंदर पड़े हैं और ऐप कुछ बताता ही नहीं। "
     "कल मेरी बेटी की फीस भरनी है। कोई इंसान जवाब दे सकता है क्या?",
     128, {"account_age_days": 640, "karma": 190, "post_count": 44}),

    ("u/deep_ranjan",
     "Charged ₹15 on a ₹2,000 wallet-to-bank transfer even though I'm well under "
     "the 10k free limit this month. Small amount but it's the principle. What is "
     "this charge for?",
     115, {"account_age_days": 2200, "karma": 8800, "post_count": 900}),

    ("u/sameer.p",
     "Been 40 days since my dispute. No resolution, no callback despite two "
     "promises. Filing with the RBI Ombudsman this week and posting the whole "
     "chat log. Absolutely done with this company.",
     104, {"account_age_days": 1500, "karma": 2100, "post_count": 300}),

    ("u/priyaaa_23",
     "Someone called claiming to be from NimbusPay and asked for the OTP to "
     "'unblock' my account. I didn't share it but they knew my last transaction "
     "amount. How did they get that?!",
     96, {"account_age_days": 410, "karma": 620, "post_count": 88}),

    # The one person in the feed who actually tagged the brand. Everyone else
    # merely described us, which is the case a mentions-based queue misses.
    ("u/karthik_rn",
     "@NimbusPay double debit for one Swiggy order — ₹640 taken twice. Order id "
     "is in my app. Please fix, this happened last month too.",
     88, {"account_age_days": 900, "karma": 1500, "post_count": 130}),

    ("u/meera.writes",
     "Honestly the new UPI autopay flow is so much cleaner than before. "
     "Whoever redesigned it, well done.",
     80, {"account_age_days": 1600, "karma": 12000, "post_count": 1400}),

    # --- The judge's cases ------------------------------------------------
    # Hour-old account, no karma, brand-level accusation, not one checkable
    # fact. Answering this with a grounded policy reply is how a support team
    # spends its afternoon on someone who was never a customer.
    ("u/real_truth_9981",
     "NimbusPay is a SCAM, they hang the app on purpose just to make hype for "
     "their so-called launch. Fake company, total loot. BOYCOTT this app "
     "everyone, don't fall for the marketing stunt.",
     34, {"account_age_days": 0, "karma": 1, "post_count": 1}),

    # The same accusation, but 61,000 people will read it. Suspicion does not
    # earn silence here; an unanswered post of this size becomes the fact.
    ("@fintechbaba",
     "Third launch in a row where NimbusPay's site conveniently hangs at drop "
     "time. At this point the outage IS the marketing. Genuinely what is the "
     "use of a wallet you can't open?",
     30, {"account_age_days": 2400, "followers": 61000, "verified": True}),

    # --- The pattern agent's case ----------------------------------------
    # Six people, twelve minutes, one cause. Individually each is a mid
    # severity ticket; together they are an incident, and no single one of
    # them says so.
    ("u/tanvi_s",
     "App down again during peak hours?? Third time this month. Can't pay at the "
     "counter, standing here like an idiot.",
     14, {"account_age_days": 800, "karma": 940, "post_count": 150}),
    ("u/nk_bhatia",
     "NimbusPay payment failing since 10 minutes, money debited ₹1,200 but "
     "merchant says not received. Anyone else facing this right now?",
     12, {"account_age_days": 1300, "karma": 2600, "post_count": 240}),
    ("u/shalu.dev",
     "Is NimbusPay down? Every UPI payment I try is failing at the last step. "
     "₹890 stuck in pending for 15 mins.",
     9, {"account_age_days": 700, "karma": 410, "post_count": 60}),
    ("u/imran_qureshi",
     "निंबस से पेमेंट फेल हो रहा है पिछले 20 मिनट से, ₹2,300 कट गए और दुकानदार "
     "कह रहा है नहीं आया। ये क्या चल रहा है?",
     7, {"account_age_days": 1100, "karma": 780, "post_count": 95}),
    ("u/gaurav.sethi",
     "NimbusPay transactions failing continuously, ₹3,400 debited and stuck. "
     "Support chat is also not loading. 20 minutes now.",
     5, {"account_age_days": 950, "karma": 1120, "post_count": 175}),
    ("u/rekha_m",
     "Payment failed 4 times on NimbusPay in the last 10 minutes, ₹560 debited "
     "each time and nothing reached the shop. Please look into this.",
     3, {"account_age_days": 1450, "karma": 300, "post_count": 70}),
]


class MockFeedConnector(Connector):
    name = "mock"

    def __init__(self):
        self._replies_path = DATA_DIR / "mock_replies.json"

    async def fetch(self, limit: int = 20) -> list[Complaint]:
        now = datetime.now(timezone.utc)
        return [
            Complaint(
                external_id=f"mock_{i:03d}",
                channel=Channel.MOCK,
                author=author,
                text=text,
                url=f"https://example.invalid/thread/mock_{i:03d}",
                created_at=(now - timedelta(minutes=minutes_ago)).isoformat(),
                author_meta=AuthorMeta(**meta),
            )
            for i, (author, text, minutes_ago, meta) in enumerate(_SEED[:limit])
        ]

    async def reply(self, external_id: str, text: str) -> dict:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        existing = []
        if self._replies_path.exists():
            existing = json.loads(self._replies_path.read_text(encoding="utf-8"))
        receipt = {
            "reply_id": f"reply_{external_id}",
            "url": f"https://example.invalid/thread/{external_id}#reply",
            "posted_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
        }
        existing.append({**receipt, "in_reply_to": external_id, "text": text})
        self._replies_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
        return receipt
