# Refunds and Failed Transaction Reversals

## [RFD-01] Auto-reversal window
A UPI debit that fails at the beneficiary end auto-reverses within
**3 working days** (T+3) under NPCI rules. Before T+3 elapses we do not
raise a manual reversal; we tell the customer the exact date T+3 falls on.

## [RFD-02] Beyond T+3
If T+3 has passed and the money has not returned, we raise a chargeback with
the sponsor bank. Chargeback resolution SLA is **7 working days**. The
customer receives a reference number at the moment the chargeback is raised.

## [RFD-03] TAT compensation
Where a reversal exceeds T+3 through our fault, RBI's Harmonisation of TAT
circular entitles the customer to **₹100 per day of delay**, credited
automatically. Agents may state this entitlement. Agents may not state a
total amount before the delay period closes.

## [RFD-04] Merchant-side disputes
If the merchant received the money and did not deliver, the dispute is with
the merchant. We assist by supplying the transaction reference and RRN. We
do not refund merchant non-delivery from our own funds.

## [RFD-05] Goodwill credits
Goodwill credits above ₹500 require team-lead approval. No agent, human or
automated, may offer a goodwill credit on a public thread.

## [RFD-06] Duplicate debits
A double debit for a single order is reversed within **24 hours** of being
reported, without waiting for merchant confirmation. This is the one case
where we act before investigating.
