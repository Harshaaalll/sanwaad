# Wallet Charges and Billing

## [BIL-01] Published charge schedule
Wallet-to-bank transfers are free up to ₹10,000 per month. Above that,
**0.5% + GST** applies, capped at ₹15 per transaction. UPI payments to
merchants are always free to the customer.

## [BIL-02] Unexplained debit protocol
Where a customer reports a charge they do not recognise, we retrieve the
charge breakdown and state each component in rupees. We never respond with
a percentage alone; customers dispute percentages and accept itemisation.

## [BIL-03] Charge reversal on our error
A charge levied contrary to [BIL-01] is reversed in full within
**2 working days**, with no requirement for the customer to prove anything.

## [BIL-04] Subscription auto-debit
Recurring mandates can be cancelled by the customer in-app at any time.
Cancellation stops future debits but does not reverse a debit already
processed; that follows the refund path in [RFD-01].

## [BIL-05] GST invoices
GST invoices are available in-app under Statements within 48 hours of a
charged transaction. We do not email invoices on request; we direct to the
in-app path.
