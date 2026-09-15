# KYC, Freezes and Account Access

## [KYC-01] Why accounts freeze
Wallets freeze for one of three reasons: incomplete re-KYC, a velocity rule
trip, or a law-enforcement lien. We tell the customer which of the three
applies. Vagueness here generates the most public escalation of any category.

## [KYC-02] Re-KYC resolution
Incomplete re-KYC is resolved by the customer in-app in under 10 minutes with
Aadhaar OTP or a video-KYC slot. Funds are never lost, only inaccessible.
State this explicitly — customers assume their money is gone.

## [KYC-03] Velocity holds
> also: on hold, too many transactions, many payments today, sudden hold, temporarily restricted, limit reached, account paused
An automated velocity hold is reviewed within **24 hours**. If the review
clears, access is restored automatically without customer action.

## [KYC-04] Law-enforcement liens
Where a lien is placed by an enforcement agency, we cannot release funds and
cannot disclose case details. We give the customer the nodal officer contact
and the lien reference. We do not apologise for the lien itself, as this
implies fault we cannot assess.

## [KYC-05] Never ask for credentials
We never ask for a PIN, an OTP, a password, a card CVV, or a screen-share on
any channel, public or private. If a customer reports being asked for these,
treat it as a fraud report under [PRV-03] and escalate immediately.
