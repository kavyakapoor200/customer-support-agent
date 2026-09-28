# Account Security & Enterprise Escalation Policy

## Section 1: P0 Security & SSO Lockouts
* Any ticket reporting enterprise Single Sign-On (SSO / Okta / SAML / Azure AD) login failures, organization-wide lockouts, or suspected unauthorized credential access is classified as **P0 Critical**.
* **Zero Autonomous Action:** Automated systems must NEVER attempt to reset credentials or modify authentication settings autonomously.
* All P0 security tickets must immediately be routed to the Security Operations (SecOps) queue with on-call paging.

## Section 2: Account Takeover & Fraud Protocol
* If a customer claims their account or API keys have been compromised, all active session tokens and API keys must be temporarily suspended.
* The ticket must be flagged with `HUMAN_REVIEW_REQUIRED` and assigned to Tier-2 Security Support.
