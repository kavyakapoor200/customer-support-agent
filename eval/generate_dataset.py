"""Generates the 100-ticket clean synthetic SaaS support evaluation dataset."""
import json
from pathlib import Path


def generate_dataset() -> list[dict]:
    tickets = []
    tid = 1

    # =========================================================================
    # 1. English Tickets (50 Total)
    # =========================================================================
    # 1.1 Refunds (12)
    refund_en = [
        ("I was charged twice on my card, duplicate accidental renewal charge, please refund my money back for $45", 45.0, True, False, "01_refund_sla"),
        ("Please issue a refund for this accidental annual renewal, amount is $39.00", 39.0, True, False, "01_refund_sla"),
        ("I made an accidental in-app upgrade, request money back reimbursement for $25.00 within 14 days", 25.0, True, False, "01_refund_sla"),
        ("Duplicate billing occurred on my account yesterday, please refund the second charge of $15.00", 15.0, True, False, "01_refund_sla"),
        ("Double charge noticed on my invoice, please refund $48.50 immediately", 48.5, True, False, "01_refund_sla"),
        ("Please refund my corporate account $180 for annual license dispute", 180.0, False, False, "01_refund_sla"),
        ("I need a full refund for $350 annual plan charge from 3 days ago", 350.0, False, False, "01_refund_sla"),
        ("Refund requested for $79 charge on Pro tier", 79.0, False, False, "01_refund_sla"),
        ("Accidental subscription renewal of $120, requesting refund", 120.0, False, False, "01_refund_sla"),
        ("Please refund $49.99 subscription fee charged this morning", 49.99, True, False, "01_refund_sla"),
        ("I was wrongly billed twice, please reimburse $30 chargeback", 30.0, True, False, "01_refund_sla"),
        ("Requesting refund of $500 for enterprise seat addition", 500.0, False, False, "01_refund_sla"),
    ]
    for text, amt, auto, fraud, pol in refund_en:
        tickets.append({
            "ticket_id": f"TK-EN-{tid:03d}",
            "text": text,
            "language": "english",
            "expected_action": "refund",
            "amount_usd": amt,
            "should_auto_execute": auto,
            "is_fraud_or_dispute": fraud,
            "ground_truth_policy_id": pol
        })
        tid += 1

    # 1.2 Cancellations (10)
    cancel_en = [
        "Please cancel my subscription before next billing cycle.",
        "I want to unsubscribe and terminate my membership immediately.",
        "Kindly stop auto renewal on my Pro workspace plan.",
        "Cancel our monthly subscription effective at the end of the billing period.",
        "Please terminate my team account, we are migrating elsewhere.",
        "I need to cancel my paid plan and downgrade to free tier.",
        "Unsubscribe me from all recurring charges and terminate subscription.",
        "How do I cancel my subscription? Please do it for me.",
        "Please stop renewal on my annual enterprise license.",
        "Terminate subscription immediately and confirm cancellation date.",
    ]
    for text in cancel_en:
        tickets.append({
            "ticket_id": f"TK-EN-{tid:03d}",
            "text": text,
            "language": "english",
            "expected_action": "cancel_subscription",
            "amount_usd": None,
            "should_auto_execute": True,
            "is_fraud_or_dispute": False,
            "ground_truth_policy_id": "02_subscription_cancellation"
        })
        tid += 1

    # 1.3 Billing Disputes (10)
    dispute_en = [
        "Unauthorized charge appeared on my card, this is fraudulent activity.",
        "I never authorized this $65 charge, someone used my stolen card.",
        "This is an unknown dispute charge from your company on my bank statement.",
        "Fraud alert: unauthorized transaction on my credit card for $120.",
        "I was overbilled and overcharged $80 on my monthly invoice without notice.",
        "Unauthorized debit on my Visa card, I never bought this product.",
        "Someone hacked my account and made unauthorized charges, please dispute.",
        "My corporate credit card was charged for seats we never added, dispute charge.",
        "Unauthorized subscription fee of $99, I demand an investigation.",
        "Dispute: I was charged despite cancelling two months ago, unauthorized.",
    ]
    for text in dispute_en:
        tickets.append({
            "ticket_id": f"TK-EN-{tid:03d}",
            "text": text,
            "language": "english",
            "expected_action": "billing_dispute",
            "amount_usd": 65.0,
            "should_auto_execute": False, # Disputes should NEVER auto-refund without verification
            "is_fraud_or_dispute": True,
            "ground_truth_policy_id": "01_refund_sla"
        })
        tid += 1

    # 1.4 Account Escalations (10)
    escalate_en = [
        "URGENT: Entire engineering team locked out of Okta SSO right now, P0 blocker.",
        "Security breach: compromised admin API keys detected in our workspace.",
        "All our developers cannot login due to SAML authentication failure.",
        "Emergency: Two-factor authentication lockout on CEO account.",
        "P0 Incident: Single sign-on down across the whole company domain.",
        "Urgent security alert: unauthorized login detected from suspicious IP address.",
        "Our workspace admin account is locked and production deployments are blocked.",
        "P0 blocker: SSO integration throwing 500 errors for all employees.",
        "Security emergency: suspect our OAuth token was leaked publicly.",
        "Immediate help needed: enterprise domain lockout affecting 500 users.",
    ]
    for text in escalate_en:
        tickets.append({
            "ticket_id": f"TK-EN-{tid:03d}",
            "text": text,
            "language": "english",
            "expected_action": "account_escalation",
            "amount_usd": None,
            "should_auto_execute": False, # P0 always requires human review
            "is_fraud_or_dispute": False,
            "ground_truth_policy_id": "03_security_escalation"
        })
        tid += 1

    # 1.5 General Inquiries (8)
    inquiry_en = [
        "Where can I find documentation on webhook API rate limits?",
        "How do I invite team members to our workspace?",
        "What are the pricing differences between Pro and Enterprise tiers?",
        "Do you support exporting billing invoices as PDF?",
        "Can I change the billing currency from USD to EUR?",
        "Where can I find the API documentation guide for Python SDK?",
        "Is there an uptime status page for your cloud platform?",
        "How do I update the credit card on file for our account?",
    ]
    for text in inquiry_en:
        tickets.append({
            "ticket_id": f"TK-EN-{tid:03d}",
            "text": text,
            "language": "english",
            "expected_action": "general_inquiry",
            "amount_usd": None,
            "should_auto_execute": True,
            "is_fraud_or_dispute": False,
            "ground_truth_policy_id": "02_subscription_cancellation"
        })
        tid += 1

    # =========================================================================
    # 2. Hinglish Tickets (35 Total)
    # =========================================================================
    hinglish_cases = [
        # Refunds (10)
        ("Mera renewal galti se ho gaya kal, $35 charge hua hai, refund kardo bhai please.", 35.0, "refund", True),
        ("Bhai do baar charge cut gaya account se, duplicate charge ka refund kardo $20.", 20.0, "refund", True),
        ("Galti se annual plan buy kar liya, paise wapas chahiye $40 refund kijiye.", 40.0, "refund", True),
        ("Mujhe service pasand nahi aayi, 14 days ke andar $15 refund maang raha hoon.", 15.0, "refund", True),
        ("Accidental payment charge ho gaya card se, please refund kardo $49.", 49.0, "refund", True),
        ("Mera yearly renewal charge $180 ho gaya, please supervisor se refund approve karvao.", 180.0, "refund", False),
        ("Bhai $120 ka galat renewal kat gaya, refund chahiye urgent.", 120.0, "refund", False),
        ("Galti se Pro plan renew ho gaya $75 ka, paise wapas kar do.", 75.0, "refund", False),
        ("Mera $250 ka enterprise charge refund kardo galti se team member ne add kiya.", 250.0, "refund", False),
        ("Account se $30 refund kardo duplicate deduction hua hai.", 30.0, "refund", True),

        # Cancellations (8)
        ("Mera yearly plan cancel kardo bhai, auto debit nahi chahiye.", None, "cancel_subscription", True),
        ("Subscription band kardo please, next month se renew mat karna.", None, "cancel_subscription", True),
        ("Humari membership terminate kardo, ab use nahi karte.", None, "cancel_subscription", True),
        ("Auto renewal off kardo aur subscription cancel kijiye.", None, "cancel_subscription", True),
        ("Bhai please mera plan cancel kar do end of month tak.", None, "cancel_subscription", True),
        ("Mujhe subscription hata do, service discontinue karni hai.", None, "cancel_subscription", True),
        ("Pro plan unsubscribe kardo, billing stop karo.", None, "cancel_subscription", True),
        ("Subscription cancel kardo bhai jaldi.", None, "cancel_subscription", True),

        # Disputes (7)
        ("Mera card se unauthorized $60 kat gaya, fraud charge dispute karo.", 60.0, "billing_dispute", False),
        ("Maine ye purchase nahi kiya, unknown charge dispute kardo.", 45.0, "billing_dispute", False),
        ("Stolen card se unauthorized transaction hua hai $90 ka.", 90.0, "billing_dispute", False),
        ("Galat charge lagaya hai bill pe, unauthorized payment dispute.", 70.0, "billing_dispute", False),
        ("Bhai ye payment maine authorize nahi kiya, fraud transaction hai.", 85.0, "billing_dispute", False),
        ("Card hacked lag raha hai, unauthorized deduction dispute.", 110.0, "billing_dispute", False),
        ("Overbilled amount dispute: bina bataye extra charge kar diya.", 50.0, "billing_dispute", False),

        # Escalations (5)
        ("URGENT P0: Poori team Okta SSO se lock ho gayi hai, login nahi ho raha.", None, "account_escalation", False),
        ("Emergency security issue: admin account compromised ho gaya hai.", None, "account_escalation", False),
        ("P0 blocker bhai: SAML SSO fail ho raha hai poore office ka.", None, "account_escalation", False),
        ("Two-factor authentication locked out, login failed urgent help.", None, "account_escalation", False),
        ("Critical P0: Organization domain lockout ho gaya sabka.", None, "account_escalation", False),

        # Inquiries (5)
        ("Pricing aur features ka details kahan milega bhai?", None, "general_inquiry", True),
        ("API documentation ka link bhej do please.", None, "general_inquiry", True),
        ("Invoice PDF download kaise karein?", None, "general_inquiry", True),
        ("Workspace mein new team members ko invite kaise karte hain?", None, "general_inquiry", True),
        ("Payment method credit card change kaise karein?", None, "general_inquiry", True),
    ]

    for text, amt, act, auto in hinglish_cases:
        tickets.append({
            "ticket_id": f"TK-HI-{tid:03d}",
            "text": text,
            "language": "hinglish",
            "expected_action": act,
            "amount_usd": amt,
            "should_auto_execute": auto,
            "is_fraud_or_dispute": (act == "billing_dispute"),
            "ground_truth_policy_id": "01_refund_sla" if act in ("refund", "billing_dispute") else ("03_security_escalation" if act == "account_escalation" else "02_subscription_cancellation")
        })
        tid += 1

    # =========================================================================
    # 3. Devanagari Hindi Tickets (15 Total)
    # =========================================================================
    devanagari_cases = [
        # Refunds (4)
        ("कृपया मेरा रिफंड प्रोसेस करें, $40 की डुप्लीकेट कटौती हुई है।", 40.0, "refund", True),
        ("गलती से वार्षिक योजना का नवीनीकरण हो गया, कृपया $45 का रिफंड दें।", 45.0, "refund", True),
        ("कृपया मेरे खाते में $150 का रिफंड प्रोसेस करें।", 150.0, "refund", False),
        ("दो बार पैसे कट गए हैं, कृपया तुरंत रिफंड प्रदान करें।", 30.0, "refund", True),

        # Cancellations (4)
        ("कृपया मेरा सब्सक्रिप्शन तुरंत रद्द करें और सेवा समाप्त करें।", None, "cancel_subscription", True),
        ("बिलिंग चक्र के अंत में सदस्यता समाप्त करने का अनुरोध करता हूँ।", None, "cancel_subscription", True),
        ("हमारा प्लान रद्द करें, आगे कोई शुल्क न काटें।", None, "cancel_subscription", True),
        ("ऑटो नवीनीकरण बंद करें और सब्सक्रिप्शन रद्द करें।", None, "cancel_subscription", True),

        # Disputes (3)
        ("यह अनाधिकृत लेनदेन है, मैंने यह खरीद नहीं की, विवाद दर्ज करें।", 75.0, "billing_dispute", False),
        ("धोखाधड़ी: मेरे क्रेडिट कार्ड से अनधिकृत कटौती हुई है।", 120.0, "billing_dispute", False),
        ("अज्ञात शुल्क का विवाद दर्ज करें और जांच करें।", 50.0, "billing_dispute", False),

        # Escalations (2)
        ("अति आवश्यक P0: हमारी पूरी कंपनी Okta SSO से बाहर हो गई है।", None, "account_escalation", False),
        ("सुरक्षा आपातकाल: एडमिन खाता लॉक हो गया है, तुरंत सहायता दें।", None, "account_escalation", False),

        # Inquiries (2)
        ("एपीआई प्रलेखन (Documentation) कहाँ उपलब्ध है?", None, "general_inquiry", True),
        ("योजनाओं के मूल्य और सुविधाओं की जानकारी चाहिए।", None, "general_inquiry", True),
    ]

    for text, amt, act, auto in devanagari_cases:
        tickets.append({
            "ticket_id": f"TK-DEV-{tid:03d}",
            "text": text,
            "language": "hindi",
            "expected_action": act,
            "amount_usd": amt,
            "should_auto_execute": auto,
            "is_fraud_or_dispute": (act == "billing_dispute"),
            "ground_truth_policy_id": "01_refund_sla" if act in ("refund", "billing_dispute") else ("03_security_escalation" if act == "account_escalation" else "02_subscription_cancellation")
        })
        tid += 1

    return tickets


def main():
    dataset = generate_dataset()
    assert len(dataset) == 100, f"Expected 100 tickets, got {len(dataset)}"

    out_dir = Path("eval/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "saas_tickets_eval.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(dataset)} clean, uncontaminated SaaS evaluation tickets at: {out_file}")


if __name__ == "__main__":
    main()
