import csv
import json
import os
from datetime import datetime
from langchain.tools import tool



@tool
def get_customer_data(email: str) -> dict:
    """Load customer profile from customers.csv by email address."""
    try:
        with open("customers.csv", newline="") as f:
            for row in csv.DictReader(f):
                if row["email"].lower() == email.lower():
                    return dict(row)
        return {"error": f"No customer found with email: {email}"}
    except FileNotFoundError:
        return {"error": "customers.csv not found"}



@tool
def calculate_retention_offer(customer_tier: str, reason: str) -> dict:
    """
    Generate retention offers using retention_rules.json.
    customer_tier: 'premium', 'regular', or 'new'
    reason: 'financial_hardship', 'product_issues', or 'service_value'
    """
    try:
        with open("retention_rules.json") as f:
            rules = json.load(f)

        reason_map = {
            "financial_hardship": "financial_hardship",
            "product_issues": "product_issues",
            "service_value": "service_value",
        }
        tier_map = {
            "premium": "premium_customers",
            "regular": "regular_customers",
            "new": "new_customers",
        }

        reason_key = reason_map.get(reason.lower())
        tier_key = tier_map.get(customer_tier.lower())

        if not reason_key or reason_key not in rules:
            return {"error": f"Unknown reason: {reason}"}

        reason_rules = rules[reason_key]

        # product_issues uses sub-keys (overheating, battery_issues), not tier
        if reason_key == "product_issues":
            offers = []
            for sub_key, sub_offers in reason_rules.items():
                offers.extend(sub_offers)
            return {"offers": offers, "note": "Product issue offers — pick most relevant"}

        if not tier_key or tier_key not in reason_rules:
            # fallback to regular
            tier_key = "regular_customers"

        return {
            "customer_tier": customer_tier,
            "reason": reason,
            "offers": reason_rules.get(tier_key, [])
        }

    except FileNotFoundError:
        return {"error": "retention_rules.json not found"}



@tool
def update_customer_status(customer_id: str, action: str) -> dict:
    """
    Process cancellations or plan changes. Logs the action to actions_log.txt.
    action examples: 'cancelled', 'paused_6_months', 'downgraded_basic', 'discount_applied'
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "customer_id": customer_id,
        "action": action,
    }
    with open("actions_log.txt", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    return {
        "success": True,
        "customer_id": customer_id,
        "action": action,
        "message": f"Action '{action}' recorded for customer {customer_id}"
    }
