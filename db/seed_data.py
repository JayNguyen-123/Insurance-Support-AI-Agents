"""Synthetic sample data generator.

Ported from the original notebook's `generate_sample_data`. One bug fixed:
the state list had `'OH' 'GA'` (missing comma -> Python string concatenation
producing a single bogus state code `'OHGA'`); it's now `'OH', 'GA'`.

This data is for demos/tests only -- it is entirely synthetic (randomly
generated names, VINs, emails). Nothing here is or represents a real person.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

FIRST_NAMES = [
    "John", "Jane", "Robert", "Maria", "David", "Lisa", "Michael", "Sarah", "James", "Emily",
    "William", "Emma", "Joseph", "Olivia", "Charles", "Ava", "Thomas", "Isabella", "Daniel", "Mia",
    "Matthew", "Sophia", "Anthony", "Charlotte", "Christopher", "Amelia", "Andrew", "Harper",
    "Joshua", "Evelyn", "Ryan", "Abigail", "Brandon", "Ella", "Justin", "Scarlett", "Tyler", "Grace",
    "Alexander", "Chloe", "Kevin", "Victoria", "Jason", "Lily", "Brian", "Hannah", "Eric", "Aria",
    "Kyle", "Zoey",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
    "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter", "Roberts",
]

STATES = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA"]


def generate_sample_data(random_state: int = 42) -> dict[str, pd.DataFrame]:
    """Generate enriched synthetic sample data for all tables."""
    random.seed(random_state)
    np.random.seed(random_state)

    customers = pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(1, 1001)],
        "first_name": [random.choice(FIRST_NAMES) for _ in range(1000)],
        "last_name": [random.choice(LAST_NAMES) for _ in range(1000)],
        "email": [f"user{i}@example.com" for i in range(1, 1001)],
        "phone": [
            f"555-{str(random.randint(100, 999)).zfill(3)}-{str(random.randint(1000, 9999)).zfill(4)}"
            for _ in range(1000)
        ],
        "date_of_birth": [
            datetime(1980, 1, 1) + timedelta(days=random.randint(0, 10000)) for _ in range(1000)
        ],
        "state": [random.choice(STATES) for _ in range(1000)],
    })

    policies = pd.DataFrame({
        "policy_number": [f"POL{str(i).zfill(6)}" for i in range(1, 1501)],
        "customer_id": [f"CUST{str(random.randint(1, 1000)).zfill(5)}" for _ in range(1500)],
        "policy_type": [random.choice(["auto", "home", "life"]) for _ in range(1500)],
        "start_date": [
            datetime(2023, 1, 1) + timedelta(days=random.randint(0, 365)) for _ in range(1500)
        ],
        "premium_amount": [round(random.uniform(50, 500), 2) for _ in range(1500)],
        "billing_frequency": [random.choice(["monthly", "quarterly", "annual"]) for _ in range(1500)],
        "status": [random.choice(["active", "active", "active", "cancelled"]) for _ in range(1500)],
    })

    auto_policies = policies[policies["policy_type"] == "auto"].copy()
    auto_policy_details = pd.DataFrame({
        "policy_number": auto_policies["policy_number"],
        "vehicle_vin": [
            f"VIN{random.randint(10000000000000000, 99999999999999999)}" for _ in range(len(auto_policies))
        ],
        "vehicle_make": [
            random.choice(["Toyota", "Honda", "Ford", "Chevrolet", "Nissan"]) for _ in range(len(auto_policies))
        ],
        "vehicle_model": [
            random.choice(["Camry", "Civic", "F-150", "Malibu", "Altima"]) for _ in range(len(auto_policies))
        ],
        "vehicle_year": [random.randint(2015, 2023) for _ in range(len(auto_policies))],
        "liability_limit": [random.choice([50000, 100000, 300000]) for _ in range(len(auto_policies))],
        "collision_deductible": [random.choice([250, 500, 1000]) for _ in range(len(auto_policies))],
        "comprehensive_deductible": [random.choice([250, 500, 1000]) for _ in range(len(auto_policies))],
        "uninsured_motorist": [random.choice([0, 1]) for _ in range(len(auto_policies))],
        "rental_car_coverage": [random.choice([0, 1]) for _ in range(len(auto_policies))],
    })

    billing = pd.DataFrame({
        "bill_id": [f"BILL{str(i).zfill(6)}" for i in range(1, 5001)],
        "policy_number": [random.choice(policies["policy_number"]) for _ in range(5000)],
        "billing_date": [
            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90)) for _ in range(5000)
        ],
        "due_date": [
            datetime(2024, 1, 15) + timedelta(days=random.randint(0, 90)) for _ in range(5000)
        ],
        "amount_due": [round(random.uniform(100, 1000), 2) for _ in range(5000)],
        "status": [random.choice(["paid", "pending", "overdue"]) for _ in range(5000)],
    })

    payments = pd.DataFrame({
        "payment_id": [f"PAY{str(i).zfill(6)}" for i in range(1, 4001)],
        "bill_id": [random.choice(billing["bill_id"]) for _ in range(4000)],
        "payment_date": [
            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90)) for _ in range(4000)
        ],
        "amount": [round(random.uniform(50, 500), 2) for _ in range(4000)],
        "payment_method": [random.choice(["credit_card", "debit_card", "bank_transfer"]) for _ in range(4000)],
        "transaction_id": [f"TXN{random.randint(100000, 999999)}" for _ in range(4000)],
        "status": [random.choice(["completed", "pending", "failed"]) for _ in range(4000)],
    })

    claims = pd.DataFrame({
        "claim_id": [f"CLM{str(i).zfill(6)}" for i in range(1, 301)],
        "policy_number": [random.choice(policies["policy_number"]) for _ in range(300)],
        "claim_date": [
            datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90)) for _ in range(300)
        ],
        "incident_type": [
            random.choice(["collision", "theft", "property_damage", "medical", "liability"])
            for _ in range(300)
        ],
        "estimated_loss": [round(random.uniform(500, 20000), 2) for _ in range(300)],
        "status": [
            random.choice(["submitted", "under_review", "approved", "paid", "denied"]) for _ in range(300)
        ],
    })

    return {
        "customers": customers,
        "policies": policies,
        "auto_policy_details": auto_policy_details,
        "billing": billing,
        "payments": payments,
        "claims": claims,
    }
