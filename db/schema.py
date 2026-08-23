"""SQLite schema for the insurance support database.

Ported from the original notebook's `drop_and_create_tables`, with one fix:
the DROP script referenced a table named `billings` (typo) while the table is
actually created as `billing`, so the drop silently no-op'd on every run
after the first. Both DROP and CREATE now agree on `billing`.
"""

from __future__ import annotations

DROP_TABLES_SQL = """
DROP TABLE IF EXISTS claims;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS billing;
DROP TABLE IF EXISTS auto_policy_details;
DROP TABLE IF EXISTS policies;
DROP TABLE IF EXISTS customers;
"""

CREATE_TABLES_SQL = """
CREATE TABLE customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    email VARCHAR(100),
    phone VARCHAR(20),
    date_of_birth DATE,
    state VARCHAR(20)
);

CREATE TABLE policies (
    policy_number VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20),
    policy_type VARCHAR(50),
    start_date DATE,
    premium_amount DECIMAL(10,2),
    billing_frequency VARCHAR(20),
    status VARCHAR(20),
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE auto_policy_details (
    policy_number VARCHAR(20) PRIMARY KEY,
    vehicle_vin VARCHAR(50),
    vehicle_make VARCHAR(50),
    vehicle_model VARCHAR(50),
    vehicle_year INTEGER,
    liability_limit DECIMAL(10,2),
    collision_deductible DECIMAL(10,2),
    comprehensive_deductible DECIMAL(10,2),
    uninsured_motorist BOOLEAN,
    rental_car_coverage BOOLEAN,
    FOREIGN KEY (policy_number) REFERENCES policies(policy_number)
);

CREATE TABLE billing (
    bill_id VARCHAR(20) PRIMARY KEY,
    policy_number VARCHAR(20),
    billing_date DATE,
    due_date DATE,
    amount_due DECIMAL(10,2),
    status VARCHAR(20),
    FOREIGN KEY (policy_number) REFERENCES policies(policy_number)
);

CREATE TABLE payments (
    payment_id VARCHAR(20) PRIMARY KEY,
    bill_id VARCHAR(20),
    payment_date DATE,
    amount DECIMAL(10,2),
    payment_method VARCHAR(50),
    transaction_id VARCHAR(100),
    status VARCHAR(20),
    FOREIGN KEY (bill_id) REFERENCES billing(bill_id)
);

CREATE TABLE claims (
    claim_id VARCHAR(20) PRIMARY KEY,
    policy_number VARCHAR(20),
    claim_date DATE,
    incident_type VARCHAR(100),
    estimated_loss DECIMAL(10,2),
    status VARCHAR(20),
    FOREIGN KEY (policy_number) REFERENCES policies(policy_number)
);

CREATE INDEX idx_policies_customer_id ON policies(customer_id);
CREATE INDEX idx_billing_policy_number ON billing(policy_number);
CREATE INDEX idx_payments_bill_id ON payments(bill_id);
CREATE INDEX idx_claims_policy_number ON claims(policy_number);
"""
