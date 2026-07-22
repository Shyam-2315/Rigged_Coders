# Requirements Document

## Introduction

The Synthetic Population Generator is a Python module that creates realistic datasets of Indian users with statistically accurate financial behavior patterns. This synthetic dataset enables training of the TrustVest AI credit scoring model without requiring actual user data, supporting financial inclusion research through alternative credit assessment methodologies.

The generator produces 100,000 user profiles with 50+ attributes spanning demographics, financial behavior, digital activity, and credit indicators. All data follows realistic distributions and correlations observed in India's credit-invisible population, ensuring the trained model learns from patterns representative of students, gig workers, freelancers, farmers, and other underserved segments.

## Glossary

- **Generator**: The synthetic population generator module that creates user datasets
- **Dataset**: The collection of synthetic user profiles output as CSV format
- **User_Profile**: A single synthetic user record containing all attributes
- **Credit_Likelihood**: A calculated score (0-100) representing creditworthiness based on alternative financial signals
- **Engineered_Feature**: A derived attribute calculated mathematically from base attributes
- **Distribution**: The statistical pattern followed by attribute values across the dataset
- **Correlation**: The mathematical relationship between two attributes where one influences another
- **Alternative_Financial_Signal**: Non-traditional credit indicators like UPI transactions, utility payments, device trust
- **Parser**: A component that reads and validates the configuration parameters
- **Pretty_Printer**: A component that formats Generator configuration into human-readable text
- **Configuration**: The set of parameters controlling generator behavior (seed, size, output path)

## Requirements

### Requirement 1: Dataset Generation

**User Story:** As a data scientist, I want to generate a configurable number of synthetic user profiles, so that I can train credit scoring models at various scales.

#### Acceptance Criteria

1. THE Generator SHALL accept a NUM_USERS configuration parameter specifying dataset size
2. THE Generator SHALL produce exactly NUM_USERS user profiles with unique user_id values
3. THE Generator SHALL create 100,000 user profiles when NUM_USERS is set to 100000
4. WHEN NUM_USERS is less than 1, THE Generator SHALL raise a validation error
5. THE Generator SHALL complete generation within 5 minutes for 100,000 users on standard hardware

### Requirement 2: User Demographic Attributes

**User Story:** As a data scientist, I want synthetic users to have realistic Indian demographic profiles, so that the model learns from representative population segments.

#### Acceptance Criteria

1. THE Generator SHALL create user_id as a unique identifier for each User_Profile
2. THE Generator SHALL generate age values between 18 and 65 years following a normal distribution centered at 32
3. THE Generator SHALL assign gender from values: Male, Female, Other
4. THE Generator SHALL assign state from Indian states with population-weighted probabilities
5. THE Generator SHALL assign city_tier from values: Tier1, Tier2, Tier3, Rural
6. THE Generator SHALL assign education from values: High School, Undergraduate, Graduate, Postgraduate, PhD, No Formal Education
7. THE Generator SHALL assign marital_status from values: Single, Married, Divorced, Widowed
8. THE Generator SHALL generate dependents as integer values between 0 and 6
9. THE Generator SHALL assign occupation from values: Student, Software Engineer, Teacher, Farmer, Driver, Shop Owner, Delivery Partner, Freelancer, Doctor, Government Employee, Factory Worker, Construction Worker, Homemaker
10. THE Generator SHALL assign employment_type from values: Salaried, Self-Employed, Freelance, Daily Wage, Unemployed, Student
11. THE Generator SHALL generate years_employed as float values between 0 and 40 years

### Requirement 3: Financial Behavior Attributes

**User Story:** As a data scientist, I want synthetic users to exhibit realistic financial behaviors, so that the credit model learns meaningful patterns.

#### Acceptance Criteria

1. THE Generator SHALL generate monthly_income between ₹8,000 and ₹2,50,000 with 80% of values between ₹15,000 and ₹40,000
2. THE Generator SHALL generate monthly_expenses that do not exceed monthly_income
3. THE Generator SHALL calculate monthly_savings as monthly_income minus monthly_expenses
4. THE Generator SHALL ensure monthly_savings is non-negative
5. THE Generator SHALL generate bank_account_age between 0 and 25 years
6. THE Generator SHALL generate mobile_number_age between 0 and 20 years
7. THE Generator SHALL generate bank_balance between ₹0 and ₹5,00,000 correlated positively with monthly_income
8. THE Generator SHALL generate missed_utility_payments between 0 and 12 per year
9. THE Generator SHALL generate late_payment_ratio between 0.0 and 1.0 representing proportion of late payments
10. THE Generator SHALL calculate savings_rate as monthly_savings divided by monthly_income
11. THE Generator SHALL generate investment_frequency between 0 and 12 investments per year
12. THE Generator SHALL generate emergency_fund_months between 0 and 24 months of expenses covered
13. THE Generator SHALL generate loan_history_length between 0 and 30 years
14. THE Generator SHALL generate existing_small_loans between 0 and 5 active loans
15. THE Generator SHALL generate repayment_consistency between 0.0 and 1.0 representing payment reliability

### Requirement 4: Digital Financial Activity Attributes

**User Story:** As a data scientist, I want synthetic users to show realistic digital financial activity, so that the model learns from alternative credit signals.

#### Acceptance Criteria

1. THE Generator SHALL generate smartphone_years between 0 and 15 years of smartphone ownership
2. THE Generator SHALL generate digital_literacy_score between 0 and 100
3. THE Generator SHALL generate upi_transactions_per_month between 0 and 400 transactions
4. THE Generator SHALL generate upi_average_transaction between ₹0 and ₹10,000
5. THE Generator SHALL assign wallet_usage from values: None, Low, Medium, High
6. THE Generator SHALL generate ecommerce_transactions between 0 and 50 transactions per month
7. THE Generator SHALL generate mobile_recharge_frequency between 0 and 12 recharges per year
8. THE Generator SHALL generate utility_payment_method from values: Cash, UPI, Net Banking, Auto-Debit
9. THE Generator SHALL generate cash_transaction_ratio between 0.0 and 1.0 representing proportion of cash transactions

### Requirement 5: Device Trust Attributes

**User Story:** As a data scientist, I want synthetic users to have device trust indicators, so that the model can assess digital reliability.

#### Acceptance Criteria

1. THE Generator SHALL generate device_trust_score between 40 and 100
2. THE Generator SHALL generate device_age between 0 and 10 years
3. THE Generator SHALL generate sim_age between 0 and 15 years
4. THE Generator SHALL generate number_of_devices between 1 and 5 devices
5. THE Generator SHALL generate location_consistency between 0.0 and 1.0 representing location stability

### Requirement 6: Realistic Attribute Correlations

**User Story:** As a data scientist, I want attributes to correlate realistically, so that the trained model learns genuine behavioral patterns instead of noise.

#### Acceptance Criteria

1. WHEN monthly_income increases, THE Generator SHALL increase monthly_savings with positive correlation coefficient ≥ 0.5
2. WHEN monthly_savings increases, THE Generator SHALL increase repayment_consistency with positive correlation coefficient ≥ 0.4
3. WHEN missed_utility_payments increases, THE Generator SHALL decrease Credit_Likelihood with negative correlation coefficient ≤ -0.3
4. WHEN years_employed increases, THE Generator SHALL increase monthly_income with positive correlation coefficient ≥ 0.6
5. WHEN digital_literacy_score increases, THE Generator SHALL increase upi_transactions_per_month with positive correlation coefficient ≥ 0.5
6. WHEN upi_transactions_per_month increases, THE Generator SHALL decrease cash_transaction_ratio with negative correlation coefficient ≤ -0.4
7. WHEN repayment_consistency increases, THE Generator SHALL increase Credit_Likelihood with positive correlation coefficient ≥ 0.6
8. WHEN bank_account_age increases, THE Generator SHALL increase device_trust_score with positive correlation coefficient ≥ 0.3
9. WHEN sim_age increases, THE Generator SHALL increase device_trust_score with positive correlation coefficient ≥ 0.3
10. WHEN education level increases, THE Generator SHALL increase monthly_income with positive correlation coefficient ≥ 0.4

### Requirement 7: Engineered Features Calculation

**User Story:** As a data scientist, I want engineered features to be mathematically derived from base attributes, so that they represent composite indicators rather than random noise.

#### Acceptance Criteria

1. THE Generator SHALL calculate financial_discipline_index from savings_rate, late_payment_ratio, missed_utility_payments, and emergency_fund_months using weighted formula
2. THE Generator SHALL calculate digital_activity_score from upi_transactions_per_month, ecommerce_transactions, digital_literacy_score, and smartphone_years using weighted formula
3. THE Generator SHALL calculate income_stability_score from years_employed, employment_type, monthly_income variance, and loan_history_length using weighted formula
4. THE Generator SHALL calculate payment_consistency_score from repayment_consistency, late_payment_ratio, missed_utility_payments, and utility_payment_method using weighted formula
5. THE Generator SHALL calculate digital_trust_index from device_trust_score, sim_age, location_consistency, and number_of_devices using weighted formula
6. THE Generator SHALL ensure all Engineered_Feature values are bounded between 0 and 100
7. THE Generator SHALL NOT generate Engineered_Feature values randomly without using base attribute inputs

### Requirement 8: Credit Likelihood Target Calculation

**User Story:** As a data scientist, I want Credit_Likelihood to be calculated from multiple financial indicators, so that the model has a realistic supervised learning target.

#### Acceptance Criteria

1. THE Generator SHALL calculate Credit_Likelihood using weighted contributions from monthly_income, monthly_savings, repayment_consistency, payment_consistency_score, upi_transactions_per_month, device_trust_score, bank_account_age, emergency_fund_months, and digital_literacy_score
2. THE Generator SHALL apply negative weights to missed_utility_payments, late_payment_ratio, and existing_small_loans when calculating Credit_Likelihood
3. THE Generator SHALL ensure Credit_Likelihood values are bounded between 0 and 100
4. THE Generator SHALL ensure Credit_Likelihood correlates positively with financial_discipline_index with correlation coefficient ≥ 0.7
5. THE Generator SHALL ensure Credit_Likelihood correlates positively with payment_consistency_score with correlation coefficient ≥ 0.6
6. THE Generator SHALL ensure Credit_Likelihood correlates negatively with missed_utility_payments with correlation coefficient ≤ -0.4

### Requirement 9: Data Validation Rules

**User Story:** As a data scientist, I want the generator to enforce validation rules, so that the dataset contains no logically impossible values.

#### Acceptance Criteria

1. THE Generator SHALL ensure monthly_income is non-negative for all User_Profile records
2. THE Generator SHALL ensure monthly_savings does not exceed monthly_income for any User_Profile
3. THE Generator SHALL ensure monthly_expenses is non-negative and does not exceed monthly_income by more than 20%
4. THE Generator SHALL ensure all score attributes (digital_literacy_score, device_trust_score, Credit_Likelihood, Engineered_Feature values) are bounded between 0 and 100
5. THE Generator SHALL ensure all ratio attributes (savings_rate, late_payment_ratio, repayment_consistency, location_consistency, cash_transaction_ratio) are bounded between 0.0 and 1.0
6. THE Generator SHALL ensure age is between 18 and 65 years
7. THE Generator SHALL ensure years_employed does not exceed age minus 18
8. THE Generator SHALL ensure bank_account_age does not exceed age minus 10
9. IF monthly_income is less than ₹15,000, THEN THE Generator SHALL set investment_frequency to values less than 4 per year
10. IF age is less than 22, THEN THE Generator SHALL set loan_history_length to values less than 4 years

### Requirement 10: Dataset Output Format

**User Story:** As a data scientist, I want the dataset saved in CSV format with proper structure, so that I can load it directly into training pipelines.

#### Acceptance Criteria

1. THE Generator SHALL save the Dataset to the file path specified by OUTPUT_PATH configuration parameter
2. THE Generator SHALL create the output directory if it does not exist
3. THE Generator SHALL save the Dataset in CSV format with comma separators
4. THE Generator SHALL include column headers in the first row of the CSV file
5. THE Generator SHALL save all 50+ attributes as separate columns
6. THE Generator SHALL use UTF-8 encoding for the output file
7. THE Generator SHALL overwrite the output file if it already exists
8. WHEN OUTPUT_PATH is invalid or unwritable, THE Generator SHALL raise an error with descriptive message

### Requirement 11: Statistical Reporting

**User Story:** As a data scientist, I want the generator to print statistical summaries, so that I can verify dataset quality without external tools.

#### Acceptance Criteria

1. WHEN generation completes, THE Generator SHALL print the Dataset shape (rows, columns)
2. WHEN generation completes, THE Generator SHALL print the count of missing values per column
3. WHEN generation completes, THE Generator SHALL print summary statistics (mean, std, min, max, quartiles) for all numeric columns
4. WHEN generation completes, THE Generator SHALL print the correlation matrix between key financial attributes
5. WHEN generation completes, THE Generator SHALL print the distribution of Credit_Likelihood values across bins: 0-20, 20-40, 40-60, 60-80, 80-100
6. THE Generator SHALL print the count of unique values for categorical attributes: gender, state, city_tier, education, occupation, employment_type

### Requirement 12: Data Visualization

**User Story:** As a data scientist, I want the generator to create visualization plots, so that I can visually inspect dataset distributions.

#### Acceptance Criteria

1. THE Generator SHALL create a histogram plot of monthly_income distribution and save it to plots/income_distribution.png
2. THE Generator SHALL create a histogram plot of Credit_Likelihood distribution and save it to plots/credit_distribution.png
3. THE Generator SHALL create a correlation heatmap of key financial attributes and save it to plots/correlation_heatmap.png
4. THE Generator SHALL create a histogram plot of age distribution and save it to plots/age_distribution.png
5. THE Generator SHALL create the plots directory if it does not exist
6. THE Generator SHALL use clear axis labels, titles, and legends on all plots
7. IF visualization libraries are not installed, THEN THE Generator SHALL log a warning and continue without creating plots

### Requirement 13: Reproducibility

**User Story:** As a data scientist, I want to control randomness through a seed parameter, so that I can regenerate identical datasets for experiments.

#### Acceptance Criteria

1. THE Generator SHALL accept a RANDOM_SEED configuration parameter
2. WHEN RANDOM_SEED is set to a specific integer value, THE Generator SHALL produce identical datasets across multiple executions
3. THE Generator SHALL apply RANDOM_SEED to all random number generators used (numpy, faker, pandas random operations)
4. WHEN RANDOM_SEED is None, THE Generator SHALL generate different datasets on each execution
5. THE Generator SHALL log the RANDOM_SEED value used during generation

### Requirement 14: Configuration Parsing

**User Story:** As a developer, I want configuration parameters centralized in a config module, so that I can adjust generation settings without modifying code.

#### Acceptance Criteria

1. THE Parser SHALL read configuration parameters from a config.py module
2. THE Parser SHALL load NUM_USERS parameter with default value 100000
3. THE Parser SHALL load RANDOM_SEED parameter with default value 42
4. THE Parser SHALL load OUTPUT_PATH parameter with default value 'datasets/synthetic_users.csv'
5. THE Parser SHALL validate that NUM_USERS is a positive integer
6. THE Parser SHALL validate that OUTPUT_PATH is a valid string path
7. IF configuration parameters are invalid, THEN THE Parser SHALL raise a validation error with descriptive message

### Requirement 15: Configuration Pretty Printing

**User Story:** As a developer, I want configuration parameters displayed in readable format, so that I can verify settings before generation starts.

#### Acceptance Criteria

1. THE Pretty_Printer SHALL format Configuration parameters as human-readable text
2. THE Pretty_Printer SHALL display NUM_USERS, RANDOM_SEED, and OUTPUT_PATH values
3. THE Pretty_Printer SHALL output formatted configuration to console before generation starts
4. THE Pretty_Printer SHALL align parameter names and values in columnar format
5. FOR ALL valid Configuration objects, parsing configuration then printing then parsing SHALL produce an equivalent Configuration object (round-trip property)

### Requirement 16: Code Architecture

**User Story:** As a developer, I want the generator code organized into modular functions, so that I can maintain and extend functionality easily.

#### Acceptance Criteria

1. THE Generator SHALL implement user generation logic in separate functions for demographics, financial attributes, digital attributes, engineered features, and target calculation
2. THE Generator SHALL use Python type hints on all function signatures
3. THE Generator SHALL use dataclasses for structured data representation where appropriate
4. THE Generator SHALL implement logging using Python's logging module
5. THE Generator SHALL NOT implement generation as a single monolithic function exceeding 200 lines
6. THE Generator SHALL use meaningful variable names following Python naming conventions (snake_case)
7. THE Generator SHALL include docstrings for all public functions describing parameters, return values, and behavior

### Requirement 17: Dependency Management

**User Story:** As a developer, I want all required Python libraries specified, so that I can set up the environment correctly.

#### Acceptance Criteria

1. THE Generator SHALL use pandas library version 2.0 or higher for DataFrame operations
2. THE Generator SHALL use numpy library version 1.24 or higher for numerical operations
3. THE Generator SHALL use faker library version 18.0 or higher for synthetic data generation
4. THE Generator SHALL use matplotlib or seaborn for visualization plots
5. THE Generator SHALL use Python standard library modules: pathlib, dataclasses, typing, logging
6. THE Generator SHALL require Python version 3.12 or higher
7. THE Generator SHALL specify all dependencies in a requirements.txt file

### Requirement 18: Documentation

**User Story:** As a project stakeholder, I want the generator documented in project files, so that the feature is discoverable and usable by the team.

#### Acceptance Criteria

1. THE Generator SHALL be documented in a README.md section titled "Phase 1 — Synthetic Population Generator"
2. THE documentation SHALL describe the generator's purpose, usage instructions, and output format
3. THE documentation SHALL include example commands for running the generator
4. THE documentation SHALL specify the expected Dataset structure and attribute descriptions
5. THE Generator SHALL be versioned in CHANGELOG.md with entry for version v0.1.0
6. THE CHANGELOG entry SHALL list key features: dataset generation, attribute correlations, validation, visualization

### Requirement 19: Error Handling

**User Story:** As a developer, I want descriptive error messages when generation fails, so that I can diagnose and fix issues quickly.

#### Acceptance Criteria

1. WHEN configuration validation fails, THE Generator SHALL raise a ValueError with message describing which parameter is invalid
2. WHEN output path is unwritable, THE Generator SHALL raise an IOError with message describing the path issue
3. WHEN required libraries are missing, THE Generator SHALL raise an ImportError with message listing missing dependencies
4. WHEN generation encounters unexpected data corruption, THE Generator SHALL log the error details and raise an exception
5. THE Generator SHALL NOT suppress errors silently without logging or raising exceptions

### Requirement 20: Performance

**User Story:** As a data scientist, I want the generator to complete efficiently, so that I can iterate quickly during model development.

#### Acceptance Criteria

1. THE Generator SHALL complete generation of 100,000 users in less than 5 minutes on hardware with 8GB RAM and 4 CPU cores
2. THE Generator SHALL use vectorized pandas and numpy operations instead of row-by-row iteration where possible
3. THE Generator SHALL log progress messages at 25%, 50%, 75%, and 100% completion
4. THE Generator SHALL report total execution time upon completion
5. THE Generator SHALL use memory-efficient data types (int32 instead of int64 where appropriate) to reduce memory footprint
