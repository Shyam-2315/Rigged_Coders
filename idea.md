# 💰 TrustVest AI
### Transparent Credit Scoring & AI-Driven Micro-Investment Advisor

> Empowering India's credit-invisible population with explainable AI and beginner-friendly investment guidance.

---

# 📖 Overview

TrustVest AI is an AI-powered FinTech platform designed to improve financial inclusion for individuals who lack a traditional credit history and have little or no investment experience.

The platform combines two core services into a single application:

- 📊 Transparent Credit-Likelihood Scoring
- 📈 AI-Powered Micro-Investment Advisor

Unlike traditional financial institutions that rely solely on CIBIL scores and banking history, TrustVest AI evaluates alternative digital financial behavior to estimate a user's creditworthiness while explaining exactly how the score was calculated.

For investments, the platform uses a conversational AI-based risk assessment to recommend beginner-friendly portfolios and simulate future investment growth.

> **Disclaimer:** This project is a hackathon prototype created for educational and demonstration purposes only. It does **not** provide regulated financial or investment advice.

---

# 🎯 Problem Statement

India has more than **190 million credit-invisible adults** who cannot easily access loans because they lack traditional credit history.

Commonly affected users include:

- Students
- Gig workers
- Freelancers
- Daily wage earners
- Small business owners
- Farmers
- Homemakers
- First-time salaried employees

At the same time, millions of first-time investors from Tier-2 and Tier-3 cities struggle with:

- Low financial literacy
- Fear of investing
- Generic investment advice
- Complex financial products
- Lack of personalized guidance

TrustVest AI addresses both challenges in a single transparent AI platform.

---

# ✨ Features

## 📊 Transparent Credit Scoring

Generate an explainable credit-likelihood score using alternative financial signals instead of traditional credit history.

### Uses Digital Signals

- Mobile recharge frequency
- Utility bill payment consistency
- UPI transaction history
- E-commerce purchases
- Wallet usage
- Savings behavior
- Income estimation
- Bank account age
- Mobile number age
- Device trust score

---

## 🔍 Explainable AI

Instead of showing only a number, the system explains:

- Top 3 factors increasing the score
- Top 3 factors decreasing the score
- Confidence score
- Personalized improvement suggestions

Example:

```

Credit Score: 78/100

Top Positive Factors

✔ Regular Utility Payments
✔ Consistent Mobile Recharge
✔ Active UPI Usage

Suggestions

• Save regularly
• Continue paying bills before due date

Estimated Future Score: 83

```

---

## 🤖 AI Risk Assessment

The platform asks a short conversational questionnaire (5–8 questions) to determine the user's investment risk profile.

Questions include:

- Investment goal
- Monthly investment amount
- Income source
- Investment experience
- Investment horizon
- Reaction to market losses

The AI classifies users into:

- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

---

## 📈 Micro-Investment Advisor

Based on the user's profile, the system recommends a diversified investment allocation.

Example:

| Asset | Allocation |
|--------|------------|
| Liquid Fund | 20% |
| Debt Fund | 30% |
| Index Fund | 35% |
| Gold ETF | 15% |

Recommendations are explained in simple language suitable for first-time investors.

---

## 📉 Growth Simulation

Users can visualize projected investment growth for:

- Conservative Scenario
- Expected Scenario
- Optimistic Scenario

Time periods:

- 1 Year
- 3 Years
- 5 Years

---

## 📊 Admin Dashboard

The dashboard includes:

- Credit Scores
- Risk Profiles
- Portfolio Recommendations
- Explainability Reports
- Growth Charts
- User Analytics

Includes **10 sample users** across:

- Low Risk
- Medium Risk
- High Risk

---

# 🏗 System Architecture

```

                    User

│

▼

        React Dashboard

│

┌───────────────┬───────────────┐

▼               ▼               ▼

Credit Engine   Risk Engine   Investment Engine

│               │               │

▼               ▼               ▼

Explainable AI  Questionnaire Portfolio Allocation

│               │               │

└───────────────┼───────────────┘

▼

Recommendation Dashboard

```

---

# ⚙️ Tech Stack

## Frontend

- React.js
- TypeScript
- Tailwind CSS
- Chart.js / Recharts

---

## Backend

- FastAPI
- Python 3.12+

---

## Machine Learning

- Scikit-Learn
- XGBoost
- SHAP
- Pandas
- NumPy

---

## Database

- PostgreSQL
- Redis

---

## Deployment

- Docker
- Nginx
- GitHub Actions
- AWS / Render / Railway

---

# 📂 Project Structure

```

trustvest-ai/

│

├── backend/

│   ├── api/

│   ├── models/

│   ├── ml/

│   ├── services/

│   ├── utils/

│   ├── database/

│   └── main.py

│

├── frontend/

│   ├── components/

│   ├── pages/

│   ├── hooks/

│   ├── charts/

│   └── assets/

│

├── datasets/

│

├── notebooks/

│

├── docs/

│

├── docker/

│

├── README.md

└── docker-compose.yml

```

---

# 🚀 Workflow

## Step 1

User creates an account.

↓

## Step 2

The user provides basic financial behavior information.

↓

## Step 3

The Credit Engine calculates a Credit-Likelihood Score.

↓

## Step 4

The Explainable AI identifies the top factors influencing the score.

↓

## Step 5

The AI chatbot conducts a short investment risk assessment.

↓

## Step 6

The Investment Engine classifies the user's risk level.

↓

## Step 7

A personalized micro-investment portfolio is generated.

↓

## Step 8

The dashboard displays projected investment growth under multiple scenarios.

↓

## Step 9

Users receive actionable suggestions to improve both their credit profile and investment habits.

---

# 🔌 API Endpoints

## Credit Score

```

POST /api/v1/credit-score

```

Returns:

- Credit Score
- Confidence Score
- Feature Importance
- Improvement Suggestions

---

## Risk Assessment

```

POST /api/v1/risk-profile

```

Returns:

- Risk Bucket
- Questionnaire Summary

---

## Investment Recommendation

```

POST /api/v1/recommendation

```

Returns:

- Portfolio Allocation
- Growth Projection
- Educational Disclaimer

---

## Dashboard

```

GET /api/v1/dashboard/users

```

Returns:

- Sample Users
- Analytics
- Charts

---

# 🧠 Machine Learning Pipeline

```

Raw User Data

↓

Data Cleaning

↓

Feature Engineering

↓

Credit Score Prediction

↓

SHAP Explainability

↓

Risk Assessment

↓

Portfolio Recommendation

↓

Growth Simulation

↓

Dashboard

```

---

# 📊 Sample User

```

Name

Ravi Patel

Age

24

Occupation

Delivery Partner

Monthly Income

₹22,000

```

### Credit Score

```

72/100

```

Top Factors

- Utility Bill Payments
- Regular UPI Transactions
- Mobile Recharge Consistency

Risk Profile

```

Medium Risk

```

Investment Recommendation

| Asset | Allocation |
|--------|------------|
| Debt Fund | 30% |
| Index Fund | 40% |
| Gold ETF | 15% |
| Liquid Fund | 15% |

---

# 🔒 Transparency

Every recommendation includes:

- Credit score explanation
- Feature importance
- Confidence score
- Personalized suggestions
- Investment rationale
- Growth assumptions

No "black-box" AI decisions.

---

# 🎯 Target Users

- Students
- Gig Workers
- Freelancers
- Farmers
- Homemakers
- Small Business Owners
- First-Time Investors
- Tier-2 & Tier-3 City Residents

---

# 🌟 Future Enhancements

- Aadhaar Consent Integration
- DigiLocker Verification
- Account Aggregator APIs
- ONDC Integration
- Voice Assistant (Regional Languages)
- WhatsApp Banking Bot
- Financial Literacy Modules
- AI Chat Assistant
- Loan Marketplace Integration
- Fraud Detection Engine

---

# 🏆 Why TrustVest AI?

✅ Transparent AI decisions

✅ Alternative credit scoring

✅ Explainable financial recommendations

✅ Beginner-friendly investment guidance

✅ Personalized improvement suggestions

✅ Supports financial inclusion

✅ Designed specifically for underserved Indian users

---

# ⚠️ Disclaimer

**This application is a hackathon prototype developed for educational and demonstration purposes only. Credit scores, investment recommendations, portfolio allocations, and projected returns are simulated using machine learning models and historical assumptions. They do not constitute regulated financial, investment, lending, or legal advice. Users should consult licensed financial professionals before making investment or borrowing decisions.**

---

# 👨‍💻 Developed For

**Hackathon Theme:**
Transparent Credit Scoring & AI-Driven Micro-Investment Advisor

Built with ❤️ to promote **Financial Inclusion**, **Explainable AI**, and **Responsible FinTech Innovation**.