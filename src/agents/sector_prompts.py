"""
Sector-Specific Analysis Frameworks

Maps yfinance sector/industry classifications to specialized analytical
frameworks for the fundamentals and sentiment agents. Each sector config
provides domain-specific metrics, valuation approaches, and sentiment
context so the LLM produces expert-level, sector-appropriate analysis.

To add a new sector:
  1. Add the yfinance industry string to INDUSTRY_TO_SECTOR_KEY
  2. Add a SectorConfig entry to SECTOR_CONFIGS
  That's it — no graph, UI, or pipeline changes needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SectorConfig:
    """Configuration for a sector-specific analysis framework."""

    key: str                        # Internal key (e.g., "semiconductors")
    label: str                      # Human-readable label shown in agent_name
    analysis_framework: str         # Multi-paragraph prompt injected into fundamentals agent
    key_metrics: list[str] = field(default_factory=list)  # Metrics to emphasize
    sentiment_context: str = ""     # One-liner for the sentiment agent


# ---------------------------------------------------------------------------
# Mapping: yfinance industry string → internal sector key
# ---------------------------------------------------------------------------

INDUSTRY_TO_SECTOR_KEY: dict[str, str] = {
    # Software
    "Software - Infrastructure":            "software_infra",
    "Software - Application":               "software_app",
    "Information Technology Services":       "software_app",

    # Cybersecurity (subset of software with unique dynamics)
    # Note: yfinance classifies most cybersecurity as "Software - Infrastructure"
    # so specific companies may need to be handled via override if needed

    # Hardware / Semiconductors
    "Semiconductors":                       "semiconductors",
    "Semiconductor Equipment & Materials":  "semi_equipment",
    "Consumer Electronics":                 "hardware_consumer",
    "Electronic Components":                "hardware_consumer",
    "Computer Hardware":                    "hardware_consumer",
    "Scientific & Technical Instruments":   "hardware_consumer",
    "Communication Equipment":              "hardware_consumer",

    # Healthcare / Life Sciences
    "Biotechnology":                        "biotech",
    "Drug Manufacturers - General":         "pharma",
    "Drug Manufacturers - Specialty & Generic": "pharma",
    "Medical Devices":                      "medtech",
    "Medical Instruments & Supplies":       "medtech",
    "Diagnostics & Research":               "medtech",
    "Healthcare Plans":                     "healthcare_services",
    "Health Information Services":          "healthcare_services",
    "Medical Care Facilities":             "healthcare_services",
    "Pharmaceutical Retailers":            "healthcare_services",

    # Financial Services
    "Banks - Diversified":                  "banking",
    "Banks - Regional":                     "banking",
    "Credit Services":                      "fintech",
    "Insurance - Diversified":              "insurance",
    "Insurance - Life":                     "insurance",
    "Insurance - Property & Casualty":      "insurance",
    "Insurance Brokers":                    "insurance",
    "Capital Markets":                      "capital_markets",
    "Financial Data & Stock Exchanges":     "capital_markets",
    "Asset Management":                     "capital_markets",

    # Energy
    "Oil & Gas Integrated":                 "energy",
    "Oil & Gas E&P":                        "energy",
    "Oil & Gas Midstream":                  "energy",
    "Oil & Gas Refining & Marketing":       "energy",
    "Oil & Gas Equipment & Services":       "energy",
    "Uranium":                              "energy",
    "Solar":                                "renewables",
    "Renewable Utilities":                  "renewables",

    # Consumer
    "Internet Retail":                      "ecommerce",
    "Specialty Retail":                     "retail",
    "Discount Stores":                      "retail",
    "Home Improvement Retail":              "retail",
    "Apparel Retail":                       "retail",
    "Auto Manufacturers":                   "auto",
    "Auto Parts":                           "auto",
    "Restaurants":                          "consumer_services",
    "Travel Services":                      "consumer_services",
    "Leisure":                              "consumer_services",
    "Resorts & Casinos":                    "consumer_services",
    "Household & Personal Products":        "consumer_staples",
    "Packaged Foods":                       "consumer_staples",
    "Beverages - Non-Alcoholic":            "consumer_staples",
    "Beverages - Alcoholic":                "consumer_staples",
    "Tobacco":                              "consumer_staples",
    "Farm Products":                        "consumer_staples",

    # Real Estate
    "REIT - Diversified":                   "real_estate",
    "REIT - Industrial":                    "real_estate",
    "REIT - Office":                        "real_estate",
    "REIT - Residential":                   "real_estate",
    "REIT - Retail":                        "real_estate",
    "REIT - Specialty":                     "real_estate",
    "REIT - Healthcare Facilities":         "real_estate",
    "REIT - Hotel & Motel":                 "real_estate",
    "Real Estate Services":                 "real_estate",
    "Real Estate - Development":            "real_estate",

    # Industrials
    "Aerospace & Defense":                  "industrials",
    "Farm & Heavy Construction Machinery":  "industrials",
    "Specialty Industrial Machinery":       "industrials",
    "Electrical Equipment & Parts":         "industrials",
    "Industrial Distribution":              "industrials",
    "Railroads":                            "industrials",
    "Airlines":                             "industrials",
    "Trucking":                             "industrials",
    "Waste Management":                     "industrials",
    "Conglomerates":                        "industrials",

    # Utilities
    "Utilities - Regulated Electric":       "utilities",
    "Utilities - Regulated Gas":            "utilities",
    "Utilities - Diversified":              "utilities",
    "Utilities - Independent Power Producers": "utilities",
    "Utilities - Renewable":                "utilities",

    # Communication / Media
    "Internet Content & Information":       "communication",
    "Entertainment":                        "communication",
    "Electronic Gaming & Multimedia":       "communication",
    "Advertising Agencies":                 "communication",
    "Broadcasting":                         "communication",
    "Telecom Services":                     "telecom",

    # Basic Materials
    "Gold":                                 "materials",
    "Silver":                               "materials",
    "Copper":                               "materials",
    "Steel":                                "materials",
    "Aluminum":                             "materials",
    "Specialty Chemicals":                  "materials",
    "Agricultural Inputs":                  "materials",
    "Building Materials":                   "materials",
    "Lumber & Wood Production":             "materials",
}


# ---------------------------------------------------------------------------
# Broader fallback: yfinance sector string → internal sector key
# ---------------------------------------------------------------------------

SECTOR_TO_SECTOR_KEY: dict[str, str] = {
    "Technology":             "technology_general",
    "Healthcare":             "healthcare_general",
    "Financial Services":     "financials_general",
    "Communication Services": "communication",
    "Consumer Cyclical":      "consumer_cyclical",
    "Consumer Defensive":     "consumer_staples",
    "Energy":                 "energy",
    "Industrials":            "industrials",
    "Real Estate":            "real_estate",
    "Utilities":              "utilities",
    "Basic Materials":        "materials",
}


# ---------------------------------------------------------------------------
# Sector Configurations — specialized analysis frameworks
# ---------------------------------------------------------------------------

SECTOR_CONFIGS: dict[str, SectorConfig] = {

    # ===== SOFTWARE =====

    "software_infra": SectorConfig(
        key="software_infra",
        label="Software Infrastructure",
        analysis_framework="""You are analyzing a SOFTWARE INFRASTRUCTURE company. Apply a cloud/SaaS-native framework:

PRIORITY METRICS (weight heavily):
- Revenue growth rate and acceleration/deceleration trends
- Gross margins (should be 70%+ for pure software; lower suggests hardware/services mix)
- Rule of 40 assessment: revenue growth % + FCF/operating margin % (>40 is strong)
- Operating leverage: are margins expanding as revenue scales?
- Free Cash Flow margins (more relevant than GAAP net income for high-growth software)
- Stock-based compensation as % of revenue (dilution risk)

VALUATION FRAMEWORK:
- EV/Revenue is the primary multiple (P/E is often misleading for growth software)
- Forward revenue growth rate vs. current EV/Revenue multiple
- Compare FCF yield to growth rate for mature names
- High multiples require sustained 25%+ growth to be justified

COMPETITIVE MOAT INDICATORS:
- Platform switching costs and developer ecosystem lock-in
- Infrastructure-layer stickiness (harder to rip out than app-layer)
- Open-source vs. proprietary positioning
- Multi-cloud / hybrid cloud optionality""",
        key_metrics=["revenue_growth", "return_on_equity", "market_cap", "forward_pe"],
        sentiment_context="Focus on cloud spending trends, enterprise IT budgets, AI/infrastructure adoption, multi-cloud dynamics, and developer platform shifts.",
    ),

    "software_app": SectorConfig(
        key="software_app",
        label="Application Software / SaaS",
        analysis_framework="""You are analyzing an APPLICATION SOFTWARE / SaaS company. Apply a recurring-revenue framework:

PRIORITY METRICS:
- Revenue growth rate and consistency (look for acceleration or deceleration)
- Gross margins (70%+ expected for SaaS; lower may indicate services drag)
- Rule of 40: revenue growth % + operating margin % — the key SaaS health metric
- Operating leverage: is the company improving margins as it scales?
- Free Cash Flow generation vs. stock-based compensation burden
- Revenue per employee as an efficiency indicator

VALUATION FRAMEWORK:
- EV/Revenue multiple relative to growth rate (the "growth-adjusted" ratio)
- Path to profitability if currently unprofitable — when does FCF turn positive?
- Compare to SaaS peers at similar growth stages
- P/E becomes relevant only for mature, profitable SaaS (growth <20%)

COMPETITIVE MOAT:
- Category leadership and market share trends
- Workflow embeddedness — is the product mission-critical or nice-to-have?
- Network effects or data advantages
- Vertical specialization vs. horizontal platform play""",
        key_metrics=["revenue_growth", "return_on_equity", "market_cap", "forward_pe"],
        sentiment_context="Focus on enterprise software spending cycles, competitive wins/losses, product-led growth signals, and AI feature monetization.",
    ),

    # ===== SEMICONDUCTORS =====

    "semiconductors": SectorConfig(
        key="semiconductors",
        label="Semiconductors",
        analysis_framework="""You are analyzing a SEMICONDUCTOR company. Apply a cyclical-tech framework:

PRIORITY METRICS:
- Revenue growth IN CONTEXT of the semiconductor cycle (are we in expansion, peak, contraction, or trough?)
- Gross margin trends — the single best indicator of pricing power and product mix
- R&D spending as % of revenue (critical for maintaining technology leadership)
- Capital expenditure intensity and capacity utilization (for fabs)
- Inventory days and inventory trends — rising inventory often signals demand weakness

VALUATION FRAMEWORK:
- P/E normalized for cycle position (low P/E at cycle peak is a value TRAP)
- High P/E at cycle trough may actually be a BUY signal (earnings are depressed)
- EV/EBITDA relative to cycle-adjusted margins
- Compare to historical valuation range across cycles

CYCLICAL AWARENESS (CRITICAL):
- Semiconductor stocks are deeply cyclical. Do NOT treat current earnings as steady-state.
- Evaluate where we are in the demand cycle for this company's end markets
- AI/datacenter demand may be structural (less cyclical) vs. PC/mobile (highly cyclical)
- Inventory correction phases typically last 2-4 quarters

COMPETITIVE MOAT:
- Technology node leadership (process advantage, e.g., TSMC's leading-edge fabs)
- Customer design wins and supply agreement duration
- IP portfolio depth and licensing revenue
- Geopolitical exposure: foundry location, export controls, supply chain concentration""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "return_on_equity"],
        sentiment_context="Focus on chip demand cycles, AI accelerator demand, datacenter buildout, inventory corrections, geopolitical/export control risks, design win announcements, and end-market demand signals.",
    ),

    "semi_equipment": SectorConfig(
        key="semi_equipment",
        label="Semiconductor Equipment",
        analysis_framework="""You are analyzing a SEMICONDUCTOR EQUIPMENT company. Apply a capital-goods-for-tech framework:

PRIORITY METRICS:
- Revenue and bookings trends — bookings are a LEADING indicator, revenue is lagging
- Book-to-bill ratio (>1.0 = growing demand, <1.0 = slowing)
- Gross margins — high margins indicate technology moat and pricing power
- Backlog duration and visibility — how far out is the order book?
- R&D intensity — maintaining equipment technology leadership is expensive

VALUATION FRAMEWORK:
- P/E relative to the WFE (Wafer Fab Equipment) spending cycle
- Equipment stocks often peak before the semiconductor cycle peaks
- Forward P/E based on normalized earnings (avoid using peak/trough earnings)
- Compare EV/EBITDA to historical range

INDUSTRY DYNAMICS:
- Wafer Fab Equipment (WFE) total spending trend
- Customer concentration (few large fabs = concentrated risk)
- Technology transitions (EUV, GAA, advanced packaging) create upgrade cycles
- China exposure and export control compliance risks""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="Focus on WFE spending forecasts, fab construction plans, technology node transitions, China export control developments, and major customer capex announcements.",
    ),

    "hardware_consumer": SectorConfig(
        key="hardware_consumer",
        label="Consumer Hardware / Electronics",
        analysis_framework="""You are analyzing a CONSUMER HARDWARE / ELECTRONICS company. Apply a product-cycle framework:

PRIORITY METRICS:
- Revenue growth by product segment (identify growth drivers vs. mature lines)
- Gross margins and product mix trends — hardware margins are lower than software
- Services/recurring revenue as % of total (higher = more valuable, more predictable)
- Average Selling Price (ASP) trends — are consumers trading up or down?
- Installed base size and growth — drives services and accessory revenue

VALUATION FRAMEWORK:
- P/E is the primary metric for profitable hardware companies
- Separate hardware (lower multiple) from services (higher multiple) in valuation
- FCF yield matters — hardware companies should generate strong cash flow
- Dividend and buyback capacity from cash generation

COMPETITIVE MOAT:
- Ecosystem lock-in (app stores, accessories, interoperability)
- Brand loyalty and pricing power
- Supply chain management excellence
- R&D scale advantages and patent portfolio""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "market_cap"],
        sentiment_context="Focus on product launch cycles, upgrade rates, services revenue growth, supply chain dynamics, and competitive threats from alternative ecosystems.",
    ),

    # ===== HEALTHCARE =====

    "biotech": SectorConfig(
        key="biotech",
        label="Biotechnology",
        analysis_framework="""You are analyzing a BIOTECHNOLOGY company. Apply a pipeline-risk-adjusted framework:

PRIORITY METRICS:
- Cash position and quarterly burn rate — calculate runway in quarters
- Revenue from approved products (if any) — trajectory and market share
- R&D spending level — is it sustainable given cash position?
- Debt levels relative to cash (net cash/debt position)
- Current ratio — near-term liquidity risk

VALUATION FRAMEWORK:
- If PRE-REVENUE: market cap vs. total addressable market (highly speculative)
- If REVENUE-GENERATING: P/S or EV/Revenue relative to growth rate
- Cash per share as a floor valuation
- Binary event risk: FDA decisions create step-function price outcomes
- Sum-of-the-parts for multi-program companies

KEY RISK ASSESSMENT:
- Clinical trial failure risk (Phase 1: ~90% fail, Phase 2: ~70%, Phase 3: ~40%)
- Single-product dependency vs. diversified pipeline
- Dilution risk from future capital raises (common in biotech)
- Patent cliff timing for existing products
- Regulatory pathway complexity (accelerated vs. standard approval)

NOTE: Be especially cautious with biotech. Data quality limitations matter more here
because pipeline value (the main driver) is NOT captured in financial statements.""",
        key_metrics=["revenue", "net_income", "current_ratio", "market_cap", "debt_to_equity"],
        sentiment_context="Focus on clinical trial data readouts, FDA approval/rejection decisions, partnership or licensing deals, patent expiration timelines, and competitive drug development.",
    ),

    "pharma": SectorConfig(
        key="pharma",
        label="Pharmaceuticals",
        analysis_framework="""You are analyzing a PHARMACEUTICAL company. Apply a diversified-pipeline framework:

PRIORITY METRICS:
- Revenue growth and composition by drug/therapeutic area
- R&D pipeline depth — number of programs by phase
- Gross margins (typically 60-80% for pharma)
- Operating margins and SG&A efficiency
- Dividend yield and payout ratio — pharma is often a dividend play

VALUATION FRAMEWORK:
- P/E relative to pharma peers and growth rate
- Patent cliff analysis: when do key drugs lose exclusivity?
- Pipeline optionality value (hard to quantify but critical)
- Dividend yield as a floor valuation support
- FCF yield and capital return capacity

RISK FACTORS:
- Patent cliff concentration (revenue loss when blockbusters go generic)
- Drug pricing / regulatory risk (government negotiation, IRA impact)
- Pipeline failure risk across therapeutic areas
- M&A strategy — acquiring growth vs. organic development""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "return_on_equity"],
        sentiment_context="Focus on patent expiration timelines, drug approval decisions, pricing regulation developments, M&A activity, and pipeline clinical data.",
    ),

    "medtech": SectorConfig(
        key="medtech",
        label="Medical Devices / MedTech",
        analysis_framework="""You are analyzing a MEDICAL DEVICE / MEDTECH company. Apply a procedure-volume framework:

PRIORITY METRICS:
- Organic revenue growth (exclude M&A contributions)
- Gross margins (typically 55-70% for medtech)
- R&D as % of revenue — innovation pipeline health
- Recurring revenue component (consumables, services, software)
- Geographic diversification — emerging market exposure as growth driver

VALUATION FRAMEWORK:
- P/E and EV/EBITDA relative to medtech peers
- Growth premium justified by procedure volume trends
- Margin expansion trajectory (operating leverage)
- M&A integration track record (serial acquirers common in medtech)

COMPETITIVE MOAT:
- Regulatory barriers (FDA clearance/approval as competitive moat)
- Surgeon training and switching costs
- Installed base and recurring consumable revenue
- Patent portfolio and technology differentiation""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "debt_to_equity"],
        sentiment_context="Focus on procedure volume trends, FDA clearances/approvals, hospital capital spending, competitive product launches, and healthcare policy changes.",
    ),

    "healthcare_services": SectorConfig(
        key="healthcare_services",
        label="Healthcare Services",
        analysis_framework="""You are analyzing a HEALTHCARE SERVICES company. Apply a managed-care/services framework:

PRIORITY METRICS:
- Revenue growth driven by membership/enrollment growth and pricing
- Medical cost ratio / medical loss ratio (MLR) — core profitability metric
- Operating margins and administrative cost efficiency
- Membership/customer count trends and retention
- Cash flow conversion — healthcare services typically generate strong FCF

VALUATION FRAMEWORK:
- P/E relative to healthcare services peers
- Growth in covered lives or patient volume
- Margin improvement potential (operating leverage)
- Regulatory risk premium (ACA, Medicare/Medicaid policy changes)""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity"],
        sentiment_context="Focus on healthcare policy changes, enrollment trends, medical cost inflation, regulatory developments, and vertical integration strategies.",
    ),

    # ===== FINANCIAL SERVICES =====

    "banking": SectorConfig(
        key="banking",
        label="Banking",
        analysis_framework="""You are analyzing a BANK. Apply a spread-lending and credit-quality framework:

PRIORITY METRICS:
- Net Interest Margin (NIM) — the core profitability driver for banks
- Loan growth rate — organic balance sheet expansion
- Net charge-off ratio and provision for credit losses — credit quality indicators
- Efficiency ratio (lower is better; <55% is strong, >65% is weak)
- CET1 capital ratio — regulatory capital adequacy (>10% is well-capitalized)
- Return on Tangible Common Equity (ROTCE) — the key bank profitability metric

VALUATION FRAMEWORK:
- Price-to-Tangible Book Value (P/TBV) is the PRIMARY bank valuation metric
- P/E relative to normalized earnings (avoid cycle-peak or trough earnings)
- Dividend yield and capital return capacity (buybacks + dividends)
- Trading at a premium to book suggests market expects sustained high ROTCE

MACRO SENSITIVITY:
- Interest rate environment directly impacts NIM
- Credit cycle position — are we early or late in the credit deterioration cycle?
- Yield curve shape (steeper = better for banks, inverted = compresses NIM)
- Regulatory capital requirements and stress test results

NOTE: Standard metrics like revenue growth and P/E are LESS informative for banks.
Focus on NIM, ROTCE, credit quality, and book value.""",
        key_metrics=["pe_ratio", "eps", "dividend_yield", "return_on_equity", "debt_to_equity"],
        sentiment_context="Focus on interest rate trajectory, credit quality trends, regulatory developments, capital return announcements, and macroeconomic indicators affecting loan demand.",
    ),

    "insurance": SectorConfig(
        key="insurance",
        label="Insurance",
        analysis_framework="""You are analyzing an INSURANCE company. Apply an underwriting-profitability framework:

PRIORITY METRICS:
- Combined ratio (P&C): <100% = underwriting profit (lower is better)
- Premium growth — organic vs. rate-driven
- Investment portfolio yield and returns
- Reserve adequacy — is the company under- or over-reserved?
- Return on Equity — the key insurance profitability metric
- Book value per share growth (insurance is a book-value business)

VALUATION FRAMEWORK:
- Price-to-Book Value is the primary valuation metric
- P/E normalized across underwriting cycles
- Dividend yield and total capital return
- Catastrophe exposure and reinsurance costs""",
        key_metrics=["pe_ratio", "return_on_equity", "dividend_yield", "debt_to_equity"],
        sentiment_context="Focus on catastrophe losses, pricing cycle trends, interest rate impacts on investment income, regulatory capital requirements, and reserve development.",
    ),

    "capital_markets": SectorConfig(
        key="capital_markets",
        label="Capital Markets / Asset Management",
        analysis_framework="""You are analyzing a CAPITAL MARKETS / ASSET MANAGEMENT company. Apply an AUM-and-fee framework:

PRIORITY METRICS:
- Assets Under Management (AUM) growth — market appreciation + net inflows
- Fee rate trends (basis points charged) — under pressure from passive/ETF competition?
- Revenue mix: management fees vs. performance fees vs. transaction fees
- Operating margins — scale advantages in asset management
- Compensation ratio (for broker-dealers and investment banks)

VALUATION FRAMEWORK:
- P/E relative to earnings growth and quality
- AUM multiple (EV/AUM) for pure asset managers
- Dividend yield for mature firms
- Fee rate sustainability risk from passive investing trend""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "dividend_yield"],
        sentiment_context="Focus on market levels (drive AUM), fund flow trends, fee compression, regulatory changes, and competitive dynamics between active and passive management.",
    ),

    "fintech": SectorConfig(
        key="fintech",
        label="Fintech / Payments",
        analysis_framework="""You are analyzing a FINTECH / PAYMENTS company. Apply a transaction-volume framework:

PRIORITY METRICS:
- Total Payment Volume (TPV) or transaction volume growth
- Revenue growth and take rate trends (revenue as % of volume processed)
- Gross margins — should be high for pure payment processors
- Operating leverage — are margins expanding with scale?
- Active accounts / customers growth and engagement metrics
- Cross-selling effectiveness (multiple products per customer)

VALUATION FRAMEWORK:
- EV/Revenue or P/E for profitable fintechs
- Growth-adjusted P/E (PEG ratio)
- Compare take rates to peers — are they sustainable?
- Regulatory risk premium for lending fintechs vs. pure payments""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="Focus on digital payment adoption trends, regulatory developments, competitive dynamics, cross-border payment volumes, and crypto/blockchain impacts.",
    ),

    # ===== ENERGY =====

    "energy": SectorConfig(
        key="energy",
        label="Energy / Oil & Gas",
        analysis_framework="""You are analyzing an ENERGY / OIL & GAS company. Apply a commodity-cycle framework:

PRIORITY METRICS:
- Production volumes and growth trajectory
- Reserve replacement ratio (>100% means growing reserves)
- Break-even oil/gas price — at what commodity price is the company profitable?
- Free Cash Flow yield — energy companies should return cash at high commodity prices
- Debt-to-EBITDA — leverage is dangerous in a cyclical commodity business
- Capital discipline — reinvestment rate vs. shareholder returns

VALUATION FRAMEWORK:
- EV/EBITDA is the primary metric (normalize for commodity prices)
- FCF yield relative to peers (should be high given commodity risk premium)
- P/E is LESS useful due to commodity price sensitivity
- Dividend yield + buyback yield = total shareholder return
- NAV based on reserve valuation (for E&P companies)

MACRO SENSITIVITY:
- Current oil/gas prices vs. historical and forward curves
- OPEC+ supply decisions and geopolitical supply risk
- Energy transition risk — long-term demand outlook
- Regulatory / carbon policy exposure""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "dividend_yield"],
        sentiment_context="Focus on oil/gas prices, OPEC+ decisions, geopolitical supply risks, energy transition policy, capital allocation decisions, and production guidance.",
    ),

    "renewables": SectorConfig(
        key="renewables",
        label="Renewable Energy",
        analysis_framework="""You are analyzing a RENEWABLE ENERGY company. Apply a project-pipeline framework:

PRIORITY METRICS:
- Installed capacity (MW/GW) and growth trajectory
- Revenue growth from new project completions
- Power Purchase Agreement (PPA) pricing and duration
- Levelized Cost of Energy (LCOE) competitiveness
- Debt levels — renewable projects are capital-intensive
- Government incentive exposure (ITC, PTC, IRA credits)

VALUATION FRAMEWORK:
- EV/EBITDA relative to contracted cash flow visibility
- Project pipeline value and development stage
- DCF based on long-term contracted revenue
- Policy risk premium for subsidy-dependent companies""",
        key_metrics=["revenue_growth", "debt_to_equity", "market_cap"],
        sentiment_context="Focus on government energy policy, tax credit extensions/expirations, utility-scale procurement, grid interconnection timelines, and technology cost curves.",
    ),

    # ===== CONSUMER =====

    "ecommerce": SectorConfig(
        key="ecommerce",
        label="E-Commerce / Internet Retail",
        analysis_framework="""You are analyzing an E-COMMERCE / INTERNET RETAIL company. Apply a GMV-and-take-rate framework:

PRIORITY METRICS:
- Gross Merchandise Value (GMV) or total revenue growth
- Take rate / monetization rate trends
- Gross margins — especially 1P vs. 3P marketplace mix
- Advertising and services revenue as % of total (high-margin revenue streams)
- Fulfillment cost trends and logistics efficiency
- Active buyers / customers and purchase frequency

VALUATION FRAMEWORK:
- EV/Revenue for high-growth phase, P/E for mature e-commerce
- Segment-level profitability (separate retail from cloud/ads/services)
- FCF generation capacity as growth investment moderates
- Compare unit economics to physical retail peers""",
        key_metrics=["revenue_growth", "pe_ratio", "forward_pe", "market_cap"],
        sentiment_context="Focus on consumer spending trends, online penetration growth, competitive dynamics, logistics/fulfillment developments, and advertising market conditions.",
    ),

    "retail": SectorConfig(
        key="retail",
        label="Retail",
        analysis_framework="""You are analyzing a RETAIL company. Apply a same-store-sales framework:

PRIORITY METRICS:
- Comparable / same-store sales growth (the single most important retail metric)
- Revenue growth: same-store vs. new store contribution
- Gross margins and inventory management efficiency
- Inventory turnover — rising inventory without revenue growth is a red flag
- SG&A as % of revenue — operating efficiency
- E-commerce penetration and omnichannel capabilities

VALUATION FRAMEWORK:
- P/E relative to retail peers and growth rate
- EV/EBITDA for comparison across capital structures
- Dividend yield for mature retailers
- Same-store sales trajectory drives multiple expansion/contraction""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "current_ratio"],
        sentiment_context="Focus on consumer spending trends, foot traffic data, promotional activity, inventory levels, e-commerce competition, and seasonal patterns.",
    ),

    "consumer_staples": SectorConfig(
        key="consumer_staples",
        label="Consumer Staples / Defensive",
        analysis_framework="""You are analyzing a CONSUMER STAPLES / DEFENSIVE company. Apply a brand-strength-and-pricing framework:

PRIORITY METRICS:
- Organic revenue growth (separate pricing from volume)
- Gross margins — pricing power indicator, input cost sensitivity
- Volume trends — are consumers trading down to private label?
- Operating margins and cost efficiency improvements
- Dividend yield and payout ratio — staples are dividend-paying stocks
- FCF conversion — stable businesses should convert earnings to cash reliably

VALUATION FRAMEWORK:
- P/E is the primary metric (these are stable, profitable companies)
- Dividend yield as a floor valuation support
- Premium P/E justified by recession-resistance and predictability
- Compare to historical valuation range and sector average""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "return_on_equity"],
        sentiment_context="Focus on consumer spending patterns, input cost inflation, private-label competition, pricing action announcements, and recession indicators.",
    ),

    "auto": SectorConfig(
        key="auto",
        label="Automotive",
        analysis_framework="""You are analyzing an AUTOMOTIVE company. Apply a unit-economics framework:

PRIORITY METRICS:
- Vehicle deliveries / unit sales and growth trajectory
- Average Selling Price (ASP) trends — mix shift toward higher-margin vehicles?
- Gross margins per vehicle (automotive gross margins are thin: 15-25%)
- R&D spending — especially for EV and autonomous driving investment
- CapEx intensity and factory utilization rates
- Inventory levels (days of supply) — too high signals demand weakness

VALUATION FRAMEWORK:
- P/E for profitable automakers (use forward earnings, not trailing)
- EV/Revenue for pre-profit or high-growth EV companies
- Delivery growth rate vs. valuation premium
- Compare gross margins to legacy and EV peers""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "debt_to_equity"],
        sentiment_context="Focus on vehicle delivery numbers, EV adoption trends, battery technology developments, autonomous driving progress, tariff/trade impacts, and consumer financing conditions.",
    ),

    "consumer_services": SectorConfig(
        key="consumer_services",
        label="Consumer Services",
        analysis_framework="""You are analyzing a CONSUMER SERVICES company. Apply a unit-economics framework:

PRIORITY METRICS:
- Revenue growth driven by transactions, bookings, or active users
- Gross margins and contribution margins per transaction
- Customer acquisition cost trends and payback periods
- Active users/customers and engagement metrics (repeat rate)
- Seasonality patterns and geographic diversification
- Operating leverage — are margins expanding with scale?

VALUATION FRAMEWORK:
- P/E or EV/EBITDA for profitable services companies
- EV/Revenue for high-growth, pre-profit companies
- Compare unit economics to peers in the category""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="Focus on consumer spending trends, travel/leisure demand, competitive dynamics, and macroeconomic indicators affecting discretionary spending.",
    ),

    "consumer_cyclical": SectorConfig(
        key="consumer_cyclical",
        label="Consumer Cyclical",
        analysis_framework="""You are analyzing a CONSUMER CYCLICAL company. Apply a cycle-and-demand framework:

PRIORITY METRICS:
- Revenue growth relative to economic cycle position
- Same-store sales or comparable metrics where applicable
- Gross margins and pricing power in different economic conditions
- Inventory management efficiency
- Balance sheet strength to weather downturns
- Consumer confidence and spending trend sensitivity

VALUATION FRAMEWORK:
- P/E normalized for cycle position
- Be cautious of low P/E at cycle peaks (cyclical trap)
- EV/EBITDA relative to peers at similar cycle points
- Dividend sustainability through full economic cycles""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "debt_to_equity", "current_ratio"],
        sentiment_context="Focus on consumer confidence, discretionary spending trends, economic cycle indicators, and competitive positioning.",
    ),

    # ===== REAL ESTATE =====

    "real_estate": SectorConfig(
        key="real_estate",
        label="Real Estate / REITs",
        analysis_framework="""You are analyzing a REAL ESTATE / REIT company. Apply a NAV-and-FFO framework:

PRIORITY METRICS:
- Funds From Operations (FFO) per share and growth — the REIT equivalent of earnings
- Adjusted FFO (AFFO) for recurring capex deductions
- Occupancy rate and tenant retention
- Same-property Net Operating Income (NOI) growth
- Weighted Average Lease Expiry (WALE) — longer = more predictable
- Debt-to-EBITDA and interest coverage — REITs use significant leverage
- Dividend yield and AFFO payout ratio

VALUATION FRAMEWORK:
- Price-to-FFO (the REIT P/E equivalent)
- NAV discount/premium — is the stock above or below asset value?
- Cap rate implied by current pricing vs. market cap rates
- Dividend yield relative to REIT peers and interest rates
- NOTE: Traditional P/E is MISLEADING for REITs due to depreciation

RATE SENSITIVITY:
- Higher interest rates increase borrowing costs and compress valuations
- Rising rates also increase competition from fixed-income alternatives
- Property type matters: industrial/data center REITs have secular tailwinds""",
        key_metrics=["pe_ratio", "dividend_yield", "debt_to_equity", "market_cap"],
        sentiment_context="Focus on interest rate trajectory, property type demand trends, occupancy rates, construction pipeline, and cap rate movements.",
    ),

    # ===== UTILITIES =====

    "utilities": SectorConfig(
        key="utilities",
        label="Utilities",
        analysis_framework="""You are analyzing a UTILITY company. Apply a regulated-returns framework:

PRIORITY METRICS:
- Rate base growth — the primary driver of earnings growth for regulated utilities
- Allowed Return on Equity (ROE) from regulators
- Earned ROE vs. allowed ROE — execution quality indicator
- Capital expenditure plan and rate case pipeline
- Dividend yield and payout ratio — utilities are primarily income investments
- Debt-to-equity — utilities use significant but manageable leverage
- Regulatory jurisdiction quality — constructive vs. challenging regulators

VALUATION FRAMEWORK:
- P/E relative to utility peers and allowed ROE
- Dividend yield as primary valuation anchor
- Rate base growth rate = approximate earnings growth rate
- Premium P/E justified by constructive regulation and growth capex

SPECIAL CONSIDERATIONS:
- AI/datacenter power demand as a growth catalyst
- Renewable energy transition capex opportunities
- Wildfire / extreme weather liability (for certain geographies)""",
        key_metrics=["pe_ratio", "eps", "dividend_yield", "debt_to_equity"],
        sentiment_context="Focus on rate case decisions, regulatory developments, weather events, AI datacenter power demand, renewable energy mandates, and interest rate sensitivity.",
    ),

    # ===== INDUSTRIALS =====

    "industrials": SectorConfig(
        key="industrials",
        label="Industrials",
        analysis_framework="""You are analyzing an INDUSTRIAL company. Apply a backlog-and-cycle framework:

PRIORITY METRICS:
- Organic revenue growth (separate from acquisitions and FX)
- Order backlog and book-to-bill ratio — leading indicators of future revenue
- Gross and operating margins — indicator of pricing power and operational efficiency
- Free Cash Flow conversion — industrials should convert earnings to cash reliably
- CapEx intensity relative to depreciation — is the company investing for growth?
- End-market diversification — reduces cyclical risk

VALUATION FRAMEWORK:
- P/E and EV/EBITDA relative to industrial peers
- Through-cycle earnings normalization (avoid valuing at peak/trough margins)
- FCF yield as a quality check
- M&A strategy and returns on acquired capital

CYCLICAL AWARENESS:
- Industrial stocks correlate with PMI and capex cycles
- Infrastructure spending and government policy tailwinds
- Supply chain normalization and input cost trends""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity", "debt_to_equity"],
        sentiment_context="Focus on PMI trends, infrastructure spending, supply chain dynamics, input costs, order/backlog trends, and government policy impacts.",
    ),

    # ===== COMMUNICATION / MEDIA =====

    "communication": SectorConfig(
        key="communication",
        label="Communication / Media",
        analysis_framework="""You are analyzing a COMMUNICATION / MEDIA company. Apply a user-engagement framework:

PRIORITY METRICS:
- User / subscriber metrics: DAU, MAU, subscribers, and growth trends
- Average Revenue Per User (ARPU) — monetization effectiveness
- Advertising revenue growth and pricing trends
- Content spending and ROI (for media companies)
- Engagement metrics: time spent, sessions, retention rates
- Operating margins and path to profitability for newer platforms

VALUATION FRAMEWORK:
- EV/Revenue or EV/User for high-growth platforms
- P/E for profitable, mature media companies
- Advertising revenue per user vs. peers
- Content asset amortization patterns""",
        key_metrics=["revenue_growth", "pe_ratio", "market_cap", "return_on_equity"],
        sentiment_context="Focus on user growth trends, advertising market conditions, content spending, regulatory scrutiny, competitive platform dynamics, and AI content impacts.",
    ),

    "telecom": SectorConfig(
        key="telecom",
        label="Telecommunications",
        analysis_framework="""You are analyzing a TELECOM company. Apply a subscriber-and-infrastructure framework:

PRIORITY METRICS:
- Subscriber count and net additions (wireless and broadband)
- ARPU trends — pricing power vs. competitive pressure
- Churn rate — lower is better, indicates customer satisfaction
- CapEx intensity (5G, fiber buildout) and network quality
- EBITDA margins and FCF generation — telecom is capital-intensive
- Debt levels — telecom companies carry significant leverage
- Dividend yield and sustainability

VALUATION FRAMEWORK:
- EV/EBITDA is the primary telecom valuation metric
- Dividend yield as an income anchor
- FCF yield after capex (determines dividend sustainability)
- Spectrum asset value and network infrastructure replacement cost""",
        key_metrics=["revenue_growth", "pe_ratio", "dividend_yield", "debt_to_equity"],
        sentiment_context="Focus on subscriber trends, 5G deployment, broadband competition, spectrum auctions, pricing dynamics, and regulatory developments.",
    ),

    # ===== MATERIALS =====

    "materials": SectorConfig(
        key="materials",
        label="Basic Materials",
        analysis_framework="""You are analyzing a BASIC MATERIALS company. Apply a commodity-and-cost-curve framework:

PRIORITY METRICS:
- Revenue and volume trends relative to commodity prices
- Cost position on the industry cost curve (low-cost producers are best)
- Gross margins — highly sensitive to commodity price movements
- Cash operating costs per unit (per ounce, per ton, etc.)
- Reserve life and resource quality (for mining)
- CapEx requirements for maintenance and growth

VALUATION FRAMEWORK:
- P/E normalized for commodity cycle position
- EV/EBITDA at various commodity price scenarios
- Price-to-NAV based on reserve valuation (mining/metals)
- FCF yield at current commodity prices
- Dividend sustainability across price cycles""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "dividend_yield"],
        sentiment_context="Focus on commodity price movements, supply-demand dynamics, mine development, trade/tariff policies, and environmental regulatory developments.",
    ),

    # ===== GENERAL FALLBACKS =====

    "technology_general": SectorConfig(
        key="technology_general",
        label="Technology",
        analysis_framework="""You are analyzing a TECHNOLOGY company. Apply a growth-and-moat framework:

PRIORITY METRICS:
- Revenue growth rate and consistency
- Gross margins (high margins suggest software/IP value; lower suggests hardware/services)
- R&D spending as % of revenue — investment in future competitiveness
- Operating leverage — are margins expanding with revenue growth?
- Free Cash Flow margins and conversion quality
- Market position and competitive dynamics

VALUATION FRAMEWORK:
- For high-growth (>20%): EV/Revenue is primary, P/E secondary
- For mature growth (<20%): P/E and PEG ratio are primary
- FCF yield for cash-generative tech
- Compare multiple to growth rate (rule of thumb: P/E ≈ growth rate for fair value)""",
        key_metrics=["revenue_growth", "pe_ratio", "forward_pe", "return_on_equity", "market_cap"],
        sentiment_context="Focus on technology adoption trends, competitive dynamics, regulatory developments, and macroeconomic impacts on tech spending.",
    ),

    "healthcare_general": SectorConfig(
        key="healthcare_general",
        label="Healthcare",
        analysis_framework="""You are analyzing a HEALTHCARE company. Apply a diversified-healthcare framework:

PRIORITY METRICS:
- Revenue growth by segment
- Gross margins (vary widely: 40% for devices to 80% for drugs)
- R&D spending and pipeline productivity
- Regulatory risk exposure
- Patent protection and exclusivity timelines
- Dividend yield for mature healthcare companies

VALUATION FRAMEWORK:
- P/E relative to healthcare sector peers
- Pipeline value for R&D-intensive companies
- Dividend yield for mature, diversified healthcare""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity", "dividend_yield"],
        sentiment_context="Focus on regulatory developments, drug pricing policy, clinical trial results, and healthcare spending trends.",
    ),

    "financials_general": SectorConfig(
        key="financials_general",
        label="Financial Services",
        analysis_framework="""You are analyzing a FINANCIAL SERVICES company. Apply a returns-and-risk framework:

PRIORITY METRICS:
- Return on Equity (ROE) or Return on Tangible Common Equity (ROTCE) — the key metric
- Revenue growth and composition (fee vs. spread income)
- Efficiency ratio or cost-to-income ratio
- Capital adequacy and regulatory capital ratios
- Credit quality metrics if applicable (charge-offs, NPLs)
- Dividend yield and total capital return

VALUATION FRAMEWORK:
- Price-to-Book Value or P/TBV for capital-heavy financials
- P/E for fee-based financials (asset managers, exchanges)
- Dividend yield as income component
- ROE above cost of equity justifies premium to book value""",
        key_metrics=["pe_ratio", "return_on_equity", "dividend_yield", "debt_to_equity"],
        sentiment_context="Focus on interest rate environment, credit conditions, regulatory changes, capital markets activity, and macroeconomic indicators.",
    ),

    # ===== DEFAULT (catch-all) =====

    "default": SectorConfig(
        key="default",
        label="General",
        analysis_framework="""Apply a general fundamental analysis framework:

PRIORITY METRICS:
- Revenue growth trends and consistency
- Profitability: gross margins, operating margins, net margins, ROE
- Balance sheet strength: debt levels, current ratio, interest coverage
- Valuation: P/E ratio relative to growth rate (PEG ratio)
- Cash flow: FCF generation and quality of earnings
- Competitive position and moat durability

VALUATION FRAMEWORK:
- P/E relative to sector average and historical range
- PEG ratio for growth-adjusted valuation
- EV/EBITDA for cross-capital-structure comparison
- Dividend yield and payout sustainability (if applicable)
- FCF yield as a quality metric""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "return_on_equity", "dividend_yield"],
        sentiment_context="Focus on general market sentiment, competitive dynamics, and macroeconomic factors affecting this company.",
    ),
}


# ---------------------------------------------------------------------------
# Resolution function
# ---------------------------------------------------------------------------

def get_sector_config(industry: str | None, sector: str | None) -> SectorConfig:
    """Resolve the best-matching SectorConfig for a given industry/sector pair.

    Resolution order:
      1. Exact industry string match → specific sector config
      2. Broad yfinance sector string → general sector config
      3. Default catch-all config
    """
    # Try fine-grained industry match first
    if industry:
        key = INDUSTRY_TO_SECTOR_KEY.get(industry)
        if key and key in SECTOR_CONFIGS:
            return SECTOR_CONFIGS[key]

    # Fall back to broad sector match
    if sector:
        key = SECTOR_TO_SECTOR_KEY.get(sector)
        if key and key in SECTOR_CONFIGS:
            return SECTOR_CONFIGS[key]

    return SECTOR_CONFIGS["default"]
