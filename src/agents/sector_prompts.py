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
        label="软件基础设施",
        analysis_framework="""你正在分析一家软件基础设施公司。请运用云计算/SaaS原生分析框架：

优先指标（重点权衡）：
- 营收增长率及加速/减速趋势
- 毛利率（纯软件公司应在70%以上；较低则表明存在硬件/服务混合）
- Rule of 40 评估：营收增长率% + FCF/营业利润率%（>40为强劲）
- 经营杠杆：随着营收规模扩大，利润率是否在扩张？
- FCF利润率（对高增长软件公司而言比GAAP净利润更具参考价值）
- 股权激励占营收百分比（稀释风险）

估值框架：
- EV/Revenue 是主要估值倍数（P/E 对增长型软件公司往往具有误导性）
- 前瞻营收增长率与当前 EV/Revenue 倍数的对比
- 对成熟标的比较 FCF 收益率与增长率
- 高估值倍数需要持续25%以上的增长率来支撑

竞争护城河指标：
- 平台切换成本和开发者生态系统锁定效应
- 基础设施层的粘性（比应用层更难被替换）
- 开源与专有定位
- 多云/混合云可选性""",
        key_metrics=["revenue_growth", "return_on_equity", "market_cap", "forward_pe"],
        sentiment_context="关注云计算支出趋势、企业IT预算、AI/基础设施采用率、多云动态以及开发者平台变化。",
    ),

    "software_app": SectorConfig(
        key="software_app",
        label="应用软件 / SaaS",
        analysis_framework="""你正在分析一家应用软件/SaaS公司。请运用经常性收入分析框架：

优先指标：
- 营收增长率及一致性（关注加速或减速趋势）
- 毛利率（SaaS公司预期70%以上；较低可能表明服务业务拖累）
- Rule of 40：营收增长率% + 营业利润率% — SaaS健康度的关键指标
- 经营杠杆：公司在规模扩张的同时利润率是否在改善？
- FCF产生能力与股权激励负担的对比
- 人均营收作为效率指标

估值框架：
- EV/Revenue 倍数相对于增长率（"增长调整后"比率）
- 若当前未盈利——FCF何时转正？盈利路径分析
- 与相似增长阶段的SaaS同行进行比较
- P/E 仅在成熟、盈利的SaaS公司（增长率<20%）中才具参考价值

竞争护城河：
- 品类领导地位和市场份额趋势
- 工作流嵌入程度——产品是关键任务级别还是锦上添花？
- 网络效应或数据优势
- 垂直专业化与水平平台化策略""",
        key_metrics=["revenue_growth", "return_on_equity", "market_cap", "forward_pe"],
        sentiment_context="关注企业软件支出周期、竞争中的客户得失、产品驱动增长信号以及AI功能货币化。",
    ),

    # ===== SEMICONDUCTORS =====

    "semiconductors": SectorConfig(
        key="semiconductors",
        label="半导体",
        analysis_framework="""你正在分析一家半导体公司。请运用周期性科技分析框架：

优先指标：
- 结合半导体周期背景的营收增长（当前处于扩张期、峰值期、收缩期还是低谷期？）
- 毛利率趋势——定价能力和产品组合的最佳单一指标
- R&D支出占营收百分比（维持技术领先地位的关键）
- 资本支出强度和产能利用率（针对晶圆厂）
- 库存天数和库存趋势——库存上升往往预示需求疲软

估值框架：
- 根据周期位置进行正常化的 P/E（周期峰值时的低P/E是价值陷阱）
- 周期低谷时的高P/E实际上可能是买入信号（盈利被压低）
- EV/EBITDA 相对于周期调整后的利润率
- 与历史跨周期估值区间进行比较

周期性意识（至关重要）：
- 半导体股票具有深度周期性。不要将当前盈利视为稳态水平。
- 评估该公司终端市场的需求周期所处阶段
- AI/数据中心需求可能是结构性的（周期性较弱），而PC/移动端（高度周期性）
- 库存修正阶段通常持续2-4个季度

竞争护城河：
- 技术节点领先（工艺优势，例如台积电的先进制程晶圆厂）
- 客户设计导入和供应协议期限
- 知识产权组合深度和许可收入
- 地缘政治风险敞口：晶圆厂所在地、出口管制、供应链集中度""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "return_on_equity"],
        sentiment_context="关注芯片需求周期、AI加速器需求、数据中心建设、库存修正、地缘政治/出口管制风险、设计导入公告以及终端市场需求信号。",
    ),

    "semi_equipment": SectorConfig(
        key="semi_equipment",
        label="半导体设备",
        analysis_framework="""你正在分析一家半导体设备公司。请运用科技领域资本品分析框架：

优先指标：
- 营收和订单趋势——订单是领先指标，营收是滞后指标
- 订单出货比（>1.0 = 需求增长，<1.0 = 需求放缓）
- 毛利率——高利润率表明技术护城河和定价能力
- 积压订单持续时间和可见性——订单簿能覆盖多远的未来？
- R&D强度——维持设备技术领先地位成本高昂

估值框架：
- P/E 相对于WFE（晶圆厂设备）支出周期
- 设备股通常在半导体周期见顶之前就已触顶
- 基于正常化盈利的前瞻 P/E（避免使用峰值/低谷盈利）
- 将 EV/EBITDA 与历史区间进行比较

行业动态：
- WFE总支出趋势
- 客户集中度（大型晶圆厂数量少 = 风险集中）
- 技术迭代（EUV、GAA、先进封装）创造升级周期
- 中国风险敞口和出口管制合规风险""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="关注WFE支出预测、晶圆厂建设计划、技术节点迭代、中国出口管制动态以及主要客户资本支出公告。",
    ),

    "hardware_consumer": SectorConfig(
        key="hardware_consumer",
        label="消费硬件/电子产品",
        analysis_framework="""你正在分析一家消费硬件/电子产品公司。请运用产品周期分析框架：

优先指标：
- 按产品线划分的营收增长（识别增长驱动力与成熟产品线）
- 毛利率和产品组合趋势——硬件利润率低于软件
- 服务/经常性收入占总收入的百分比（占比越高越有价值，可预测性越强）
- 平均售价（ASP）趋势——消费者是在升级消费还是降级消费？
- 装机量规模和增长——驱动服务和配件收入

估值框架：
- P/E 是盈利硬件公司的主要指标
- 在估值中将硬件（较低倍数）与服务（较高倍数）分开
- FCF收益率很重要——硬件公司应产生强劲的现金流
- 现金产生能力所支撑的股息和回购空间

竞争护城河：
- 生态系统锁定（应用商店、配件、互操作性）
- 品牌忠诚度和定价能力
- 供应链管理卓越能力
- R&D规模优势和专利组合""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "market_cap"],
        sentiment_context="关注产品发布周期、升级率、服务收入增长、供应链动态以及来自替代生态系统的竞争威胁。",
    ),

    # ===== HEALTHCARE =====

    "biotech": SectorConfig(
        key="biotech",
        label="生物科技",
        analysis_framework="""你正在分析一家生物科技公司。请运用管线风险调整分析框架：

优先指标：
- 现金储备和季度消耗率——以季度为单位计算资金跑道
- 已获批产品的收入（如有）——增长轨迹和市场份额
- R&D支出水平——相对于现金储备是否可持续？
- 相对于现金的负债水平（净现金/净负债状况）
- 流动比率——近期流动性风险

估值框架：
- 若为临床前/未有收入阶段：市值与总可及市场规模对比（高度投机性）
- 若已产生收入：P/S 或 EV/Revenue 相对于增长率
- 每股现金作为估值底部
- 二元事件风险：FDA决策产生阶梯式价格变动
- 多管线公司的分部加总估值

关键风险评估：
- 临床试验失败风险（I期：约90%失败，II期：约70%，III期：约40%）
- 单一产品依赖与多元化管线
- 未来融资导致的稀释风险（生物科技中常见）
- 现有产品的专利悬崖时间节点
- 监管审批路径复杂性（加速审批与标准审批）

注意：对生物科技公司需格外谨慎。数据质量的局限性在此更为重要，
因为管线价值（主要驱动因素）无法在财务报表中体现。""",
        key_metrics=["revenue", "net_income", "current_ratio", "market_cap", "debt_to_equity"],
        sentiment_context="关注临床试验数据公布、FDA批准/拒绝决定、合作或授权交易、专利到期时间线以及竞争性药物研发动态。",
    ),

    "pharma": SectorConfig(
        key="pharma",
        label="制药",
        analysis_framework="""你正在分析一家制药公司。请运用多元化管线分析框架：

优先指标：
- 按药品/治疗领域划分的营收增长和构成
- R&D管线深度——按临床阶段划分的项目数量
- 毛利率（制药公司通常为60-80%）
- 营业利润率和销售管理费用效率
- 股息收益率和派息率——制药常作为股息投资标的

估值框架：
- P/E 相对于制药同行和增长率
- 专利悬崖分析：关键药品何时失去专利保护？
- 管线期权价值（难以量化但至关重要）
- 股息收益率作为估值底部支撑
- FCF收益率和资本回报能力

风险因素：
- 专利悬崖集中度（重磅药物仿制化时的收入损失）
- 药品定价/监管风险（政府议价、IRA影响）
- 各治疗领域的管线失败风险
- 并购策略——收购增长与有机发展""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "return_on_equity"],
        sentiment_context="关注专利到期时间线、药品审批决定、定价监管动态、并购活动以及管线临床数据。",
    ),

    "medtech": SectorConfig(
        key="medtech",
        label="医疗器械/医疗科技",
        analysis_framework="""你正在分析一家医疗器械/医疗科技公司。请运用手术量驱动分析框架：

优先指标：
- 有机营收增长（剔除并购贡献）
- 毛利率（医疗科技通常为55-70%）
- R&D占营收百分比——创新管线健康度
- 经常性收入组成部分（耗材、服务、软件）
- 地域多元化——新兴市场敞口作为增长驱动力

估值框架：
- P/E 和 EV/EBITDA 相对于医疗科技同行
- 手术量趋势支撑的增长溢价
- 利润率扩张轨迹（经营杠杆）
- 并购整合记录（医疗科技领域的连续收购者常见）

竞争护城河：
- 监管壁垒（FDA准入/批准作为竞争护城河）
- 外科医生培训和切换成本
- 装机基础和经常性耗材收入
- 专利组合和技术差异化""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "debt_to_equity"],
        sentiment_context="关注手术量趋势、FDA准入/批准、医院资本支出、竞争性产品发布以及医疗保健政策变化。",
    ),

    "healthcare_services": SectorConfig(
        key="healthcare_services",
        label="医疗保健服务",
        analysis_framework="""你正在分析一家医疗保健服务公司。请运用管理式医疗/服务分析框架：

优先指标：
- 由会员/参保人数增长和定价驱动的营收增长
- 医疗成本率/医疗赔付率（MLR）——核心盈利能力指标
- 营业利润率和管理费用效率
- 会员/客户数量趋势和留存率
- 现金流转化——医疗保健服务通常产生强劲的FCF

估值框架：
- P/E 相对于医疗保健服务同行
- 参保人数或患者量的增长
- 利润率改善潜力（经营杠杆）
- 监管风险溢价（ACA、Medicare/Medicaid政策变化）""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity"],
        sentiment_context="关注医疗保健政策变化、参保趋势、医疗成本通胀、监管动态以及纵向整合策略。",
    ),

    # ===== FINANCIAL SERVICES =====

    "banking": SectorConfig(
        key="banking",
        label="银行业",
        analysis_framework="""你正在分析一家银行。请运用利差贷款和信贷质量分析框架：

优先指标：
- 净息差（NIM）——银行核心盈利能力驱动因素
- 贷款增长率——资产负债表的有机扩张
- 净冲销率和信贷损失拨备——信贷质量指标
- 效率比率（越低越好；<55%为强劲，>65%为疲弱）
- CET1资本比率——监管资本充足率（>10%为资本充足）
- 有形普通股权益回报率（ROTCE）——银行盈利能力的关键指标

估值框架：
- 有形账面价值比（P/TBV）是银行估值的首要指标
- P/E 相对于正常化盈利（避免使用周期峰值或低谷盈利）
- 股息收益率和资本回报能力（回购+股息）
- 溢价交易（高于账面价值）表明市场预期持续的高ROTCE

宏观敏感性：
- 利率环境直接影响NIM
- 信贷周期所处阶段——我们处于信贷恶化周期的早期还是晚期？
- 收益率曲线形态（更陡 = 对银行更有利，倒挂 = 压缩NIM）
- 监管资本要求和压力测试结果

注意：对银行而言，营收增长和P/E等标准指标参考价值较低。
重点关注NIM、ROTCE、信贷质量和账面价值。""",
        key_metrics=["pe_ratio", "eps", "dividend_yield", "return_on_equity", "debt_to_equity"],
        sentiment_context="关注利率走势、信贷质量趋势、监管动态、资本回报公告以及影响贷款需求的宏观经济指标。",
    ),

    "insurance": SectorConfig(
        key="insurance",
        label="保险业",
        analysis_framework="""你正在分析一家保险公司。请运用承保盈利能力分析框架：

优先指标：
- 综合成本率（财产险/意外险）：<100% = 承保盈利（越低越好）
- 保费增长——有机增长与费率驱动增长
- 投资组合收益率和回报
- 准备金充足性——公司是准备金不足还是超额计提？
- ROE——保险盈利能力的关键指标
- 每股账面价值增长（保险是账面价值驱动的业务）

估值框架：
- 市净率（P/B）是主要估值指标
- 跨承保周期正常化的 P/E
- 股息收益率和总资本回报
- 巨灾风险敞口和再保险成本""",
        key_metrics=["pe_ratio", "return_on_equity", "dividend_yield", "debt_to_equity"],
        sentiment_context="关注巨灾损失、定价周期趋势、利率对投资收益的影响、监管资本要求以及准备金变动。",
    ),

    "capital_markets": SectorConfig(
        key="capital_markets",
        label="资本市场/资产管理",
        analysis_framework="""你正在分析一家资本市场/资产管理公司。请运用AUM和费率分析框架：

优先指标：
- 管理资产规模（AUM）增长——市场增值+净流入
- 费率趋势（收取的基点数）——是否受到被动投资/ETF竞争的压力？
- 收入构成：管理费 vs. 业绩报酬 vs. 交易费
- 营业利润率——资产管理的规模优势
- 薪酬比率（针对经纪交易商和投资银行）

估值框架：
- P/E 相对于盈利增长和质量
- AUM倍数（EV/AUM）适用于纯资产管理公司
- 成熟公司的股息收益率
- 被动投资趋势下的费率可持续性风险""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "dividend_yield"],
        sentiment_context="关注市场水平（驱动AUM）、资金流向趋势、费率压缩、监管变化以及主动管理与被动管理之间的竞争动态。",
    ),

    "fintech": SectorConfig(
        key="fintech",
        label="金融科技/支付",
        analysis_framework="""你正在分析一家金融科技/支付公司。请运用交易量驱动分析框架：

优先指标：
- 总支付金额（TPV）或交易量增长
- 营收增长和费率趋势（营收占处理交易量的百分比）
- 毛利率——纯支付处理商应有较高的毛利率
- 经营杠杆——利润率是否随规模扩张而提高？
- 活跃账户/客户增长和参与度指标
- 交叉销售效果（每客户多产品数量）

估值框架：
- 盈利的金融科技公司使用 EV/Revenue 或 P/E
- 增长调整后的 P/E（PEG比率）
- 与同行比较费率——是否可持续？
- 贷款类金融科技与纯支付类的监管风险溢价""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="关注数字支付普及趋势、监管动态、竞争格局、跨境支付量以及加密货币/区块链的影响。",
    ),

    # ===== ENERGY =====

    "energy": SectorConfig(
        key="energy",
        label="能源/石油天然气",
        analysis_framework="""你正在分析一家能源/石油天然气公司。请运用大宗商品周期分析框架：

优先指标：
- 产量和增长轨迹
- 储量替代率（>100%意味着储量在增长）
- 盈亏平衡油/气价格——在何种商品价格下公司能实现盈利？
- FCF收益率——能源公司在高商品价格时应回报现金
- Debt-to-EBITDA——在周期性大宗商品业务中高杠杆是危险的
- 资本纪律——再投资率与股东回报

估值框架：
- EV/EBITDA 是主要指标（根据商品价格进行正常化）
- FCF收益率相对于同行（考虑到商品风险溢价应较高）
- P/E 参考价值较低，因受商品价格敏感性影响
- 股息收益率+回购收益率 = 总股东回报
- 基于储量估值的NAV（适用于勘探开发公司）

宏观敏感性：
- 当前油/气价格与历史及远期曲线对比
- OPEC+供应决策和地缘政治供应风险
- 能源转型风险——长期需求前景
- 监管/碳政策风险敞口""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "dividend_yield"],
        sentiment_context="关注油气价格、OPEC+决策、地缘政治供应风险、能源转型政策、资本配置决策以及产量指引。",
    ),

    "renewables": SectorConfig(
        key="renewables",
        label="可再生能源",
        analysis_framework="""你正在分析一家可再生能源公司。请运用项目管线分析框架：

优先指标：
- 装机容量（MW/GW）及增长轨迹
- 新项目投产带来的营收增长
- 购电协议（PPA）定价和期限
- 平准化能源成本（LCOE）竞争力
- 负债水平——可再生能源项目属于资本密集型
- 政府激励政策敞口（ITC、PTC、IRA税收抵免）

估值框架：
- EV/EBITDA 相对于合同现金流可见度
- 项目管线价值和开发阶段
- 基于长期合同收入的DCF
- 依赖补贴的公司需考虑政策风险溢价""",
        key_metrics=["revenue_growth", "debt_to_equity", "market_cap"],
        sentiment_context="关注政府能源政策、税收抵免延期/到期、公用事业规模采购、电网并网时间线以及技术成本曲线。",
    ),

    # ===== CONSUMER =====

    "ecommerce": SectorConfig(
        key="ecommerce",
        label="电子商务/网络零售",
        analysis_framework="""你正在分析一家电子商务/网络零售公司。请运用GMV和费率分析框架：

优先指标：
- 商品交易总额（GMV）或总营收增长
- 费率/货币化率趋势
- 毛利率——尤其是自营（1P）与第三方市场（3P）的比例构成
- 广告和服务收入占总收入百分比（高利润率收入来源）
- 履约成本趋势和物流效率
- 活跃买家/客户数量和购买频率

估值框架：
- 高增长阶段使用 EV/Revenue，成熟电商使用 P/E
- 分部盈利能力（将零售与云计算/广告/服务分开）
- 随着增长投资放缓，FCF产生能力
- 与实体零售同行比较单位经济模型""",
        key_metrics=["revenue_growth", "pe_ratio", "forward_pe", "market_cap"],
        sentiment_context="关注消费支出趋势、线上渗透率增长、竞争格局、物流/履约动态以及广告市场状况。",
    ),

    "retail": SectorConfig(
        key="retail",
        label="零售业",
        analysis_framework="""你正在分析一家零售公司。请运用同店销售分析框架：

优先指标：
- 可比/同店销售增长（零售业最重要的单一指标）
- 营收增长：同店贡献与新店贡献
- 毛利率和库存管理效率
- 库存周转率——库存上升而营收未增长是危险信号
- 销售管理费用占营收百分比——运营效率
- 电商渗透率和全渠道能力

估值框架：
- P/E 相对于零售同行和增长率
- EV/EBITDA 用于跨资本结构比较
- 成熟零售商的股息收益率
- 同店销售轨迹驱动估值倍数扩张/收缩""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "current_ratio"],
        sentiment_context="关注消费支出趋势、客流量数据、促销活动、库存水平、电商竞争以及季节性因素。",
    ),

    "consumer_staples": SectorConfig(
        key="consumer_staples",
        label="必需消费品/防御性消费",
        analysis_framework="""你正在分析一家必需消费品/防御性消费公司。请运用品牌实力和定价能力分析框架：

优先指标：
- 有机营收增长（区分价格贡献和销量贡献）
- 毛利率——定价能力指标，对原材料成本敏感
- 销量趋势——消费者是否在转向自有品牌？
- 营业利润率和成本效率改善
- 股息收益率和派息率——必需消费品是股息型股票
- FCF转化率——稳定的业务应能可靠地将盈利转化为现金

估值框架：
- P/E 是主要指标（这些是稳定的盈利公司）
- 股息收益率作为估值底部支撑
- 抗衰退性和可预测性支撑溢价P/E
- 与历史估值区间和行业平均水平比较""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "dividend_yield", "return_on_equity"],
        sentiment_context="关注消费支出模式、原材料成本通胀、自有品牌竞争、提价公告以及衰退指标。",
    ),

    "auto": SectorConfig(
        key="auto",
        label="汽车行业",
        analysis_framework="""你正在分析一家汽车公司。请运用单位经济模型分析框架：

优先指标：
- 汽车交付量/销量及增长轨迹
- 平均售价（ASP）趋势——产品组合是否向更高利润率车型转移？
- 每辆车的毛利率（汽车毛利率较薄：15-25%）
- R&D支出——尤其是电动汽车和自动驾驶投资
- 资本支出强度和工厂产能利用率
- 库存水平（供应天数）——过高则预示需求疲软

估值框架：
- 盈利车企使用 P/E（使用前瞻盈利而非过去盈利）
- 未盈利或高增长电动车公司使用 EV/Revenue
- 交付增长率与估值溢价的对比
- 与传统车企和电动车同行比较毛利率""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "debt_to_equity"],
        sentiment_context="关注汽车交付数据、电动汽车普及趋势、电池技术发展、自动驾驶进展、关税/贸易影响以及消费金融条件。",
    ),

    "consumer_services": SectorConfig(
        key="consumer_services",
        label="消费服务",
        analysis_framework="""你正在分析一家消费服务公司。请运用单位经济模型分析框架：

优先指标：
- 由交易量、预订量或活跃用户驱动的营收增长
- 毛利率和每笔交易的边际贡献
- 客户获取成本趋势和回收期
- 活跃用户/客户数量和参与度指标（复购率）
- 季节性规律和地域多元化
- 经营杠杆——利润率是否随规模扩张而提高？

估值框架：
- 盈利的服务公司使用 P/E 或 EV/EBITDA
- 高增长、未盈利的公司使用 EV/Revenue
- 与同类别同行比较单位经济模型""",
        key_metrics=["revenue_growth", "pe_ratio", "return_on_equity", "market_cap"],
        sentiment_context="关注消费支出趋势、旅游/休闲需求、竞争格局以及影响可选消费的宏观经济指标。",
    ),

    "consumer_cyclical": SectorConfig(
        key="consumer_cyclical",
        label="可选消费",
        analysis_framework="""你正在分析一家可选消费公司。请运用周期与需求分析框架：

优先指标：
- 营收增长相对于经济周期所处阶段
- 同店销售或适用的可比指标
- 不同经济环境下的毛利率和定价能力
- 库存管理效率
- 抵御经济低迷的资产负债表实力
- 对消费者信心和支出趋势的敏感性

估值框架：
- 根据周期位置正常化的 P/E
- 警惕周期峰值时的低P/E（周期性陷阱）
- EV/EBITDA 相对于同行在相似周期阶段的水平
- 完整经济周期中的股息可持续性""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "debt_to_equity", "current_ratio"],
        sentiment_context="关注消费者信心、可选消费支出趋势、经济周期指标以及竞争定位。",
    ),

    # ===== REAL ESTATE =====

    "real_estate": SectorConfig(
        key="real_estate",
        label="房地产/REITs",
        analysis_framework="""你正在分析一家房地产/REIT公司。请运用NAV和FFO分析框架：

优先指标：
- 每股运营资金（FFO）及增长——REIT的盈利等价指标
- 调整后FFO（AFFO）扣除经常性资本支出
- 出租率和租户留存率
- 同物业净营业收入（NOI）增长
- 加权平均租约到期期限（WALE）——越长越可预测
- Debt-to-EBITDA 和利息覆盖率——REITs使用大量杠杆
- 股息收益率和AFFO派息率

估值框架：
- P/FFO（REIT的P/E等价指标）
- NAV折价/溢价——股价高于还是低于资产价值？
- 当前定价隐含的资本化率与市场资本化率对比
- 股息收益率相对于REIT同行和利率水平
- 注意：传统P/E对REITs具有误导性，因折旧的影响

利率敏感性：
- 更高的利率增加借贷成本并压缩估值
- 利率上升也增加了来自固定收益替代品的竞争
- 物业类型很重要：工业/数据中心REITs具有长期结构性顺风""",
        key_metrics=["pe_ratio", "dividend_yield", "debt_to_equity", "market_cap"],
        sentiment_context="关注利率走势、各物业类型需求趋势、出租率、在建工程管线以及资本化率变动。",
    ),

    # ===== UTILITIES =====

    "utilities": SectorConfig(
        key="utilities",
        label="公用事业",
        analysis_framework="""你正在分析一家公用事业公司。请运用受监管回报分析框架：

优先指标：
- 费率基数增长——受监管公用事业盈利增长的主要驱动力
- 监管机构允许的ROE
- 实际ROE与允许ROE的对比——执行质量指标
- 资本支出计划和费率审批管线
- 股息收益率和派息率——公用事业主要是收益型投资
- 负债权益比——公用事业使用较高但可控的杠杆
- 监管辖区质量——友好型与挑战型监管机构

估值框架：
- P/E 相对于公用事业同行和允许ROE
- 股息收益率作为主要估值锚点
- 费率基数增长率 ≈ 近似盈利增长率
- 友好型监管和增长性资本支出支撑溢价P/E

特别考量：
- AI/数据中心电力需求作为增长催化剂
- 可再生能源转型带来的资本支出机会
- 野火/极端天气责任风险（针对特定地区）""",
        key_metrics=["pe_ratio", "eps", "dividend_yield", "debt_to_equity"],
        sentiment_context="关注费率审批决定、监管动态、天气事件、AI数据中心电力需求、可再生能源强制要求以及利率敏感性。",
    ),

    # ===== INDUSTRIALS =====

    "industrials": SectorConfig(
        key="industrials",
        label="工业",
        analysis_framework="""你正在分析一家工业公司。请运用积压订单与周期分析框架：

优先指标：
- 有机营收增长（剔除并购和汇率影响）
- 订单积压和订单出货比——未来营收的领先指标
- 毛利率和营业利润率——定价能力和运营效率的指标
- FCF转化率——工业公司应能可靠地将盈利转化为现金
- 资本支出强度相对于折旧——公司是否在为增长进行投资？
- 终端市场多元化——降低周期性风险

估值框架：
- P/E 和 EV/EBITDA 相对于工业同行
- 跨周期盈利正常化（避免在峰值/低谷利润率下估值）
- FCF收益率作为质量验证
- 并购策略和被收购资本的回报

周期性意识：
- 工业股与PMI和资本支出周期相关
- 基础设施支出和政府政策顺风
- 供应链正常化和原材料成本趋势""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity", "debt_to_equity"],
        sentiment_context="关注PMI趋势、基础设施支出、供应链动态、原材料成本、订单/积压趋势以及政府政策影响。",
    ),

    # ===== COMMUNICATION / MEDIA =====

    "communication": SectorConfig(
        key="communication",
        label="通信/媒体",
        analysis_framework="""你正在分析一家通信/媒体公司。请运用用户参与度分析框架：

优先指标：
- 用户/订阅者指标：DAU、MAU、订阅者数量及增长趋势
- 每用户平均收入（ARPU）——货币化效果
- 广告收入增长和定价趋势
- 内容支出和投资回报率（针对媒体公司）
- 参与度指标：使用时长、会话数、留存率
- 新兴平台的营业利润率和盈利路径

估值框架：
- 高增长平台使用 EV/Revenue 或 EV/用户数
- 盈利的成熟媒体公司使用 P/E
- 每用户广告收入与同行对比
- 内容资产摊销模式""",
        key_metrics=["revenue_growth", "pe_ratio", "market_cap", "return_on_equity"],
        sentiment_context="关注用户增长趋势、广告市场状况、内容支出、监管审查、竞争性平台动态以及AI内容影响。",
    ),

    "telecom": SectorConfig(
        key="telecom",
        label="电信",
        analysis_framework="""你正在分析一家电信公司。请运用用户数和基础设施分析框架：

优先指标：
- 用户数量和净新增数（无线和宽带）
- ARPU趋势——定价能力与竞争压力
- 流失率——越低越好，表明客户满意度
- 资本支出强度（5G、光纤建设）和网络质量
- EBITDA利润率和FCF产生能力——电信是资本密集型行业
- 负债水平——电信公司承载大量杠杆
- 股息收益率和可持续性

估值框架：
- EV/EBITDA 是电信行业的主要估值指标
- 股息收益率作为收益锚点
- 资本支出后的FCF收益率（决定股息可持续性）
- 频谱资产价值和网络基础设施重置成本""",
        key_metrics=["revenue_growth", "pe_ratio", "dividend_yield", "debt_to_equity"],
        sentiment_context="关注用户趋势、5G部署、宽带竞争、频谱拍卖、资费动态以及监管动态。",
    ),

    # ===== MATERIALS =====

    "materials": SectorConfig(
        key="materials",
        label="基础材料",
        analysis_framework="""你正在分析一家基础材料公司。请运用大宗商品和成本曲线分析框架：

优先指标：
- 营收和销量趋势相对于大宗商品价格
- 在行业成本曲线上的位置（低成本生产商最佳）
- 毛利率——对大宗商品价格波动高度敏感
- 单位现金运营成本（每盎司、每吨等）
- 储量寿命和资源品质（针对矿业）
- 维护性和增长性资本支出需求

估值框架：
- 根据大宗商品周期位置正常化的 P/E
- 不同大宗商品价格情景下的 EV/EBITDA
- 基于储量估值的P/NAV（矿业/金属）
- 当前大宗商品价格下的FCF收益率
- 跨价格周期的股息可持续性""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "dividend_yield"],
        sentiment_context="关注大宗商品价格走势、供需动态、矿山开发、贸易/关税政策以及环境监管动态。",
    ),

    # ===== GENERAL FALLBACKS =====

    "technology_general": SectorConfig(
        key="technology_general",
        label="科技",
        analysis_framework="""你正在分析一家科技公司。请运用增长与护城河分析框架：

优先指标：
- 营收增长率及一致性
- 毛利率（高利润率表明软件/IP价值；较低则表明硬件/服务）
- R&D支出占营收百分比——对未来竞争力的投资
- 经营杠杆——利润率是否随营收增长而扩张？
- FCF利润率和转化质量
- 市场地位和竞争格局

估值框架：
- 高增长（>20%）：EV/Revenue 为主，P/E 为辅
- 成熟增长（<20%）：P/E 和 PEG比率为主
- 现金产生型科技公司的FCF收益率
- 将估值倍数与增长率比较（经验法则：P/E ≈ 增长率为合理估值）""",
        key_metrics=["revenue_growth", "pe_ratio", "forward_pe", "return_on_equity", "market_cap"],
        sentiment_context="关注技术采用趋势、竞争格局、监管动态以及宏观经济对科技支出的影响。",
    ),

    "healthcare_general": SectorConfig(
        key="healthcare_general",
        label="医疗保健",
        analysis_framework="""你正在分析一家医疗保健公司。请运用多元化医疗保健分析框架：

优先指标：
- 按业务板块划分的营收增长
- 毛利率（差异很大：器械约40%，药品可达80%）
- R&D支出和管线生产力
- 监管风险敞口
- 专利保护和独占期时间线
- 成熟医疗保健公司的股息收益率

估值框架：
- P/E 相对于医疗保健行业同行
- 研发密集型公司的管线价值
- 成熟、多元化医疗保健公司的股息收益率""",
        key_metrics=["revenue_growth", "pe_ratio", "eps", "return_on_equity", "dividend_yield"],
        sentiment_context="关注监管动态、药品定价政策、临床试验结果以及医疗保健支出趋势。",
    ),

    "financials_general": SectorConfig(
        key="financials_general",
        label="金融服务",
        analysis_framework="""你正在分析一家金融服务公司。请运用回报与风险分析框架：

优先指标：
- ROE 或 ROTCE——关键指标
- 营收增长和构成（手续费收入与利差收入）
- 效率比率或成本收入比
- 资本充足率和监管资本比率
- 适用时的信贷质量指标（冲销率、不良贷款率）
- 股息收益率和总资本回报

估值框架：
- 资本密集型金融机构使用市净率（P/B）或P/TBV
- 手续费驱动型金融机构（资产管理、交易所）使用 P/E
- 股息收益率作为收益组成部分
- ROE 高于权益成本则支撑溢价于账面价值""",
        key_metrics=["pe_ratio", "return_on_equity", "dividend_yield", "debt_to_equity"],
        sentiment_context="关注利率环境、信贷状况、监管变化、资本市场活动以及宏观经济指标。",
    ),

    # ===== DEFAULT (catch-all) =====

    "default": SectorConfig(
        key="default",
        label="通用",
        analysis_framework="""请运用通用基本面分析框架：

优先指标：
- 营收增长趋势及一致性
- 盈利能力：毛利率、营业利润率、净利率、ROE
- 资产负债表实力：负债水平、流动比率、利息覆盖率
- 估值：P/E 相对于增长率（PEG比率）
- 现金流：FCF产生能力和盈利质量
- 竞争地位和护城河持久性

估值框架：
- P/E 相对于行业平均水平和历史区间
- PEG比率用于增长调整后的估值
- EV/EBITDA 用于跨资本结构比较
- 股息收益率和派息可持续性（如适用）
- FCF收益率作为质量指标""",
        key_metrics=["revenue_growth", "pe_ratio", "debt_to_equity", "return_on_equity", "dividend_yield"],
        sentiment_context="关注市场整体情绪、竞争格局以及影响该公司的宏观经济因素。",
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
