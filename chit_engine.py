"""Core calculation engine for chit fund analysis.

Monthly payments are linearly interpolated from ``first_auction_payment``
at the first auction month to ``V/N`` at month N. Each auction month's
prize is derived from that payment so the cash-flow identity holds:
    prize_m = N * payment_m - V * commission
    net at lift = (N - 1) * payment_m - V * commission
"""

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional, Tuple

from dateutil.relativedelta import relativedelta
import numpy as np
from scipy import optimize


@dataclass
class ChitParams:
    """Input parameters for chit fund analysis."""
    chit_value: float
    num_members: int
    commission_pct: float
    first_auction_payment: float = 0.0
    last_payment: float = 0.0
    annual_discount_rate: float = 12.0
    start_date: date = field(default_factory=date.today)
    entry_month: int = 1
    till_date_payment: float = 0.0
    company_chit_months: Tuple[int, ...] = (1, 3)

    def __post_init__(self) -> None:
        if self.num_members <= 0:
            return
        base = self.chit_value / self.num_members
        if self.first_auction_payment <= 0:
            self.first_auction_payment = base * 0.65
        if self.last_payment <= 0:
            self.last_payment = base


@dataclass
class MonthlyFlow:
    """Cash flow for a single month.

    ``is_reference`` marks a completed month shown for context only (e.g. the
    last completed auction before a middle-joining entry). Reference rows carry
    the prize that was distributed that month but are excluded from the
    member's cash flow, NPV and XIRR.
    """
    month: int
    date: date
    payment: float
    prize: float
    net_flow: float
    cumulative: float
    is_reference: bool = False


@dataclass
class LiftingAnalysis:
    """Analysis result for lifting in a specific month."""
    lift_month: int
    cash_flows: List[MonthlyFlow]
    total_paid: float
    prize_received: float
    net_cost: float
    npv: float
    xirr: Optional[float]


@dataclass
class ChitAnalysisResult:
    """Complete analysis across all possible lifting months."""
    params: ChitParams
    analyses: List[LiftingAnalysis]
    optimal_month: int
    recommendation: str


def compute_xnpv(rate: float, cash_flows: List[float], dates: List[date]) -> float:
    """Compute XNPV given a discount rate, cash flows, and dates."""
    if not cash_flows:
        return 0.0
    d0 = dates[0]
    total = 0.0
    for cf, dt in zip(cash_flows, dates):
        exponent = (dt - d0).days / 365.0
        base = 1 + rate
        if base <= 0:
            return float('inf')
        total += cf / (base ** exponent)
    return total


def compute_xirr(cash_flows: List[float], dates: List[date]) -> Optional[float]:
    """Compute XIRR by solving XNPV=0. Returns None if no solution found."""
    if not cash_flows or len(cash_flows) < 2:
        return None

    has_positive = any(cf > 0 for cf in cash_flows)
    has_negative = any(cf < 0 for cf in cash_flows)
    if not (has_positive and has_negative):
        return None

    def f(r):
        return compute_xnpv(r, cash_flows, dates)

    try:
        return optimize.brentq(f, -0.5, 10.0, xtol=1e-8, maxiter=1000)
    except (ValueError, RuntimeError):
        pass

    try:
        result = optimize.newton(f, 0.1, maxiter=1000, tol=1e-8)
        if np.isfinite(result) and -1.0 < result < 20.0:
            return float(result)
    except (ValueError, RuntimeError, OverflowError):
        pass

    return None


def compute_schedule(params: ChitParams) -> Tuple[List[float], List[float]]:
    """Build per-month (payments, prizes) lists for the whole tenure.

    Model:
      - Company chit months: payment = V/N, prize = 0 (foreman takes the
        chit; members cannot lift).
      - Auction months in [first_auction, N]: payment is linearly
        interpolated from ``first_auction_payment`` at the first auction
        month (first non-company month >= entry_month) to ``last_payment``
        at month N. Prize is derived so contributions reconcile:
          prize_m = N * payment_m - V * commission
      - Months before the entry month don't affect the analysis; they're
        returned as V/N placeholders.
    """
    V = params.chit_value
    N = params.num_members
    c = params.commission_pct / 100.0
    base_sub = V / N
    company = {m for m in params.company_chit_months if 1 <= m <= N}
    entry = max(1, params.entry_month)

    if entry > 1:
        # Middle joining: the last completed auction (the month before entry) is
        # the interpolation anchor. ``first_auction_payment`` is that month's
        # chit amount, and payments rise from it toward ``last_payment`` at
        # month N. The anchor month gets a real prize so it can be shown as a
        # completed-month reference row.
        anchor = entry - 1
    else:
        anchor = next((m for m in range(1, N + 1) if m not in company), N)
    first_payment = params.first_auction_payment
    last_payment = params.last_payment
    span = N - anchor

    payments: List[float] = []
    prizes: List[float] = []
    for m in range(1, N + 1):
        if m in company:
            payments.append(base_sub)
            prizes.append(0.0)
        elif m < anchor:
            payments.append(base_sub)
            prizes.append(0.0)
        else:
            if span > 0:
                pmt = first_payment + (last_payment - first_payment) * (m - anchor) / span
            else:
                pmt = last_payment
            payments.append(pmt)
            prizes.append(N * pmt - V * c)
    return payments, prizes


def analyze_lifting_month(params: ChitParams, lift_month: int) -> LiftingAnalysis:
    """Build cash flows and compute NPV/XIRR for lifting in a given month."""
    N = params.num_members
    rate = params.annual_discount_rate / 100.0
    entry = params.entry_month
    till_date = params.till_date_payment

    payments, prizes = compute_schedule(params)
    prize_for_lift = prizes[lift_month - 1]

    flows: List[MonthlyFlow] = []
    cf_values: List[float] = []
    cf_dates: List[date] = []
    cumulative = 0.0
    total_paid = 0.0

    # Middle joining: show the last completed auction (month entry-1) as a
    # reference row. It displays the prize distributed that month but does not
    # contribute to the member's cash flow, NPV or XIRR (those months are
    # already settled by the till-date lump paid at entry).
    if entry > 1:
        ref_month = entry - 1
        ref_date = params.start_date + relativedelta(months=-1)
        ref_prize = prizes[ref_month - 1]
        flows.append(MonthlyFlow(
            month=ref_month,
            date=ref_date,
            payment=0.0,
            prize=ref_prize,
            net_flow=ref_prize,
            cumulative=0.0,
            is_reference=True,
        ))

    for m in range(entry, N + 1):
        months_from_entry = m - entry
        dt = params.start_date + relativedelta(months=months_from_entry)

        if m == entry and till_date > 0:
            # Entry month: the till-date lump catches up the months already
            # completed (through the prior month), PLUS this month's own
            # projected chit subscription, which is due now.
            payment = -(till_date + payments[m - 1])
        else:
            payment = -payments[m - 1]

        prize_this_month = prize_for_lift if m == lift_month else 0.0
        net = payment + prize_this_month
        cumulative += net
        total_paid += abs(payment)

        flows.append(MonthlyFlow(
            month=m,
            date=dt,
            payment=payment,
            prize=prize_this_month,
            net_flow=net,
            cumulative=cumulative,
        ))
        cf_values.append(net)
        cf_dates.append(dt)

    npv = compute_xnpv(rate, cf_values, cf_dates)
    xirr = compute_xirr(cf_values, cf_dates)

    return LiftingAnalysis(
        lift_month=lift_month,
        cash_flows=flows,
        total_paid=total_paid,
        prize_received=prize_for_lift,
        net_cost=total_paid - prize_for_lift,
        npv=npv,
        xirr=xirr,
    )


def analyze_chit(params: ChitParams) -> ChitAnalysisResult:
    """Run analysis for all possible lifting months and find the optimal one."""
    N = params.num_members
    entry = params.entry_month
    company = {m for m in params.company_chit_months if 1 <= m <= N}
    candidates = [k for k in range(entry, N + 1) if k not in company]
    if not candidates:
        candidates = [N]
    analyses = [analyze_lifting_month(params, k) for k in candidates]

    best = max(analyses, key=lambda a: a.npv)
    optimal_month = best.lift_month

    if best.npv > 0:
        xirr_str = f"{best.xirr * 100:.1f}%" if best.xirr is not None else "N/A"
        recommendation = (
            f"ENTER the chit. Optimal strategy: lift in month {optimal_month} "
            f"(NPV = {best.npv:,.0f}, XIRR = {xirr_str}). "
            f"The chit outperforms your required rate of return."
        )
    else:
        recommendation = (
            f"AVOID this chit. Even the best lifting month ({optimal_month}) "
            f"yields a negative NPV of {best.npv:,.0f}. "
            f"The chit underperforms your required rate of return."
        )

    return ChitAnalysisResult(
        params=params,
        analyses=analyses,
        optimal_month=optimal_month,
        recommendation=recommendation,
    )


@dataclass
class ReportFinding:
    """A single observation in the analysis report.

    ``severity`` is one of 'good', 'warn', 'bad', 'info' and drives how the
    finding is rendered (green / amber / red / blue).
    """
    severity: str
    title: str
    detail: str


@dataclass
class AnalysisReport:
    """Human-facing advisory built on top of a ChitAnalysisResult."""
    decision: str            # 'ENTER' or 'AVOID'
    verdict: str
    best_month: int
    best_npv: float
    best_xirr: Optional[float]
    required_rate: float
    timing: List[ReportFinding]
    fairness: List[ReportFinding]
    fairness_verdict: str


def build_analysis_report(result: ChitAnalysisResult) -> AnalysisReport:
    """Turn raw cash-flow analytics into an enter/avoid + timing + fairness advisory.

    Three questions are answered:
      1. Enter or avoid (best NPV vs the required rate).
      2. When to act — lift early (borrow cheaply), lift late (save), or the
         NPV-optimal slot at the required rate.
      3. Was it fairly done — foreman commission, how strongly timing tilts the
         outcome, the dividend trend, and (for mid-entry) whether the entry
         price fairly credits dividends already earned.
    """
    p = result.params
    V, N = p.chit_value, p.num_members
    c = p.commission_pct / 100.0
    rate = p.annual_discount_rate
    base_sub = V / N if N else 0.0
    entry = max(1, p.entry_month)

    analyses = sorted(result.analyses, key=lambda a: a.lift_month)
    best = max(analyses, key=lambda a: a.npv)
    last = analyses[-1]
    enter = best.npv > 0

    # ---- 1. Decision ----
    # XIRR is only meaningful when the lift stream changes sign once. With a large
    # mid-entry lump most lift months flip sign several times, so guard the value.
    best_xirr = best.xirr if (best.xirr is not None and -1.0 < best.xirr < 2.0) else None
    xs = f"{best_xirr * 100:.1f}%" if best_xirr is not None else "N/A"
    if enter:
        decision = "ENTER"
        rate_clause = f" (effective annual rate {xs})" if best_xirr is not None else ""
        verdict = (
            f"Entering is worthwhile. At your required return of {rate:.1f}% the best "
            f"outcome is to lift in month {best.lift_month}, giving a positive NPV of "
            f"{best.npv:,.0f}{rate_clause}. The chit beats your benchmark, so the capital "
            "is well deployed here."
        )
    else:
        decision = "AVOID"
        verdict = (
            f"Avoid this chit. Even the best lifting month ({best.lift_month}) yields a "
            f"negative NPV of {best.npv:,.0f} at your {rate:.1f}% required return — it "
            "underperforms simply keeping or investing the money elsewhere."
        )

    # ---- 2. Timing: when to bid vs stay ----
    # Built on NPV (and undiscounted net cost), which stay robust even when XIRR
    # is undefined for multi-sign-change lift streams.
    timing: List[ReportFinding] = []
    midpoint = (entry + N) / 2.0
    earliest = analyses[0]
    cheapest_cost = min(analyses, key=lambda a: a.net_cost)

    rec_title = (
        f"Recommended: bid to lift in month {best.lift_month}"
        if best.lift_month <= midpoint else
        f"Recommended: hold and lift in month {best.lift_month}"
    )
    timing.append(ReportFinding(
        "good", rec_title,
        f"This is the NPV-optimal month at your {rate:.1f}% required return "
        f"(NPV {best.npv:,.0f}). It strikes the best balance between prize size, dividends "
        "collected, and how long your capital is tied up.",
    ))
    timing.append(ReportFinding(
        "info", f"Bid early — month {earliest.lift_month}",
        f"Taking the prize at the first opportunity nets {earliest.prize_received:,.0f} "
        f"(NPV {earliest.npv:,.0f}; undiscounted net cost {earliest.net_cost:,.0f}). Choose "
        "this if you genuinely need the liquidity now.",
    ))
    timing.append(ReportFinding(
        "info", f"Stay till the end — month {last.lift_month}",
        f"Riding to the last month pays the biggest prize, {last.prize_received:,.0f} "
        f"(NPV {last.npv:,.0f}; undiscounted net cost {last.net_cost:,.0f}), after collecting "
        "every dividend — but your capital is committed the whole term.",
    ))
    if cheapest_cost.lift_month != best.lift_month:
        timing.append(ReportFinding(
            "info", "Sticker cost vs. time-adjusted value",
            f"Month {cheapest_cost.lift_month} has the lowest undiscounted cost "
            f"({cheapest_cost.net_cost:,.0f}) because its prize is largest, but once your "
            f"{rate:.1f}% time value of money is applied, month {best.lift_month} comes out "
            "ahead.",
        ))

    # ---- 3. Fairness ----
    fairness: List[ReportFinding] = []

    comm_amt = c * V
    norm_amt = 0.05 * V
    if p.commission_pct > 5.0 + 1e-9:
        fairness.append(ReportFinding(
            "warn", "Foreman commission is above the norm",
            f"The organiser keeps {p.commission_pct:.1f}% of the chit value "
            f"({comm_amt:,.0f}) each cycle. The customary cap is ~5% ({norm_amt:,.0f}); the "
            f"extra {comm_amt - norm_amt:,.0f} comes straight out of members' returns.",
        ))
    else:
        fairness.append(ReportFinding(
            "good", "Foreman commission is within the norm",
            f"The organiser takes {p.commission_pct:.1f}% ({comm_amt:,.0f}), at or below the "
            "customary 5% — nothing excessive being skimmed.",
        ))

    npvs = [a.npv for a in analyses]
    if len(npvs) >= 2 and V:
        npv_range = max(npvs) - min(npvs)
        spread_pct = npv_range / V * 100
        if spread_pct >= 3.0:
            fairness.append(ReportFinding(
                "warn", "Timing strongly tilts the outcome",
                f"Your result swings by about {spread_pct:.1f}% of the chit value "
                f"({npv_range:,.0f} in NPV) depending on which month you lift. Members who can "
                "choose their timing — especially those able to wait — gain materially over "
                "anyone forced to take cash at a fixed time.",
            ))
        else:
            fairness.append(ReportFinding(
                "good", "Outcome is fairly even across timing",
                f"Your result varies only about {spread_pct:.1f}% of the chit value across "
                "lifting months, so no slot is dramatically advantaged — well balanced.",
            ))

    payments, prizes = compute_schedule(p)
    auction_idx = [i for i in range(entry - 1, N) if prizes[i] > 0]
    if auction_idx:
        div_first = V - N * payments[auction_idx[0]]
        div_last = max(0.0, V - N * payments[auction_idx[-1]])
        first_m, last_m = auction_idx[0] + 1, auction_idx[-1] + 1
        fairness.append(ReportFinding(
            "info", "Dividend trend over your remaining term",
            f"Monthly dividend runs about {div_first:,.0f} at the first auction (month "
            f"{first_m}) and tapers toward {div_last:,.0f} by month {last_m} as auction "
            "discounts shrink. Earlier months return more dividend; later months a bigger "
            "prize.",
        ))

    if entry > 1 and p.till_date_payment > 0:
        gross = (entry - 1) * base_sub
        credit = gross - p.till_date_payment
        credit_pct = (credit / gross * 100) if gross else 0.0
        if credit > 0.01 * gross:
            fairness.append(ReportFinding(
                "good", "Mid-entry price looks fair",
                f"You pay {p.till_date_payment:,.0f} to step into a seat that has absorbed "
                f"{gross:,.0f} of gross subscriptions over {entry - 1} completed months — a "
                f"{credit:,.0f} ({credit_pct:.1f}%) discount that reflects dividends already "
                "earned. The entry price is reasonable.",
            ))
        elif credit < -0.01 * gross:
            fairness.append(ReportFinding(
                "bad", "You are overpaying to enter",
                f"You pay {p.till_date_payment:,.0f} to enter — MORE than the {gross:,.0f} of "
                f"gross subscriptions for the {entry - 1} completed months. You're overpaying "
                f"by {-credit:,.0f}; the seat is priced against you and someone is taking "
                "advantage. Negotiate the entry price down.",
            ))
        else:
            fairness.append(ReportFinding(
                "warn", "Little dividend credit on entry",
                f"You pay {p.till_date_payment:,.0f} to enter, almost exactly the {gross:,.0f} "
                f"gross subscription for {entry - 1} months — you're getting little or no "
                "credit for the dividends those months earned. Push for a lower entry price.",
            ))

    if any(f.severity == "bad" for f in fairness):
        fairness_verdict = (
            "Unfair in at least one respect — see the red flag(s) below; a party is taking "
            "advantage."
        )
    elif any(f.severity == "warn" for f in fairness):
        fairness_verdict = (
            "Broadly workable, with caveats — a couple of terms tilt against you; negotiate "
            "where flagged."
        )
    else:
        fairness_verdict = (
            "The chit looks fairly structured — commission, timing and (where relevant) entry "
            "price are all within reasonable bounds."
        )

    return AnalysisReport(
        decision=decision,
        verdict=verdict,
        best_month=best.lift_month,
        best_npv=best.npv,
        best_xirr=best_xirr,
        required_rate=rate,
        timing=timing,
        fairness=fairness,
        fairness_verdict=fairness_verdict,
    )
