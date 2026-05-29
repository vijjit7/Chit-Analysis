"""Unit tests for chit_engine."""

from datetime import date

import pytest

from chit_engine import (
    ChitParams,
    analyze_chit,
    analyze_lifting_month,
    compute_xirr,
    compute_xnpv,
)

FIXED_PARAMS = ChitParams(
    chit_value=500_000,
    num_members=25,
    commission_pct=5.0,
    fixed_prize=400_000,
    fixed_monthly_payment=18_000,
    annual_discount_rate=12.0,
    start_date=date(2025, 1, 1),
)

MID_ENTRY_PARAMS = ChitParams(
    chit_value=500_000,
    num_members=25,
    commission_pct=5.0,
    fixed_prize=400_000,
    fixed_monthly_payment=18_000,
    annual_discount_rate=12.0,
    start_date=date(2025, 10, 1),
    entry_month=10,
    till_date_payment=135_000,
)


class TestCashFlows:
    """Test cash flow construction."""

    def test_flow_count(self):
        result = analyze_lifting_month(FIXED_PARAMS, 10)
        assert len(result.cash_flows) == FIXED_PARAMS.num_members

    def test_prize_in_lifting_month_only(self):
        lift_month = 10
        result = analyze_lifting_month(FIXED_PARAMS, lift_month)
        for flow in result.cash_flows:
            if flow.month == lift_month:
                assert flow.prize > 0
            else:
                assert flow.prize == 0

    def test_all_payments_negative(self):
        result = analyze_lifting_month(FIXED_PARAMS, 5)
        for flow in result.cash_flows:
            assert flow.payment < 0


class TestXNPV:
    """Test XNPV computation."""

    def test_xnpv_at_zero_rate(self):
        result = analyze_lifting_month(FIXED_PARAMS, 10)
        cf_values = [f.net_flow for f in result.cash_flows]
        cf_dates = [f.date for f in result.cash_flows]
        xnpv_zero = compute_xnpv(0.0, cf_values, cf_dates)
        assert xnpv_zero == pytest.approx(sum(cf_values), abs=0.01)

    def test_xnpv_empty(self):
        assert compute_xnpv(0.1, [], []) == 0.0


class TestXIRR:
    """Test XIRR computation."""

    def test_known_xirr(self):
        cf = [-1000.0, 1100.0]
        dates = [date(2025, 1, 1), date(2026, 1, 1)]
        xirr = compute_xirr(cf, dates)
        assert xirr is not None
        assert xirr == pytest.approx(0.10, abs=0.001)

    def test_xirr_all_negative(self):
        cf = [-100.0, -200.0]
        dates = [date(2025, 1, 1), date(2025, 7, 1)]
        assert compute_xirr(cf, dates) is None

    def test_xirr_all_positive(self):
        cf = [100.0, 200.0]
        dates = [date(2025, 1, 1), date(2025, 7, 1)]
        assert compute_xirr(cf, dates) is None

    def test_xirr_too_few(self):
        assert compute_xirr([100.0], [date(2025, 1, 1)]) is None


class TestAnalyzeChit:
    """Test full chit analysis."""

    def test_optimal_month_in_range(self):
        result = analyze_chit(FIXED_PARAMS)
        assert 1 <= result.optimal_month <= FIXED_PARAMS.num_members

    def test_all_months_analyzed(self):
        result = analyze_chit(FIXED_PARAMS)
        assert len(result.analyses) == FIXED_PARAMS.num_members

    def test_recommendation_not_empty(self):
        result = analyze_chit(FIXED_PARAMS)
        assert len(result.recommendation) > 0


class TestFixedChit:
    """Test fixed chit specifics."""

    def test_fixed_chit_runs(self):
        result = analyze_chit(FIXED_PARAMS)
        assert len(result.analyses) == FIXED_PARAMS.num_members
        assert 1 <= result.optimal_month <= FIXED_PARAMS.num_members

    def test_fixed_chit_prize(self):
        result = analyze_lifting_month(FIXED_PARAMS, 5)
        assert result.prize_received == FIXED_PARAMS.fixed_prize

    def test_fixed_chit_payment(self):
        result = analyze_lifting_month(FIXED_PARAMS, 5)
        assert result.cash_flows[0].payment == pytest.approx(-FIXED_PARAMS.fixed_monthly_payment)

    def test_fixed_chit_uniform_payments(self):
        """Fixed chit should have the same payment every month."""
        result = analyze_lifting_month(FIXED_PARAMS, 25)
        payments = [f.payment for f in result.cash_flows]
        assert all(p == payments[0] for p in payments)


class TestMiddleEntry:
    """Test middle entry support."""

    def test_flow_count_middle_entry(self):
        result = analyze_lifting_month(MID_ENTRY_PARAMS, 15)
        expected_months = MID_ENTRY_PARAMS.num_members - MID_ENTRY_PARAMS.entry_month + 1
        assert len(result.cash_flows) == expected_months

    def test_first_flow_is_entry_month(self):
        result = analyze_lifting_month(MID_ENTRY_PARAMS, 15)
        assert result.cash_flows[0].month == MID_ENTRY_PARAMS.entry_month

    def test_entry_month_pays_lump_sum(self):
        result = analyze_lifting_month(MID_ENTRY_PARAMS, 15)
        assert result.cash_flows[0].payment == pytest.approx(-MID_ENTRY_PARAMS.till_date_payment)

    def test_mid_entry_analysis_count(self):
        result = analyze_chit(MID_ENTRY_PARAMS)
        expected = MID_ENTRY_PARAMS.num_members - MID_ENTRY_PARAMS.entry_month + 1
        assert len(result.analyses) == expected

    def test_mid_entry_optimal_in_range(self):
        result = analyze_chit(MID_ENTRY_PARAMS)
        assert MID_ENTRY_PARAMS.entry_month <= result.optimal_month <= MID_ENTRY_PARAMS.num_members

    def test_mid_entry_prize_appears(self):
        lift = 15
        result = analyze_lifting_month(MID_ENTRY_PARAMS, lift)
        prizes = [f for f in result.cash_flows if f.prize > 0]
        assert len(prizes) == 1
        assert prizes[0].month == lift
