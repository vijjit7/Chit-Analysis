"""Streamlit UI for Chit Fund Analysis Tool."""

import json
import re
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytesseract
import streamlit as st
from dateutil.relativedelta import relativedelta
from PIL import Image

from chit_engine import (
    ChitParams,
    analyze_chit,
    build_analysis_report,
    compute_schedule,
)

SAVED_CHITS_PATH = Path(__file__).parent / "saved_chits.json"
SIMILAR_VALUE_TOLERANCE = 0.20  # ±20% of chit value counts as "similar"
SIMILAR_TENURE_TOLERANCE = 2  # within ±2 members/months counts as "similar"

st.set_page_config(page_title="Chit Fund Analyzer", layout="wide")

# --- Global sidebar tightening: less vertical space at the top ---
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > label {
        margin-bottom: 0.15rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 0.15rem !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 0.5rem 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stButton"] {
        margin-top: 0.25rem !important;
        margin-bottom: 0.25rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1 — Landing: choose entry type
# ============================================================
if "entry_type" not in st.session_state:
    st.markdown(
        """
        <style>
        /* Flashy animated background on the main app surface */
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at 20% 30%, rgba(255, 99, 132, 0.55), transparent 45%),
                radial-gradient(circle at 80% 20%, rgba(255, 206, 86, 0.55), transparent 45%),
                radial-gradient(circle at 70% 80%, rgba(75, 192, 192, 0.55), transparent 45%),
                radial-gradient(circle at 30% 75%, rgba(153, 102, 255, 0.55), transparent 45%),
                linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            background-size: 200% 200%, 220% 220%, 180% 180%, 240% 240%, 100% 100%;
            animation: bgShift 18s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }
        @keyframes bgShift {
            0%   { background-position:  0%   0%,  100%   0%,  100% 100%,   0% 100%, 0 0; }
            50%  { background-position: 100% 100%,   0% 100%,    0%   0%, 100%   0%, 0 0; }
            100% { background-position:  0%   0%,  100%   0%,  100% 100%,   0% 100%, 0 0; }
        }

        /* Floating glow orbs sprinkled over the background */
        [data-testid="stAppViewContainer"]::before,
        [data-testid="stAppViewContainer"]::after {
            content: "";
            position: fixed;
            border-radius: 50%;
            filter: blur(60px);
            opacity: 0.55;
            pointer-events: none;
            z-index: 0;
        }
        [data-testid="stAppViewContainer"]::before {
            width: 420px; height: 420px;
            background: radial-gradient(circle, #ff4b4b 0%, transparent 70%);
            top: -80px; left: -80px;
            animation: orbA 14s ease-in-out infinite;
        }
        [data-testid="stAppViewContainer"]::after {
            width: 500px; height: 500px;
            background: radial-gradient(circle, #4bc0ff 0%, transparent 70%);
            bottom: -120px; right: -100px;
            animation: orbB 16s ease-in-out infinite;
        }
        @keyframes orbA {
            0%,100% { transform: translate(0,0)   scale(1); }
            50%     { transform: translate(60vw, 30vh) scale(1.25); }
        }
        @keyframes orbB {
            0%,100% { transform: translate(0,0)    scale(1); }
            50%     { transform: translate(-50vw, -25vh) scale(1.15); }
        }

        /* Keep the actual content above the orbs */
        .main .block-container { position: relative; z-index: 1; }

        /* 3D stage for the heading */
        .landing-stage {
            perspective: 1200px;
            text-align: center;
            padding: 4rem 1rem 3rem 1rem;
            position: relative;
            z-index: 1;
        }
        .landing-title {
            font-size: 4.2rem;
            font-weight: 900;
            margin: 0;
            transform-style: preserve-3d;
            background: linear-gradient(135deg, #fff 0%, #ffd86b 35%, #ff7b3d 70%, #ff4b8b 100%);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            color: transparent;
            filter: drop-shadow(0 6px 24px rgba(255, 180, 75, 0.45))
                    drop-shadow(0 2px 0 rgba(0,0,0,0.25));
            animation: floatTitle 5s ease-in-out infinite,
                       hueSlide 7s ease-in-out infinite;
        }
        @keyframes hueSlide {
            0%,100% { background-position: 0%   50%; }
            50%     { background-position: 100% 50%; }
        }
        @keyframes floatTitle {
            0%   { transform: rotateX(8deg)  rotateY(-6deg) translateY(0); }
            50%  { transform: rotateX(-4deg) rotateY(6deg)  translateY(-12px); }
            100% { transform: rotateX(8deg)  rotateY(-6deg) translateY(0); }
        }

        /* 3D buttons */
        div[data-testid="stButton"] {
            perspective: 900px;
        }
        div[data-testid="stButton"] > button {
            height: 9rem;
            font-weight: 800 !important;
            border-radius: 18px !important;
            border: none !important;
            transform-style: preserve-3d;
            transition: transform 220ms ease, box-shadow 220ms ease, filter 220ms ease;
            box-shadow:
                0 14px 28px rgba(255, 75, 75, 0.35),
                0 8px 0 rgba(180, 30, 30, 0.45),
                inset 0 -6px 12px rgba(0,0,0,0.12),
                inset 0 2px 0 rgba(255,255,255,0.35);
            transform: translateY(0) rotateX(0) rotateY(0);
        }
        /* Streamlit wraps button label in a <p>; size that too */
        div[data-testid="stButton"] > button p {
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 6px rgba(0,0,0,0.25);
        }
        div[data-testid="stButton"] > button:hover {
            transform: translateY(-6px) rotateX(8deg) rotateY(-4deg) scale(1.02);
            box-shadow:
                0 22px 40px rgba(255, 75, 75, 0.45),
                0 12px 0 rgba(180, 30, 30, 0.5),
                inset 0 -6px 12px rgba(0,0,0,0.12),
                inset 0 2px 0 rgba(255,255,255,0.5);
            filter: brightness(1.05);
        }
        div[data-testid="stButton"] > button:active {
            transform: translateY(4px) rotateX(-2deg) scale(0.99);
            box-shadow:
                0 6px 14px rgba(255, 75, 75, 0.3),
                0 2px 0 rgba(180, 30, 30, 0.4),
                inset 0 -3px 8px rgba(0,0,0,0.18);
        }
        </style>
        <div class='landing-stage'>
            <h1 class='landing-title'>Are you Joining the Chit?</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _, c1, c2, _ = st.columns([1, 2, 2, 1])
    with c1:
        if st.button("New Chit", use_container_width=True, type="primary"):
            st.session_state["entry_type"] = "New Chit"
            st.rerun()
    with c2:
        if st.button("Middle Joining", use_container_width=True, type="primary"):
            st.session_state["entry_type"] = "Middle Joining"
            st.rerun()
    st.stop()


def format_inr(amount: float) -> str:
    """Format number in Indian Rupee style (lakhs/crores)."""
    if amount < 0:
        return "-" + format_inr(-amount)
    abs_val = abs(amount)
    if abs_val >= 1_00_00_000:
        return f"\u20b9{abs_val / 1_00_00_000:,.2f} Cr"
    if abs_val >= 1_00_000:
        return f"\u20b9{abs_val / 1_00_000:,.2f} L"
    return f"\u20b9{abs_val:,.0f}"


def _params_to_dict(p: ChitParams) -> dict:
    """Serialize ChitParams to a JSON-safe dict."""
    return {
        "chit_value": p.chit_value,
        "num_members": p.num_members,
        "commission_pct": p.commission_pct,
        "first_auction_payment": p.first_auction_payment,
        "last_payment": p.last_payment,
        "annual_discount_rate": p.annual_discount_rate,
        "start_date": p.start_date.isoformat(),
        "entry_month": p.entry_month,
        "till_date_payment": p.till_date_payment,
        "company_chit_months": list(p.company_chit_months),
    }


def _params_from_dict(d: dict) -> ChitParams:
    """Rebuild ChitParams from a saved dict."""
    return ChitParams(
        chit_value=float(d["chit_value"]),
        num_members=int(d["num_members"]),
        commission_pct=float(d["commission_pct"]),
        first_auction_payment=float(d.get("first_auction_payment", 0.0)),
        last_payment=float(d.get("last_payment", 0.0)),
        annual_discount_rate=float(d.get("annual_discount_rate", 12.0)),
        start_date=date.fromisoformat(d["start_date"]),
        entry_month=int(d.get("entry_month", 1)),
        till_date_payment=float(d.get("till_date_payment", 0.0)),
        company_chit_months=tuple(d.get("company_chit_months", (1, 3))),
    )


def load_saved_chits() -> list:
    """Load the list of saved chits from disk (empty list if none/corrupt)."""
    if SAVED_CHITS_PATH.exists():
        try:
            data = json.loads(SAVED_CHITS_PATH.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, OSError):
            return []
    return []


def write_saved_chits(chits: list) -> None:
    """Persist the list of saved chits to disk."""
    SAVED_CHITS_PATH.write_text(json.dumps(chits, indent=2), encoding="utf-8")


def save_chit(name: str, result) -> None:
    """Save (or overwrite by name) a chit's params and summary metrics."""
    chits = load_saved_chits()
    best = next(a for a in result.analyses if a.lift_month == result.optimal_month)
    entry = {
        "name": name,
        "saved_at": date.today().isoformat(),
        "params": _params_to_dict(result.params),
        "summary": {
            "optimal_month": result.optimal_month,
            "best_npv": best.npv,
            "best_xirr": best.xirr,
            "best_net_cost": best.net_cost,
        },
    }
    chits = [c for c in chits if c.get("name") != name]
    chits.append(entry)
    write_saved_chits(chits)


def _parse_image_chit(img: Image.Image, discount_rate: float = 12.0) -> "ChitParams | None":
    """Extract chit parameters from a screenshot using OCR."""
    try:
        text = pytesseract.image_to_string(img)
    except Exception:
        return None

    if not text.strip():
        return None

    lines = text.strip().split("\n")
    # Normalize: lowercase, collapse whitespace
    raw = "\n".join(lines)
    low = raw.lower()

    def find_number(pattern: str, txt: str = low) -> float | None:
        """Find a number following a label pattern."""
        m = re.search(pattern + r'[\s:]*[₹rs.]*\s*([\d,]+(?:\.\d+)?)', txt)
        if m:
            return float(m.group(1).replace(",", ""))
        return None

    def find_date_val(pattern: str, txt: str = raw) -> date | None:
        """Find a date following a label pattern."""
        # Try common date formats
        m = re.search(pattern + r'[\s:]*(\d{1,2}[-/]\w{3}[-/]\d{2,4}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})', txt, re.IGNORECASE)
        if m:
            date_str = m.group(1)
            for fmt in ("%d-%b-%y", "%d-%b-%Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y"):
                try:
                    return pd.to_datetime(date_str, format=fmt).date()
                except (ValueError, TypeError):
                    continue
            try:
                return pd.to_datetime(date_str).date()
            except (ValueError, TypeError):
                pass
        return None

    # Extract chit value
    chit_value = (
        find_number(r'chit\s*(?:amount|value)')
        or find_number(r'amount')
        or find_number(r'value')
    )
    if not chit_value:
        return None

    # Extract tenure / members
    tenure = find_number(r'tenure') or find_number(r'members') or find_number(r'duration')
    if tenure:
        tenure = int(tenure)
    else:
        tenure = 25  # default

    # Mid entry
    is_mid = bool(re.search(r'mid\s*entry[\s:]*yes', low))

    entry_month = 1
    till_date = 0.0
    if is_mid:
        em = find_number(r'start\s*month\s*(?:nos?|number)?') or find_number(r'entry\s*month')
        if em:
            entry_month = int(em)
        td = find_number(r'till\s*date\s*(?:payment)?')
        if td:
            till_date = td

    # Commission
    comm_match = re.search(r'commission[\s:]*(\d+(?:\.\d+)?)\s*%', low)
    commission_pct = float(comm_match.group(1)) if comm_match else 5.0

    # Max discount % (try to extract; default 30)
    disc_match = re.search(r'(?:max\s*|expected\s*)?discount[\s:]*(\d+(?:\.\d+)?)\s*%', low)
    max_discount_pct = float(disc_match.group(1)) if disc_match else 30.0

    # Start date
    start_dt = (
        find_date_val(r'start\s*(?:month\s*)?date')
        or find_date_val(r'start\s*date')
        or date.today()
    )

    first_auction = chit_value / tenure * (1 - max_discount_pct / 100.0)
    return ChitParams(
        chit_value=chit_value,
        num_members=tenure,
        commission_pct=commission_pct,
        first_auction_payment=first_auction,
        annual_discount_rate=discount_rate,
        start_date=start_dt,
        entry_month=max(1, entry_month),
        till_date_payment=till_date if is_mid else 0.0,
    )


def _parse_excel_chit(df: pd.DataFrame, discount_rate: float = 12.0) -> "ChitParams | None":
    """Try to parse chit parameters from an Excel sheet.

    Supports two formats:
    1. Metadata format: Row 0 has 'Chit Amount', rows 0-5 have params, row 7+ has matrix.
    2. Matrix-only format: Row 0 has 'Month' or dates, row 1+ has cash flow data.
    """
    if df.empty or df.shape[0] < 3:
        return None

    # Check for metadata format (LTH47/GXZ03 style)
    cell_00 = str(df.iloc[0, 0]).strip() if pd.notna(df.iloc[0, 0]) else ""
    if "chit amount" in cell_00.lower():
        try:
            chit_value = float(df.iloc[0, 1])
            tenure = int(df.iloc[1, 1])

            mid_entry_val = str(df.iloc[2, 1]).strip().upper() if pd.notna(df.iloc[2, 1]) else "NO"
            is_mid = mid_entry_val == "YES"

            entry_month = int(df.iloc[3, 1]) if is_mid and pd.notna(df.iloc[3, 1]) else 1
            if entry_month < 1:
                entry_month = 1

            till_date = float(df.iloc[4, 1]) if is_mid and pd.notna(df.iloc[4, 1]) else 0.0

            start_dt = pd.to_datetime(df.iloc[5, 1]).date() if pd.notna(df.iloc[5, 1]) else date.today()

            # Check for commission in col 3/4 or derive from data
            commission_pct = 5.0  # default
            # Try to find the last month's prize from the matrix to derive commission
            # Find the header row with "Opt" columns
            header_row = None
            for i in range(min(10, len(df))):
                row_vals = [str(v) for v in df.iloc[i] if pd.notna(v)]
                if any("Opt" in v for v in row_vals):
                    header_row = i
                    break

            if header_row is not None and header_row + 1 < len(df):
                # Count option columns to verify tenure
                opt_cols = [v for v in df.iloc[header_row] if pd.notna(v) and str(v).startswith("Opt")]
                num_opts = len(opt_cols)

                # Try to find last data row to get last month's net flow
                # Last option column in last data row = prize_N - payment_N = V - commission - payment_N
                # When lifting in last month: all payments are regular, prize = V - commission (div=0)
                last_data_row = header_row + num_opts
                if last_data_row < len(df):
                    last_opt_col_idx = None
                    for c in range(df.shape[1] - 1, 0, -1):
                        val = df.iloc[header_row, c]
                        if pd.notna(val) and str(val).startswith("Opt"):
                            last_opt_col_idx = c
                            break
                    if last_opt_col_idx is not None:
                        # Get the prize value from the last lifting option's lift month
                        lift_val = df.iloc[last_data_row, last_opt_col_idx]
                        if pd.notna(lift_val):
                            try:
                                lift_net = float(lift_val)
                                # In last month of last option: net = prize - payment
                                # payment_N = base_sub (div=0), prize_N = V - commission
                                # But we need another equation. Use first month data instead.
                                pass
                            except (ValueError, TypeError):
                                pass

            # Try to estimate discount % from first month data
            expected_discount_pct = 30.0  # default
            if header_row is not None:
                first_data_row = header_row + 1
                if first_data_row < len(df):
                    # First option column, first data row = prize_entry - till_date (for mid) or prize_1 - payment_1
                    first_opt_col = None
                    for c in range(df.shape[1]):
                        val = df.iloc[header_row, c]
                        if pd.notna(val) and str(val).startswith("Opt"):
                            first_opt_col = c
                            break
                    if first_opt_col is not None:
                        lift_net_1 = df.iloc[first_data_row, first_opt_col]
                        # Non-lift value from next row, same column = -payment for month after entry
                        if first_data_row + 1 < len(df):
                            non_lift = df.iloc[first_data_row + 1, first_opt_col]
                            if pd.notna(lift_net_1) and pd.notna(non_lift):
                                try:
                                    payment_after = abs(float(non_lift))
                                    net_lift = float(lift_net_1)
                                    if is_mid and till_date > 0:
                                        prize_1 = net_lift + till_date
                                    else:
                                        prize_1 = net_lift + payment_after
                                    discount_1 = chit_value - prize_1
                                    if discount_1 > 0:
                                        expected_discount_pct = round(discount_1 / chit_value * 100, 1)
                                        # Commission = base_sub - payment_after = dividend
                                        # dividend_1 * N + commission = discount_1
                                        base_sub = chit_value / tenure
                                        dividend_1 = base_sub - payment_after
                                        if dividend_1 > 0:
                                            # commission = discount_1 - dividend_1 * N
                                            comm = discount_1 - dividend_1 * tenure
                                            if 0 < comm < chit_value * 0.2:
                                                commission_pct = round(comm / chit_value * 100, 1)
                                except (ValueError, TypeError):
                                    pass

            first_auction = chit_value / tenure * (1 - expected_discount_pct / 100.0)
            return ChitParams(
                chit_value=chit_value,
                num_members=tenure,
                commission_pct=commission_pct,
                first_auction_payment=first_auction,
                annual_discount_rate=discount_rate,
                start_date=start_dt,
                entry_month=max(1, entry_month),
                till_date_payment=till_date if is_mid else 0.0,
            )
        except (ValueError, TypeError, IndexError):
            return None

    # Check for matrix-only format (LT007C style): row 0 has "Month" or dates
    if "month" in cell_00.lower() or isinstance(df.iloc[0, 0], pd.Timestamp):
        try:
            # Count data rows (each row = one month in the chit)
            # Find how many rows have date values in column 0
            n_months = 0
            for i in range(1, len(df)):
                val = df.iloc[i, 0]
                if pd.notna(val):
                    try:
                        pd.to_datetime(val)
                        n_months += 1
                    except (ValueError, TypeError):
                        break
                else:
                    break

            if n_months < 2:
                return None

            # Get dates from column headers (row 0)
            dates = []
            for c in range(1, df.shape[1]):
                val = df.iloc[0, c]
                if pd.notna(val):
                    try:
                        dates.append(pd.to_datetime(val).date())
                    except (ValueError, TypeError):
                        break

            chit_value_est = 0
            # Estimate chit value from first row: all values should be the same negative number
            first_row_vals = []
            for c in range(1, min(df.shape[1], len(dates) + 1)):
                val = df.iloc[1, c]
                if pd.notna(val):
                    try:
                        first_row_vals.append(float(val))
                    except (ValueError, TypeError):
                        pass

            if first_row_vals:
                # First row might have a positive (lift) value
                negatives = [v for v in first_row_vals if v < 0]
                positives = [v for v in first_row_vals if v > 0]
                if negatives and positives:
                    payment_1 = abs(negatives[0])
                    prize_1 = positives[0] + payment_1
                    # chit_value ≈ prize_1 / 0.7 (assuming 30% discount)
                    chit_value_est = round(prize_1 / 0.7 / 100000) * 100000

            if chit_value_est > 0:
                return ChitParams(
                    chit_value=chit_value_est,
                    num_members=n_months,
                    commission_pct=5.0,
                    first_auction_payment=chit_value_est / n_months * 0.70,
                    annual_discount_rate=discount_rate,
                    start_date=dates[0] if dates else date.today(),
                )
        except (ValueError, TypeError, IndexError):
            return None

    return None


def _render_uploaded_data():
    """Display uploaded chit documents (images and Excel files)."""
    files = st.session_state.get("uploaded_files", [])
    if not files:
        st.info("No files uploaded yet.")
        return

    for f in files:
        ext = f.name.rsplit(".", 1)[-1].lower()
        st.markdown(f"**{f.name}**")

        if ext in ("jpg", "jpeg", "png"):
            img = Image.open(BytesIO(f.getvalue()))
            st.image(img, caption=f.name, use_container_width=True)

        elif ext in ("xlsx", "xls"):
            try:
                xls = pd.ExcelFile(BytesIO(f.getvalue()))
                sheet_names = xls.sheet_names
                if len(sheet_names) > 1:
                    sheet = st.selectbox(
                        f"Sheet — {f.name}", sheet_names, key=f"sheet_{f.name}"
                    )
                else:
                    sheet = sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet)
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.download_button(
                    f"Download {f.name} as CSV",
                    df.to_csv(index=False),
                    file_name=f.name.rsplit(".", 1)[0] + ".csv",
                    mime="text/csv",
                    key=f"dl_{f.name}",
                )
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")

        st.divider()


def _render_saved_chits():
    """Show saved chits in a comparison table, flagging similar-value chits."""
    chits = load_saved_chits()
    if not chits:
        st.info(
            "No saved chits yet. Run an analysis and click "
            "**Save this chit** in the Summary tab to store it for later comparison."
        )
        return

    current = st.session_state.get("result")
    current_value = current.params.chit_value if current else None
    current_members = current.params.num_members if current else None

    rows = []
    for c in chits:
        p = c.get("params", {})
        s = c.get("summary", {})
        cv = float(p.get("chit_value", 0) or 0)
        members = int(p.get("num_members", 0))
        xirr = s.get("best_xirr")
        similar = bool(
            current_value
            and cv > 0
            and abs(cv - current_value) / current_value <= SIMILAR_VALUE_TOLERANCE
            and current_members
            and abs(members - current_members) <= SIMILAR_TENURE_TOLERANCE
        )
        rows.append({
            "Name": c.get("name", ""),
            "Saved": c.get("saved_at", ""),
            "Chit Value": cv,
            "Members": int(p.get("num_members", 0)),
            "Entry": int(p.get("entry_month", 1)),
            "Commission %": p.get("commission_pct", ""),
            "Optimal Month": s.get("optimal_month", ""),
            "Best NPV": s.get("best_npv", 0.0),
            "Best XIRR %": round(xirr * 100, 1) if xirr is not None else None,
            "Similar": similar,
        })

    df = pd.DataFrame(rows).sort_values("Chit Value", ascending=False).reset_index(drop=True)

    if current_value:
        st.caption(
            f"Highlighted rows are within ±{SIMILAR_VALUE_TOLERANCE * 100:.0f}% of the current "
            f"value ({format_inr(current_value)}) **and** ±{SIMILAR_TENURE_TOLERANCE} months of "
            f"its tenure ({current_members} members)."
        )

    similar_mask = df["Similar"]
    display = df.drop(columns=["Similar"]).copy()
    display["Chit Value"] = df["Chit Value"].apply(format_inr)
    display["Best NPV"] = df["Best NPV"].apply(
        lambda x: format_inr(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
    )
    display["Best XIRR %"] = df["Best XIRR %"].apply(
        lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) and not isinstance(x, bool) else "N/A"
    )

    similar_style = "background-color: #d1e7dd; color: #0f5132; font-weight: 600;"

    def _hl(row):
        return [similar_style if similar_mask.loc[row.name] else ""] * len(row)

    st.dataframe(display.style.apply(_hl, axis=1), use_container_width=True, hide_index=True)

    names = [c.get("name", "") for c in chits]
    csel1, csel2, csel3 = st.columns([2, 1, 1])
    with csel1:
        sel = st.selectbox("Select a saved chit", names, key="saved_chit_sel")
    with csel2:
        if st.button("Re-analyze", use_container_width=True, key="saved_chit_reanalyze"):
            entry = next(c for c in chits if c.get("name") == sel)
            st.session_state["result"] = analyze_chit(_params_from_dict(entry["params"]))
            st.rerun()
    with csel3:
        if st.button("Delete", use_container_width=True, key="saved_chit_delete"):
            write_saved_chits([c for c in chits if c.get("name") != sel])
            st.rerun()

    st.download_button(
        "Download comparison as CSV",
        df.drop(columns=["Similar"]).to_csv(index=False),
        file_name="saved_chits_comparison.csv",
        mime="text/csv",
        key="saved_chit_csv",
    )


def _render_finding(f) -> None:
    """Render a single ReportFinding as a coloured callout."""
    body = f"**{f.title}** — {f.detail}"
    if f.severity == "good":
        st.success(body)
    elif f.severity == "warn":
        st.warning(body)
    elif f.severity == "bad":
        st.error(body)
    else:
        st.info(body)


def _report_to_markdown(result, report) -> str:
    """Serialize the analysis report to downloadable markdown."""
    p = result.params
    xirr = f"{report.best_xirr * 100:.1f}%" if report.best_xirr is not None else "N/A"
    lines = [
        "# Chit Fund Analysis Report",
        "",
        f"- **Chit value:** {format_inr(p.chit_value)}",
        f"- **Tenure:** {p.num_members} months",
        f"- **Commission:** {p.commission_pct}%",
        f"- **Required return:** {report.required_rate:.1f}%",
    ]
    if p.entry_month > 1:
        lines.append(f"- **Entry month:** {p.entry_month} (mid-joining)")
        lines.append(f"- **Till-date payment:** {format_inr(p.till_date_payment)}")
    lines += [
        "",
        f"## Decision: {report.decision}",
        "",
        report.verdict,
        "",
        f"**Best month:** {report.best_month} | **Best NPV:** "
        f"{format_inr(report.best_npv)} | **Best XIRR:** {xirr}",
        "",
        "## When to bid vs. stay",
        "",
    ]
    for f in report.timing:
        lines.append(f"- **{f.title}** — {f.detail}")
    lines += ["", "## Was it fairly done?", "", f"_{report.fairness_verdict}_", ""]
    for f in report.fairness:
        lines.append(f"- **{f.title}** — {f.detail}")
    return "\n".join(lines)


def _render_analysis_report(result) -> None:
    """Render the enter/avoid + timing + fairness advisory."""
    report = build_analysis_report(result)

    if report.decision == "ENTER":
        st.success(f"### ✅ Decision: ENTER\n\n{report.verdict}")
    else:
        st.error(f"### ⛔ Decision: AVOID\n\n{report.verdict}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Lifting Month", f"Month {report.best_month}")
    c2.metric("Best NPV", format_inr(report.best_npv))
    c3.metric(
        "Best XIRR",
        f"{report.best_xirr * 100:.1f}%" if report.best_xirr is not None else "N/A",
    )

    st.divider()
    st.subheader("⏱️ When to bid vs. stay till the end")
    for f in report.timing:
        _render_finding(f)

    st.divider()
    st.subheader("⚖️ Was it fairly done?")
    st.markdown(f"**{report.fairness_verdict}**")
    for f in report.fairness:
        _render_finding(f)

    st.divider()
    st.download_button(
        "Download report (Markdown)",
        _report_to_markdown(result, report),
        file_name="chit_analysis_report.md",
        mime="text/markdown",
        key="report_md_dl",
    )


def _build_cash_flow_matrix(result) -> pd.DataFrame:
    """Build a matrix: rows=months, columns=lifting options, values=net cash flow."""
    rows = []
    for month_idx, flow in enumerate(result.analyses[0].cash_flows):
        month_label = f"{flow.month}*" if flow.is_reference else flow.month
        row = {"Month": month_label, "Date": flow.date.strftime("%b %Y")}
        for analysis in result.analyses:
            col_name = f"Opt {analysis.lift_month}"
            row[col_name] = analysis.cash_flows[month_idx].net_flow
        rows.append(row)

    # Add totals row (reference/completed months are excluded)
    totals = {"Month": "", "Date": "Net Payment"}
    for analysis in result.analyses:
        col_name = f"Opt {analysis.lift_month}"
        totals[col_name] = sum(
            f.net_flow for f in analysis.cash_flows if not f.is_reference
        )
    rows.append(totals)

    return pd.DataFrame(rows)


def _style_matrix(df: pd.DataFrame):
    """Format the matrix values as INR for display, highlighting summary rows."""
    display_df = df.copy()
    opt_cols = [c for c in df.columns if c.startswith("Opt ")]
    for col in opt_cols:
        display_df[col] = df[col].apply(
            lambda x: format_inr(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x
        )

    highlight = {"Net Payment", "XNPV", "XIRR"}

    best_xnpv_col = None
    xnpv_mask = df["Date"] == "XNPV"
    if xnpv_mask.any() and opt_cols:
        numeric = pd.to_numeric(df.loc[xnpv_mask, opt_cols].iloc[0], errors="coerce")
        if numeric.notna().any():
            best_xnpv_col = numeric.idxmax()

    best_net_pmt_col = None
    net_mask = df["Date"] == "Net Payment"
    if net_mask.any() and opt_cols:
        numeric = pd.to_numeric(df.loc[net_mask, opt_cols].iloc[0], errors="coerce")
        if numeric.notna().any():
            best_net_pmt_col = numeric.idxmax()

    best_xirr_col = None
    xirr_mask = df["Date"] == "XIRR"
    if xirr_mask.any() and opt_cols:
        def _parse_pct(v):
            if isinstance(v, str) and v.endswith("%"):
                try:
                    return float(v[:-1])
                except ValueError:
                    return None
            return None
        parsed = df.loc[xirr_mask, opt_cols].iloc[0].map(_parse_pct).dropna()
        if not parsed.empty:
            best_xirr_col = parsed.idxmax()

    base_style = "background-color: #fff3cd; font-weight: 700; color: #000;"
    best_style = (
        "background-color: #2e7d32; font-weight: 800; color: #fff; "
        "border: 2px solid #1b5e20;"
    )
    inflow_style = "background-color: #d1e7dd; color: #0f5132;"

    ref_style = "background-color: #e9ecef; color: #495057; font-style: italic;"

    def row_style(row):
        styles = [""] * len(row)
        label = row.get("Date")
        if label in highlight:
            styles = [base_style] * len(row)
            if label == "XNPV" and best_xnpv_col is not None:
                styles[row.index.get_loc(best_xnpv_col)] = best_style
            elif label == "XIRR" and best_xirr_col is not None:
                styles[row.index.get_loc(best_xirr_col)] = best_style
            elif label == "Net Payment" and best_net_pmt_col is not None:
                styles[row.index.get_loc(best_net_pmt_col)] = best_style
            return styles

        # Completed-month reference row (marked with a trailing '*')
        if str(row.get("Month", "")).endswith("*"):
            return [ref_style] * len(row)

        # Per-month rows: highlight cash-inflow cells (positive net flow)
        for col in opt_cols:
            val = df.loc[row.name, col]
            if isinstance(val, (int, float)) and not isinstance(val, bool) and val > 0:
                styles[row.index.get_loc(col)] = inflow_style
        return styles

    return display_df.style.apply(row_style, axis=1)


# ============================================================
# SIDEBAR — Step-by-step workflow
# ============================================================
st.sidebar.header("Chit Fund Parameters")

# --- Step 1: Entry Type ---
_options = ["New Chit", "Middle Joining"]
_default_idx = _options.index(st.session_state.get("entry_type", "New Chit"))
entry_type = st.sidebar.radio(
    "How are you entering this chit?",
    _options,
    index=_default_idx,
    horizontal=True,
)
st.session_state["entry_type"] = entry_type
if st.sidebar.button("← Back to start", use_container_width=True):
    for k in ("entry_type", "result", "uploaded_files", "upload_parsed_params", "ocr_text"):
        st.session_state.pop(k, None)
    st.rerun()

st.sidebar.divider()

# --- Step 2: Fields based on entry type ---
if entry_type == "New Chit":
    # ---- New Chit inputs ----
    st.sidebar.subheader("Chit Details")
    chit_value = st.sidebar.number_input(
        "Chit Value (\u20b9)", min_value=10_000, value=500_000, step=10_000
    )
    num_members = st.sidebar.number_input(
        "Members / Duration (months)", min_value=2, max_value=100, value=25
    )
    commission_pct = st.sidebar.number_input(
        "Commission %", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )
    start_date = st.sidebar.date_input("Start Date", value=date.today())
    company_chits_str = st.sidebar.text_input(
        "Company Chit Months (comma-separated)",
        value="1,3",
        help="Months where the company/foreman takes the chit. No dividend, member pays full V/N.",
    )
    annual_discount_rate = st.sidebar.number_input(
        "XNPV Rate (% per year)",
        min_value=0.0,
        max_value=50.0,
        value=12.0,
        step=0.5,
        help="Required rate of return used to discount cash flows for XNPV.",
    )

    # Middle entry fields (not applicable)
    entry_month = 1
    till_date_payment = 0.0

else:
    # ---- Middle Joining inputs ----
    st.sidebar.subheader("Chit Details")
    chit_value = st.sidebar.number_input(
        "Chit Amount (\u20b9)", min_value=10_000, value=3_000_000, step=100_000
    )
    num_members = st.sidebar.number_input(
        "Tenure (months)", min_value=2, max_value=100, value=50
    )
    commission_pct = st.sidebar.number_input(
        "Commission %", min_value=0.0, max_value=20.0, value=5.0, step=0.5
    )

    st.sidebar.subheader("Your Entry Details")
    months_completed = st.sidebar.number_input(
        "Months completed till date",
        min_value=1,
        max_value=num_members - 1,
        value=min(10, num_members - 1),
        help="Number of months already elapsed before you join. Your start month = this + 1.",
    )
    entry_month = months_completed + 1
    till_date_payment = st.sidebar.number_input(
        "Till Date Payment (\u20b9)", min_value=0.0, value=0.0, step=50_000.0,
        help="Total lump-sum you pay to catch up for missed months"
    )
    start_date = st.sidebar.date_input("Chit Start Month", value=date.today())
    company_chits_str = st.sidebar.text_input(
        "Company Chit Months (comma-separated)",
        value="1,3",
        help="Months where the company/foreman takes the chit. No dividend, member pays full V/N.",
    )
    annual_discount_rate = st.sidebar.number_input(
        "XNPV Rate (% per year)",
        min_value=0.0,
        max_value=50.0,
        value=12.0,
        step=0.5,
        help="Required rate of return used to discount cash flows for XNPV.",
    )

    # Auto-calculate and display end date
    remaining_months = num_members - entry_month
    end_date = start_date + relativedelta(months=remaining_months)
    st.sidebar.text_input("End Date (auto-calculated)", value=end_date.strftime("%d-%b-%Y"), disabled=True)

st.sidebar.divider()

# --- Step 3: First & last outflow (drives the per-month payment schedule) ---
_base_sub = float(chit_value) / int(num_members)
first_auction_payment = st.sidebar.number_input(
    "Last Auction Chit Amount (\u20b9)",
    min_value=0.0,
    max_value=float(_base_sub) * 2,
    value=float(round(_base_sub * 0.65)),
    step=100.0,
    help=(
        "Member payment at the most recent auction. "
        "For New Chit this is the first non-company month; "
        "for Middle Joining this is your entry month."
    ),
)
last_payment = st.sidebar.number_input(
    "Last Month Outflow (\u20b9)",
    min_value=0.0,
    max_value=float(_base_sub) * 2,
    value=float(_base_sub),
    step=100.0,
    help=f"Member payment at the last month. Defaults to V/N = \u20b9{_base_sub:,.0f}.",
)
st.sidebar.caption(
    f"V/N = \u20b9{_base_sub:,.0f} \u2014 implied last-auction discount "
    f"= {(_base_sub - first_auction_payment) / _base_sub * 100:.1f}%"
)

st.sidebar.divider()

# --- Step 4: Analyze ---
analyze_clicked = st.sidebar.button("Analyze", type="primary", use_container_width=True)

# --- Upload Section ---
st.sidebar.divider()
st.sidebar.header("Upload Existing Chit")
uploaded_file = st.sidebar.file_uploader(
    "Upload chit file (Excel or Image)",
    type=["xlsx", "xls", "png", "jpg", "jpeg"],
)

upload_sheet = None
upload_analyze_clicked = False
upload_is_image = False
if uploaded_file:
    st.session_state["uploaded_files"] = [uploaded_file]
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext in ("png", "jpg", "jpeg"):
        upload_is_image = True
        upload_img = Image.open(BytesIO(uploaded_file.getvalue()))
        st.sidebar.image(
            upload_img,
            caption=uploaded_file.name,
            use_container_width=True,
        )
        upload_analyze_clicked = st.sidebar.button(
            "Analyze Upload", type="secondary", use_container_width=True
        )
    else:
        try:
            xls = pd.ExcelFile(BytesIO(uploaded_file.getvalue()))
            if len(xls.sheet_names) > 1:
                upload_sheet = st.sidebar.selectbox(
                    "Select Sheet (each sheet = a chit)",
                    xls.sheet_names,
                    key="upload_sheet_select",
                )
            else:
                upload_sheet = xls.sheet_names[0]
            st.sidebar.caption(f"Selected: **{upload_sheet}**")
            upload_analyze_clicked = st.sidebar.button(
                "Analyze Upload", type="secondary", use_container_width=True
            )
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")


# ============================================================
# MAIN AREA
# ============================================================
_back_col, _title_col = st.columns([1, 9])
with _back_col:
    if st.button("← Back", use_container_width=True, key="main_back"):
        for k in ("entry_type", "result", "uploaded_files", "upload_parsed_params", "ocr_text"):
            st.session_state.pop(k, None)
        st.rerun()
with _title_col:
    st.title("Chit Fund Analyzer")

# --- Validation & Analysis ---
if analyze_clicked:
    errors = []
    if entry_type == "Middle Joining" and till_date_payment <= 0:
        errors.append("Till Date Payment must be greater than zero for middle joining.")

    company_months: tuple[int, ...] = ()
    raw = (company_chits_str or "").strip()
    if raw:
        try:
            company_months = tuple(
                sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
            )
            if any(m < 1 or m > num_members for m in company_months):
                errors.append(f"Company Chit Months must be between 1 and {num_members}.")
        except ValueError:
            errors.append("Company Chit Months must be comma-separated integers (e.g. 1,3).")

    if errors:
        for err in errors:
            st.error(err)
    else:
        params = ChitParams(
            chit_value=float(chit_value),
            num_members=int(num_members),
            commission_pct=float(commission_pct),
            first_auction_payment=float(first_auction_payment),
            last_payment=float(last_payment),
            annual_discount_rate=float(annual_discount_rate),
            start_date=start_date,
            entry_month=int(entry_month),
            till_date_payment=float(till_date_payment),
            company_chit_months=company_months,
        )
        st.session_state["result"] = analyze_chit(params)

# --- Upload Analysis ---
if upload_analyze_clicked and uploaded_file and upload_is_image:
    # OCR: extract chit parameters from the screenshot
    parsed = _parse_image_chit(upload_img, discount_rate=annual_discount_rate)
    if parsed is not None:
        st.session_state["result"] = analyze_chit(parsed)
        st.session_state["upload_parsed_params"] = parsed
        # Show extracted text for transparency
        try:
            ocr_text = pytesseract.image_to_string(upload_img)
            st.session_state["ocr_text"] = ocr_text
        except Exception:
            pass
        st.toast(
            f"Parsed: V={format_inr(parsed.chit_value)} | N={parsed.num_members} | "
            f"Commission={parsed.commission_pct}% | First Outflow={format_inr(parsed.first_auction_payment)} | "
            f"Entry={parsed.entry_month}",
            icon="\u2705",
        )
    else:
        try:
            ocr_text = pytesseract.image_to_string(upload_img)
            st.error(
                f"Could not extract chit parameters from the screenshot. "
                f"OCR extracted text:\n```\n{ocr_text}\n```\n"
                f"Please enter parameters manually in the sidebar above."
            )
        except Exception as e:
            st.error(f"OCR failed: {e}. Please enter parameters manually.")

elif upload_analyze_clicked and uploaded_file and upload_sheet:
    try:
        xls = pd.ExcelFile(BytesIO(uploaded_file.getvalue()))
        raw_df = pd.read_excel(xls, sheet_name=upload_sheet, header=None)
        parsed = _parse_excel_chit(raw_df, discount_rate=annual_discount_rate)
        if parsed is not None:
            st.session_state["result"] = analyze_chit(parsed)
            st.session_state["upload_parsed_params"] = parsed
            st.toast(
                f"Parsed: {upload_sheet} | V={format_inr(parsed.chit_value)} | "
                f"N={parsed.num_members} | Entry={parsed.entry_month}",
                icon="\u2705",
            )
        else:
            st.error(
                f"Could not parse chit parameters from sheet '{upload_sheet}'. "
                "Please enter parameters manually in the sidebar."
            )
    except Exception as e:
        st.error(f"Error analyzing upload: {e}")

# --- Display Results ---
if "result" in st.session_state:
    result = st.session_state["result"]
    best_idx = next(
        i for i, a in enumerate(result.analyses) if a.lift_month == result.optimal_month
    )
    best = result.analyses[best_idx]

    tab_names = [
        "Summary",
        "Analysis Report",
        "Cash Flow Detail",
        "Cash Flow Matrix",
        "Comparison Table",
    ]
    has_uploads = "uploaded_files" in st.session_state and st.session_state["uploaded_files"]
    if has_uploads:
        tab_names.append("Uploaded Data")
    tab_names.append("Saved Chits")
    saved_tab_idx = tab_names.index("Saved Chits")
    report_tab_idx = tab_names.index("Analysis Report")
    tabs = st.tabs(tab_names)

    # --- Tab 1: Summary ---
    with tabs[0]:
        # Show input summary
        if result.params.entry_month > 1:
            st.caption(
                f"Middle Joining | "
                f"Amount: {format_inr(result.params.chit_value)} | "
                f"Tenure: {result.params.num_members} months | "
                f"Entry at Month {result.params.entry_month} | "
                f"Till Date: {format_inr(result.params.till_date_payment)}"
            )
        else:
            st.caption(
                f"New Chit | "
                f"Value: {format_inr(result.params.chit_value)} | "
                f"Members: {result.params.num_members} | "
                f"Commission: {result.params.commission_pct}%"
            )

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal Lifting Month", f"Month {result.optimal_month}")
        col2.metric("Best NPV", format_inr(best.npv))
        xirr_display = f"{best.xirr * 100:.1f}%" if best.xirr is not None else "N/A"
        col3.metric("Best XIRR", xirr_display)

        # Show OCR extracted text if from image upload
        if "ocr_text" in st.session_state and "upload_parsed_params" in st.session_state:
            with st.expander("OCR Extracted Text (from screenshot)"):
                st.code(st.session_state["ocr_text"])
                p = st.session_state["upload_parsed_params"]
                st.markdown(
                    f"**Parsed Values:** Chit Value={format_inr(p.chit_value)}, "
                    f"Tenure={p.num_members}, Commission={p.commission_pct}%, "
                    f"Prize={format_inr(p.fixed_prize)}, "
                    f"Monthly Payment={format_inr(p.fixed_monthly_payment)}, "
                    f"Mid Entry={'YES' if p.entry_month > 1 else 'NO'}"
                    + (f", Entry Month={p.entry_month}, Till Date={format_inr(p.till_date_payment)}" if p.entry_month > 1 else "")
                )

        if best.npv > 0:
            st.success(result.recommendation)
        else:
            st.warning(result.recommendation)

        # --- Save this chit for later comparison ---
        with st.expander("\U0001f4be Save this chit for later comparison"):
            default_name = (
                f"Chit {format_inr(result.params.chit_value)} / "
                f"{result.params.num_members}m"
            )
            existing_names = {c.get("name") for c in load_saved_chits()}
            save_name = st.text_input("Name", value=default_name, key="save_chit_name")
            if save_name.strip() in existing_names:
                st.caption(f"⚠️ A chit named '{save_name.strip()}' exists — saving will overwrite it.")
            if st.button("Save this chit", key="save_chit_btn", type="primary"):
                if save_name.strip():
                    save_chit(save_name.strip(), result)
                    st.success(f"Saved '{save_name.strip()}'. See the **Saved Chits** tab to compare.")
                else:
                    st.warning("Please enter a name.")

        # NPV chart
        npv_data = pd.DataFrame({
            "Lifting Month": [a.lift_month for a in result.analyses],
            "NPV (\u20b9)": [a.npv for a in result.analyses],
        })
        st.subheader("NPV by Lifting Month")
        st.line_chart(npv_data, x="Lifting Month", y="NPV (\u20b9)")

        # XIRR chart
        xirr_data = pd.DataFrame({
            "Lifting Month": [a.lift_month for a in result.analyses],
            "XIRR (%)": [a.xirr * 100 if a.xirr is not None else None for a in result.analyses],
        })
        st.subheader("XIRR by Lifting Month")
        st.line_chart(xirr_data, x="Lifting Month", y="XIRR (%)")

    # --- Tab: Analysis Report ---
    with tabs[report_tab_idx]:
        _render_analysis_report(result)

    # --- Tab: Cash Flow Detail ---
    with tabs[2]:
        lift_options = [a.lift_month for a in result.analyses]
        selected_month = st.selectbox(
            "Select Lifting Month",
            lift_options,
            index=best_idx,
        )
        sel_idx = next(i for i, a in enumerate(result.analyses) if a.lift_month == selected_month)
        analysis = result.analyses[sel_idx]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Paid", format_inr(analysis.total_paid))
        col2.metric("Prize Received", format_inr(analysis.prize_received))
        col3.metric("Net Cost", format_inr(analysis.net_cost))

        col4, col5 = st.columns(2)
        col4.metric("NPV", format_inr(analysis.npv))
        xirr_val = f"{analysis.xirr * 100:.1f}%" if analysis.xirr is not None else "N/A"
        col5.metric("XIRR", xirr_val)

        # Cash flow table
        cf_df = pd.DataFrame([
            {
                "Month": f"{f.month}*" if f.is_reference else f.month,
                "Date": f.date.strftime("%b %Y"),
                "Payment": "-" if f.is_reference else format_inr(abs(f.payment)),
                "Prize": format_inr(f.prize) if f.prize > 0 else "-",
                "Net Flow": "-" if f.is_reference else format_inr(f.net_flow),
                "Cumulative": "-" if f.is_reference else format_inr(f.cumulative),
            }
            for f in analysis.cash_flows
        ])
        if any(f.is_reference for f in analysis.cash_flows):
            st.caption(
                "\\* Last completed auction, shown for reference — its prize is "
                "displayed but it is excluded from your cash flow, NPV and XIRR."
            )
        if result.params.entry_month > 1 and result.params.till_date_payment > 0:
            st.caption(
                f"Entry month ({result.params.entry_month}) payment = till-date "
                f"catch-up ({format_inr(result.params.till_date_payment)}) + that "
                "month's own projected chit subscription."
            )
        st.subheader("Monthly Cash Flows")
        st.dataframe(cf_df, use_container_width=True, hide_index=True)

        # Cumulative chart
        cum_df = pd.DataFrame({
            "Month": [f.month for f in analysis.cash_flows],
            "Cumulative (\u20b9)": [f.cumulative for f in analysis.cash_flows],
        })
        st.subheader("Cumulative Cash Flow")
        st.area_chart(cum_df, x="Month", y="Cumulative (\u20b9)")

    # --- Tab: Cash Flow Matrix ---
    with tabs[3]:
        p = result.params
        base_sub = p.chit_value / p.num_members
        commission_amt = p.commission_pct / 100.0 * p.chit_value

        payments, prizes = compute_schedule(p)
        _start_idx = max(0, p.entry_month - 1)
        first_pmt, last_pmt = payments[_start_idx], payments[-1]
        first_prize, last_prize = prizes[_start_idx], prizes[-1]

        remaining = p.num_members - p.entry_month
        end_date = p.start_date + relativedelta(months=remaining)

        h1, h2 = st.columns(2)
        with h1:
            st.markdown(f"**Chit Amount:** {format_inr(p.chit_value)}")
            st.markdown(f"**Tenure:** {p.num_members}")
            st.markdown(f"**Mid Entry:** {'YES' if p.entry_month > 1 else 'NO'}")
            if p.entry_month > 1:
                st.markdown(f"**Start Month Nos:** {p.entry_month}")
                st.markdown(f"**Till Date Payment:** {format_inr(p.till_date_payment)}")
            st.markdown(f"**Chit Start Month:** {p.start_date.strftime('%d-%b-%y')}")
            st.markdown(f"**End Date:** {end_date.strftime('%d-%b-%y')}")
        with h2:
            st.markdown(f"**Base Subscription:** {format_inr(base_sub)}")
            st.markdown(f"**Commission:** {p.commission_pct}% ({format_inr(commission_amt)})")
            _implied_d = (p.chit_value / p.num_members - p.first_auction_payment) / (p.chit_value / p.num_members) * 100
            st.markdown(
                f"**Last Auction Chit Amount:** {format_inr(p.first_auction_payment)} "
                f"(≈ {_implied_d:.1f}% discount)"
            )
            st.markdown(f"**Last Month Outflow:** {format_inr(p.last_payment)}")
            _start_m = max(1, p.entry_month)
            st.markdown(f"**Prize (Month {_start_m} → {p.num_members}):** {format_inr(first_prize)} → {format_inr(last_prize)}")
            st.markdown(f"**Monthly Payment (Month {_start_m} → {p.num_members}):** {format_inr(first_pmt)} → {format_inr(last_pmt)}")

        st.divider()

        # --- Cash flow matrix ---
        st.subheader("Cash Flow Matrix")
        if p.entry_month > 1:
            st.caption(
                "Rows marked with \\* are the last completed auction (month "
                f"{p.entry_month - 1}), shown for reference with the prize "
                "distributed that month. They are excluded from Net Payment, "
                "XNPV and XIRR."
            )
        matrix_df = _build_cash_flow_matrix(result)

        # Add XNPV row
        xnpv_row = {"Month": "", "Date": "XNPV"}
        for a in result.analyses:
            xnpv_row[f"Opt {a.lift_month}"] = a.npv
        # Add XIRR row
        xirr_row = {"Month": "", "Date": "XIRR"}
        for a in result.analyses:
            if a.xirr is not None:
                xirr_row[f"Opt {a.lift_month}"] = f"{a.xirr * 100:.2f}%"
            else:
                xirr_row[f"Opt {a.lift_month}"] = "N/A"

        # Append summary rows to matrix
        xnpv_df = pd.DataFrame([xnpv_row])
        full_matrix = pd.concat([matrix_df, xnpv_df], ignore_index=True)
        # XIRR row has mixed types (string), handle separately
        xirr_df = pd.DataFrame([xirr_row])
        full_matrix = pd.concat([full_matrix, xirr_df], ignore_index=True)

        display_matrix = _style_matrix(full_matrix)
        st.dataframe(display_matrix, use_container_width=True, hide_index=True, height=700)

        # Download matrix as CSV or Excel
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download Matrix as CSV",
                full_matrix.to_csv(index=False),
                file_name="chit_cash_flow_matrix.csv",
                mime="text/csv",
            )
        with dl_col2:
            xlsx_buf = BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                full_matrix.to_excel(writer, index=False, sheet_name="Cash Flow Matrix")
            st.download_button(
                "Download Matrix as Excel",
                xlsx_buf.getvalue(),
                file_name="chit_cash_flow_matrix.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    # --- Tab: Comparison Table ---
    with tabs[4]:
        comp_data = []
        for a in result.analyses:
            comp_data.append({
                "Lifting Month": a.lift_month,
                "Total Paid": format_inr(a.total_paid),
                "Prize": format_inr(a.prize_received),
                "Net Cost": format_inr(a.net_cost),
                "NPV": format_inr(a.npv),
                "XIRR (%)": f"{a.xirr * 100:.1f}" if a.xirr is not None else "N/A",
                "Optimal": "\u2705" if a.lift_month == result.optimal_month else "",
            })
        comp_df = pd.DataFrame(comp_data)
        st.subheader("All Lifting Months Comparison")
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # CSV download
        csv_df = pd.DataFrame([
            {
                "Lifting Month": a.lift_month,
                "Total Paid": a.total_paid,
                "Prize": a.prize_received,
                "Net Cost": a.net_cost,
                "NPV": round(a.npv, 2),
                "XIRR (%)": round(a.xirr * 100, 2) if a.xirr is not None else "",
            }
            for a in result.analyses
        ])
        st.download_button(
            "Download as CSV",
            csv_df.to_csv(index=False),
            file_name="chit_analysis.csv",
            mime="text/csv",
        )

    # --- Uploaded Data tab (if files present) ---
    if has_uploads:
        with tabs[tab_names.index("Uploaded Data")]:
            _render_uploaded_data()

    # --- Saved Chits tab ---
    with tabs[saved_tab_idx]:
        _render_saved_chits()

elif "uploaded_files" in st.session_state and st.session_state["uploaded_files"]:
    st.subheader("Uploaded Data")
    _render_uploaded_data()
    st.info("Configure chit parameters in the sidebar and click **Analyze** to see results.")
    st.divider()
    st.subheader("Saved Chits")
    _render_saved_chits()
else:
    st.info("Configure chit parameters in the sidebar and click **Analyze** to see results.")
    st.divider()
    st.subheader("Saved Chits")
    _render_saved_chits()
