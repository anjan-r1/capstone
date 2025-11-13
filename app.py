# app.py
import os
import textwrap

import streamlit as st
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from live_search import fetch_used_cars_live
from value_model import compute_value_scores


load_dotenv()  # loads variables from .env

# Optional: LLM (OpenAI). Comment out if not using.
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


def get_llm_recommendation(df: pd.DataFrame, user_profile: dict) -> str:
    """
    Use an LLM to explain the top 3 recommendations in natural language.
    Safe: if no API key or library, returns a simple fallback string.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not OPENAI_AVAILABLE or not api_key or df.empty:
        return "LLM explanation not available (missing API key or library)."

    client = OpenAI(api_key=api_key)

    top3 = df.head(3)
    cars_summary = []
    for _, row in top3.iterrows():
        cars_summary.append(
            f"- {row.get('title')} (Price: ${row.get('price_sgd'):,}, "
            f"Mileage: {row.get('mileage_km')} km, "
            f"Value score: {row.get('value_score')})\n  URL: {row.get('listing_url')}"
        )

    profile_txt = "\n".join(f"{k}: {v}" for k, v in user_profile.items())
    cars_txt = "\n".join(cars_summary)

    prompt = f"""
You are an expert Singapore car consultant.

User profile:
{profile_txt}

You have the following top 3 car options from SGCarMart (already filtered & ranked by a value score):

{cars_txt}

Explain to the user:
1. Which car suits them best and why.
2. Trade-offs between the top 3.
3. Any caveats they should watch out for (e.g. high mileage, older COE).

Keep it within 3 paragraphs. Use simple, friendly language.
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful automotive advisor in Singapore."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    return resp.choices[0].message.content.strip()


def main():
    st.set_page_config(
        page_title="SG Car Advisor (Capstone)",
        layout="wide",
    )

    st.title("🚗 SG Car Advisor")
    st.markdown(
        "Smart car recommendations using live SGCarMart data, value scoring, and (optionally) an LLM."
    )

    with st.sidebar:
        st.header("Your Preferences")

        budget_min = st.number_input("Min Budget (SGD)", min_value=0, value=50000, step=5000)
        budget_max = st.number_input("Max Budget (SGD)", min_value=0, value=120000, step=5000)

        if budget_max and budget_min and budget_max < budget_min:
            st.error("Max budget must be ≥ Min budget.")

        make_choice = st.text_input("Preferred Brand (optional, e.g. Toyota, Honda)")

        body_type = st.selectbox(
            "Body Type (future use)",
            ["Any", "Hatchback", "Sedan", "SUV", "MPV", "Coupe"],
            index=0,
        )

        mileage_pref = st.select_slider(
            "How important is low mileage?",
            options=["Not important", "Somewhat", "Very important"],
            value="Somewhat",
        )

        running_cost_pref = st.select_slider(
            "How important is low yearly running cost (depreciation)?",
            options=["Not important", "Somewhat", "Very important"],
            value="Very important",
        )

        st.markdown("---")
        st.caption("⚠ This app fetches only the first page of live SGCarMart results per query.")

        search_btn = st.button("🔎 Search Cars")

    if not search_btn:
        st.info("Set your preferences in the sidebar and click **Search Cars**.")
        return

    # User profile dict (for LLM)
    user_profile = {
        "Budget range": f"${budget_min:,} - ${budget_max:,}",
        "Preferred brand": make_choice or "Any",
        "Body type": body_type,
        "Low mileage importance": mileage_pref,
        "Low depreciation importance": running_cost_pref,
    }

    # Fetch live data
    with st.spinner("Fetching live used-car listings from SGCarMart..."):
        df_raw = fetch_used_cars_live(
            budget_min=budget_min or None,
            budget_max=budget_max or None,
            make=make_choice or None,
            max_results=40,
        )

    if df_raw.empty:
        st.error("No cars found or request blocked. Try relaxing filters or trying again later.")
        return

    st.subheader("Raw listings fetched")
    st.caption("First few rows (before scoring):")
    st.dataframe(df_raw.head(10))

    # Compute value scores
    df_scored = compute_value_scores(df_raw)

    st.subheader("Top Recommendations by Value Score")
    st.caption("Higher score = better relative value in this search result set.")

    # Display top cars as cards
    top_n = min(10, len(df_scored))
    for idx in range(top_n):
        row = df_scored.iloc[idx]
        with st.container(border=True):
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"### #{int(row['value_rank'])} – {row.get('title','(No title)')}")
                st.markdown(
                    f"**Estimated Value Score:** `{row['value_score']}` / 100"
                )
                details_lines = []
                if row.get("price_sgd"):
                    details_lines.append(f"💰 Price: **${row['price_sgd']:,}**")
                if row.get("mileage_km"):
                    details_lines.append(f"🚘 Mileage: **{int(row['mileage_km']):,} km**")
                if row.get("depreciation_per_year"):
                    details_lines.append(
                        f"📉 Depreciation: **${int(row['depreciation_per_year']):,}/yr**"
                    )
                if row.get("year"):
                    details_lines.append(f"📅 Year: **{int(row['year'])}**")
                if row.get("coe_left_years") is not None:
                    details_lines.append(
                        f"🪪 COE left (approx): **{row['coe_left_years']} years**"
                    )

                if details_lines:
                    st.markdown("\n".join(details_lines))

                if row.get("listing_url"):
                    st.markdown(
                        f"[View on SGCarMart]({row['listing_url']})",
                        unsafe_allow_html=True,
                    )

            with cols[1]:
                st.markdown("**Score breakdown (relative)**")
                # Display columns if present
                for col, label in [
                    ("price_sgd", "Price"),
                    ("mileage_km", "Mileage"),
                    ("depreciation_per_year", "Depreciation"),
                    ("year", "Year"),
                ]:
                    val = row.get(col)
                    st.text(f"{label}: {val if pd.notna(val) else '-'}")

    # Charts
    with st.expander("📊 Visualise this search (scatter plots)"):
        chart_cols = st.columns(2)

        if "price_sgd" in df_scored and "mileage_km" in df_scored:
            with chart_cols[0]:
                st.markdown("**Price vs Mileage**")
                st.scatter_chart(
                    df_scored[["price_sgd", "mileage_km"]].rename(
                        columns={"price_sgd": "Price", "mileage_km": "Mileage"}
                    )
                )

        if "price_sgd" in df_scored and "value_score" in df_scored:
            with chart_cols[1]:
                st.markdown("**Price vs Value Score**")
                st.scatter_chart(
                    df_scored[["price_sgd", "value_score"]].rename(
                        columns={"price_sgd": "Price", "value_score": "Value Score"}
                    )
                )

    # LLM explanation
    st.markdown("---")
    st.subheader("🤖 AI Explanation (Optional)")

    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        if st.button("Generate AI explanation of top 3 cars"):
            with st.spinner("Calling LLM to explain recommendations..."):
                explanation = get_llm_recommendation(df_scored, user_profile)
            st.markdown(explanation)
    else:
        st.info(
            "Set `OPENAI_API_KEY` and install `openai` package to enable AI explanations."
        )


if __name__ == "__main__":
    main()
