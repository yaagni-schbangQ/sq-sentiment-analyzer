import streamlit as st
import pandas as pd
from predict_sentiment import predict_sentiment

st.title("🧠 SchbangQ Sentiment Analyzer")

uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Raw Input Data")
    st.dataframe(df)

    text_col = st.selectbox("Select the column with text responses", df.columns)

    with st.spinner("Analyzing sentiments..."):
        df["Sentiment"] = df[text_col].apply(predict_sentiment)

    st.success("Analysis complete!")
    st.write("### Sentiment Results")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Results as CSV", data=csv, file_name="sentiment_results.csv")