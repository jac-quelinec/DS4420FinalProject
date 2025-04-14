import streamlit as st

st.set_page_config(page_title="Food Insecurity Project", layout="centered")

st.title("Utilizing MultiLayer Perceptron and Collaborative Filtering to Explore Food Insecurity in the U.S.")
st.markdown("By Jacqueline Chao, Avita Islam, and Veronica Yang")

st.markdown("## Project Overview")
st.markdown("""
Our project combines a Multilayer Perceptron model and collaborative filtering to address food access inequality in the United States on the county level.
The MLP, built in Python, is trained on different socioeconomic and demographic data collected by the United States Department of Agriculture's Food Access Research Atlas, which predicts areas that are at risk of being food deserts (defined below).
After exploring our MLP model's capabilities, we built a collaborative filtering model in R to reommend specific government policies (e.g., SNAP outreach or mobile markets) based on similar communities.
Put together, these two models can offer people a way to identify food-insecure areas and suggest data-driven solutions to policymakers and their governments. 
""")

st.markdown("## Terms to Know")
st.markdown("""
- Census Tract: a small geographic area used by the U.S. Census Bureau to report population and demographic data.
- Food Desert: a low-income area where residents live far from supermarkets—over 1 mile in urban areas or 10 miles in rural areas (as defined by the USDA).
- Predicted Food Desert: a label (1 == yes or 0 == no) from the model indicating whether a tract is likely a food desert.
- USDA Food Desert: the USDA's official classification of food deserts using specific criteria.
- Food Assistance and Access Programs: government or community led initiatives designed to help underserved communities have better and more nutritious food access using developed solutions.
""")