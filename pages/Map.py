import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Predicted Food Desert Map", layout="wide")

st.title("Predicted Food Desert Prevalence by County")

st.markdown("""
This map displays the predicted prevalence of food deserts across U.S. counties based on our MLP.
Each county is shaded according to how the model classified census tracts as as food deserts, given in a percentage. This percentage was calculated by taking the number of census tracts where the model predicted a food desert (label = 1), divided by the total number of tracts in that county. Then that value was multiplied by 100 to get the predicted food desert as a percent.

Hover over each county to view:
- The county name
- The state
- The model's predicted percentage of food desert tracts

NOTE: gray areas reveal missing data or counties that could not be matched in the shapefile.
""")

# Loads the map
with open("predicted_food_desert_map.html", "r", encoding="utf-8") as f:
    map_html = f.read()

components.html(map_html, height=700, scrolling=True)

