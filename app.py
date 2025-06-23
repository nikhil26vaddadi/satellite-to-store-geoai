import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

# Load predicted demand
df = pd.read_csv("data/predictions/forecasted_demand.csv")

# Add dummy store coordinates for demo
city_coords = {
    "Cork_IE": (51.8985, -8.4756),
    "Dublin_IE": (53.3498, -6.2603),
    "London_UK": (51.5072, -0.1276)
}

# Attach lat/lon to each row
df["lat"] = df["city"].map(lambda c: city_coords.get(c, (0, 0))[0])
df["lon"] = df["city"].map(lambda c: city_coords.get(c, (0, 0))[1])

# Streamlit layout
st.set_page_config(layout="wide")
st.title("📦 Satellite-to-Store GeoAI Dashboard")

col1, col2, col3 = st.columns(3)
col1.metric("Total Forecasted Demand", f"{df['predicted_demand'].sum():,.0f}")
col2.metric("Avg. Demand per Store", f"{df['predicted_demand'].mean():.1f}")
col3.metric("Cities Forecasted", df['city'].nunique())

st.markdown("### 🗺️ Store Demand Map")

# Folium map
m = folium.Map(location=[52, -5], zoom_start=5)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    popup_text = f"""
    <b>City:</b> {row['city']}<br>
    <b>Predicted Demand:</b> {row['predicted_demand']:.0f}
    """
    folium.Marker(
        location=[row["lat"], row["lon"]],
        popup=popup_text,
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(marker_cluster)

folium_static(m)

st.markdown("### 🔍 Raw Forecast Data")
st.dataframe(df[["city", "datetime", "predicted_demand"]].head(10))


# --- New Section: Future Demand Forecast ---
st.header("📈 Future Demand Forecast (Next 48 Hours)")

# Load CSV
FUTURE_PATH = "data/predictions/future_demand_48h.csv"
@st.cache_data
def load_future_demand():
    df = pd.read_csv(FUTURE_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# Show results
future_df = load_future_demand()

if not future_df.empty:
    city = st.selectbox("Select City", future_df["city"].unique())
    city_df = future_df[future_df["city"] == city]

    st.line_chart(city_df.set_index("datetime")["predicted_demand"])
    st.dataframe(city_df.reset_index(drop=True), use_container_width=True)
else:
    st.warning("⚠️ No future forecast available. Please generate it first.")
