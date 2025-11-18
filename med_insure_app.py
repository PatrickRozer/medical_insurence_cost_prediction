import streamlit as st
import pandas as pd
import joblib
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, text
from sqlalchemy import types as satypes
import io

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Medical Insurance Cost Prediction",
    page_icon="💊",
    layout="centered",
)
st.title("💊 Medical Insurance Cost Prediction")
st.write("Fill in the details below and get an estimated insurance charge.")

# -----------------------------
# Database setup (edit values as needed)
# -----------------------------
DB_USER = "postgres"
DB_PASS = "142789"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "Medical_Insurance_Cost_Prediction_db"
TABLE_NAME = "Medical_Insurance"   # user-visible canonical name

# -----------------------------
# Utility: create engine
# -----------------------------
@st.cache_resource
def get_engine():
    # URL-encode the password to avoid issues with special chars
    pw = quote_plus(DB_PASS)
    url = f"postgresql+psycopg2://{DB_USER}:{pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, echo=False)

engine = get_engine()

# -----------------------------
# Helper: SQL table name referencing
# -----------------------------
def sql_table_ref(schema: str, tablename: str) -> str:
    """
    Return a SQL reference to the table, quoting the table name if it contains
    uppercase or special characters so Postgres will match exact-case names.
    """
    if tablename != tablename.lower():
        return f'{schema}."{tablename}"'
    return f"{schema}.{tablename}"

# -----------------------------
# Helper: find existing table name (case-insensitive match)
# -----------------------------
def find_existing_table(engine, desired_name, schema="public"):
    ins = inspect(engine)
    try:
        tables = ins.get_table_names(schema=schema)
    except Exception:
        # fallback: empty list if inspection fails
        tables = []
    # Try to find exact-case match first, then case-insensitive
    for t in tables:
        if t == desired_name:
            return t
    for t in tables:
        if t.lower() == desired_name.lower():
            return t
    return None

# Determine which table name to use (existing or fallback to lower-case)
existing_table = find_existing_table(engine, TABLE_NAME, schema="public")
if existing_table:
    DB_TABLE_NAME = existing_table
else:
    # default to lower-case name; `to_sql` will create this if needed
    DB_TABLE_NAME = TABLE_NAME.lower()

DB_TABLE_SQL = sql_table_ref("public", DB_TABLE_NAME)

# -----------------------------
# Load trained pipeline
# -----------------------------
@st.cache_resource
def load_pipeline():
    obj = joblib.load("best_model.pkl")
    if isinstance(obj, dict):
        return obj["pipeline"], obj.get("feature_order")
    return obj, None

try:
    pipeline, feature_order = load_pipeline()
except Exception as e:
    st.error(f"❌ Could not load `best_model.pkl` — {e}")
    st.stop()

# -----------------------------
# Input form
# -----------------------------
with st.form("insurance_form", clear_on_submit=False):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("👤 Age", min_value=18, max_value=100, value=30, step=1)
        bmi = st.number_input("⚖️ BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        children = st.number_input("👶 Number of Children", min_value=0, max_value=10, value=0, step=1)

    with col2:
        sex = st.radio("⚧ Sex", ["male", "female"], horizontal=True)
        smoker = st.radio("🚬 Smoker", ["no", "yes"], horizontal=True)
        region = st.selectbox("🌍 Region", ["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("🔮 Predict Insurance Cost")

# -----------------------------
# Prediction + Logging
# -----------------------------
if submitted:
    input_data = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
    }])

    if feature_order is not None:
        # ensure we don't break when the model expects a particular feature order
        input_data = input_data.reindex(columns=feature_order)

    try:
        # Predict
        prediction = pipeline.predict(input_data)[0]
        st.subheader("✅ Prediction Result")
        st.success(f"💵 Estimated Insurance Cost: **${float(prediction):,.2f}**")

        # Prepare row to insert
        log_df = input_data.copy()
        log_df["predicted_cost"] = float(prediction)
        log_df["created_at"] = pd.Timestamp.utcnow()

        # Ensure columns ordering matches the database columns (optional)
        # dtype mapping for to_sql
        dtype_map = {
            "age": satypes.Integer(),
            "bmi": satypes.Numeric(6, 2),
            "children": satypes.Integer(),
            "sex": satypes.Text(),
            "smoker": satypes.Text(),
            "region": satypes.Text(),
            "predicted_cost": satypes.Numeric(12, 2),
            "created_at": satypes.DateTime(),
        }

        # Use the resolved DB table name for insertion (case handled)
        with engine.begin() as conn:
            # pandas to_sql will create the table if it doesn't exist.
            # Use DB_TABLE_NAME (actual table name to create/append) and schema='public'
            log_df.to_sql(
                name=DB_TABLE_NAME,
                con=conn,
                if_exists="append",
                index=False,
                dtype=dtype_map,
                method="multi",
                schema="public",
            )

            # Query count using DB_TABLE_SQL which is properly quoted when necessary
            total = conn.execute(text(f"SELECT COUNT(*) FROM {DB_TABLE_SQL}")).scalar()

        st.info(f"🗄️ Prediction logged. Table `{DB_TABLE_NAME}` now has {total} rows.")

    except Exception as e:
        st.error("⚠️ Prediction logging failed.")
        st.exception(e)
        st.write("Payload we tried to insert:")
        st.dataframe(log_df if "log_df" in locals() else input_data)

# -----------------------------
# Optional: Show past predictions
# -----------------------------
if st.checkbox("📊 Show previous predictions from database"):
    try:
        st.markdown("### 🔎 Filter Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_sex = st.selectbox("⚧ Sex", ["All", "male", "female"])
        with col2:
            filter_smoker = st.selectbox("🚬 Smoker", ["All", "yes", "no"])
        with col3:
            filter_region = st.selectbox("🌍 Region", ["All", "northeast", "northwest", "southeast", "southwest"])

        num_rows = st.slider("How many records to show?", min_value=5, max_value=50, value=10, step=5)
        sort_by = st.selectbox("Sort by:", ["created_at", "age", "predicted_cost"], index=0)

        conditions = []
        if filter_sex != "All":
            conditions.append(f"sex = :sex")
        if filter_smoker != "All":
            conditions.append(f"smoker = :smoker")
        if filter_region != "All":
            conditions.append(f"region = :region")

        where_clause = " AND ".join(conditions)
        if where_clause:
            where_clause = "WHERE " + where_clause

        # Use parameterized query to avoid SQL injection
        query = f'SELECT * FROM {sql_table_ref("public", DB_TABLE_NAME)} {where_clause} ORDER BY "{sort_by}" DESC LIMIT :limit;'

        params = {"limit": int(num_rows)}
        if filter_sex != "All":
            params["sex"] = filter_sex
        if filter_smoker != "All":
            params["smoker"] = filter_smoker
        if filter_region != "All":
            params["region"] = filter_region

        history = pd.read_sql(text(query), engine, params=params)

        st.subheader("📋 Recent Predictions from Database")
        st.dataframe(history)

        if not history.empty:
            avg_cost = history["predicted_cost"].mean()
            max_cost = history["predicted_cost"].max()
            min_cost = history["predicted_cost"].min()

            st.markdown("### 📊 Key Statistics")
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric(label="📈 Average Cost", value=f"${avg_cost:,.2f}")
            with kpi2:
                st.metric(label="💰 Highest Cost", value=f"${max_cost:,.2f}")
            with kpi3:
                st.metric(label="🏷️ Lowest Cost", value=f"${min_cost:,.2f}")

        # Downloads
        st.markdown("### 📥 Download Prediction History")
        csv_buffer = io.StringIO()
        history.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download as CSV",
            data=csv_buffer.getvalue(),
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
            history.to_excel(writer, index=False, sheet_name="Predictions")
        st.download_button(
            label="⬇️ Download as Excel",
            data=excel_buffer.getvalue(),
            file_name="prediction_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # Clear History
        st.markdown("### 🗑️ Manage History")
        if st.button("⚠️ Clear All Prediction History"):
            with engine.begin() as conn:
                conn.execute(text(f'DELETE FROM {sql_table_ref("public", DB_TABLE_NAME)};'))
            st.warning("🗑️ All prediction history has been cleared! Refresh to see changes.")

    except Exception as e:
        st.error(f"Could not fetch past predictions: {e}")
