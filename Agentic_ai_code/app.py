import streamlit as st
import pandas as pd
import os
import re
import json
import matplotlib.pyplot as plt
from fpdf import FPDF
from langchain_ollama import ChatOllama
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. PAGE SETUP & DATA LOADING
# ==========================================
st.set_page_config(page_title="Enterprise AI Data Agent", layout="wide", page_icon="🏢")


@st.cache_data
def load_data():
    emp_path = "view_agent_context_full.csv"
    course_path = "view_course_effectiveness.csv"

    for p in (emp_path, course_path):
        if not os.path.exists(p):
            st.error(f"Please ensure '{p}' is in the same folder as this script.")
            st.stop()

    df_emp = pd.read_csv(emp_path)
    df_course = pd.read_csv(course_path)
    return df_emp, df_course


df_emp, df_course = load_data()


# ==========================================
# 2. CHART GENERATION (DETERMINISTIC, NO AI)
# ==========================================
def generate_chart(chart_spec, filename="temp_graph.png"):
    """
    chart_spec is a dict like:
    {
        "chart_type": "bar" | "scatter" | "line" | "hist",
        "title": "...",
        "x_label": "...",
        "y_label": "...",
        "categories": ["Meera Reddy's Team", "Rohan Joshi's Team"],   # for bar/line
        "values": [43.94, 43.04],                                      # for bar/line/hist
        "x_values": [0.1, 0.2, ...],   # for scatter
        "y_values": [50, 60, ...]      # for scatter
    }
    Returns True if a chart was created, False otherwise.
    """
    try:
        chart_type = chart_spec.get("chart_type", "bar").lower()
        title = chart_spec.get("title", "")
        x_label = chart_spec.get("x_label", "")
        y_label = chart_spec.get("y_label", "")

        plt.figure(figsize=(8, 5))

        if chart_type == "bar":
            categories = chart_spec["categories"]
            values = chart_spec["values"]
            colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
            bars = plt.bar(categories, values, color=colors[:len(categories)])
            for bar, val in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

        elif chart_type == "scatter":
            x_values = chart_spec["x_values"]
            y_values = chart_spec["y_values"]
            plt.scatter(x_values, y_values, alpha=0.6, color="#4C72B0")

        elif chart_type == "line":
            categories = chart_spec["categories"]
            values = chart_spec["values"]
            plt.plot(categories, values, marker='o', color="#4C72B0")

        elif chart_type == "hist":
            values = chart_spec["values"]
            plt.hist(values, bins=chart_spec.get("bins", 10), color="#4C72B0", alpha=0.8)

        elif chart_type == "pie":
            categories = chart_spec["categories"]
            values = chart_spec["values"]
            colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860", "#DA8BC3", "#8C8C8C"]
            total = sum(values)

            def make_autopct(vals):
                def autopct(pct):
                    count = round(pct * total / 100.0)
                    return f"{count}\n({pct:.1f}%)"
                return autopct

            plt.pie(values, labels=categories, autopct=make_autopct(values), startangle=90,
                    colors=(colors * (len(categories) // len(colors) + 1))[:len(categories)])
            plt.axis('equal')

        else:
            plt.close()
            return False

        plt.title(title, fontsize=13, fontweight='bold')
        if chart_type != "pie":
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        return True

    except Exception:
        plt.close()
        return False


def generate_table_pdf(title, df_table, output_path, graph_filename=None):
    """Generate a PDF with a tabular listing of employees and their KPIs.
    Optionally embeds a chart image on a new page if graph_filename exists."""
    pdf = FPDF(orientation='L')  # landscape for wide tables
    pdf.add_page()

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, txt="Enterprise AI Analytics Report", ln=True, align='C')
    pdf.ln(2)

    pdf.set_font("Arial", 'B', 11)
    clean_title = str(title).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 7, txt=clean_title)
    pdf.ln(3)

    # Table
    cols = df_table.columns.tolist()
    n_cols = len(cols)
    page_width = pdf.w - 20
    col_width = page_width / n_cols

    pdf.set_font("Arial", 'B', 8)
    for col in cols:
        label = str(col).encode('latin-1', 'replace').decode('latin-1')
        pdf.cell(col_width, 8, txt=label[:18], border=1, align='C')
    pdf.ln()

    pdf.set_font("Arial", '', 8)
    for _, row in df_table.iterrows():
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                val = round(val, 2)
            text = str(val).encode('latin-1', 'replace').decode('latin-1')
            pdf.cell(col_width, 7, txt=text[:18], border=1, align='C')
        pdf.ln()

    # Embed chart on a new page if provided
    if graph_filename and os.path.exists(graph_filename):
        pdf.add_page(orientation='P')
        pdf.image(graph_filename, x=15, w=180)

    pdf.output(output_path)
    return output_path


def find_course_id(course_name_query, df_course):
    """Fuzzy match a course name to its course_id."""
    matches = df_course[df_course['course_name'].str.contains(course_name_query, case=False, na=False)]
    if matches.empty:
        return None, None
    return matches.iloc[0]['course_id'], matches.iloc[0]['course_name']


def try_handle_employee_list_request(prompt, df_emp, df_course):
    """
    Deterministically handle: 'list employees (with KPIs) who completed [course name or course_id]'.
    Supports requests for manager_name and a voucher-eligibility bar chart.
    Returns (handled: bool, text_answer: str, table_df: pd.DataFrame or None, chart_spec: dict or None).
    """
    p = prompt.lower()
    if "completed" not in p:
        return False, None, None, None
    if not re.search(r'\b(list|employee id|employee_id|kpi|details|table|name)\b', p):
        return False, None, None, None

    course_id = None
    course_name = None

    # 1. Direct course_id match (e.g. "C1001_DB")
    id_match = re.search(r'\bC\d{3,4}_[A-Z]{2,3}\b', prompt, flags=re.IGNORECASE)
    if id_match:
        candidate_id = id_match.group(0).upper()
        row = df_course[df_course['course_id'].str.upper() == candidate_id]
        if not row.empty:
            course_id = row.iloc[0]['course_id']
            course_name = row.iloc[0]['course_name']

    # 2. Fall back to course name extraction
    if course_id is None:
        # Extract course name: after "completed" up to " and ", " or ", " as per ", " with ", " show ", etc.
        # Non-greedy match stops at the first natural boundary
        m = re.search(
            r'completed\s+(?:course\s+)?(.+?)(?:\s+(?:and|or|as\s+per|with|show|generate|in\s+bar|chart|graph|pdf|for|to)\b|$)',
            prompt,
            flags=re.IGNORECASE
        )
        if not m:
            return False, None, None, None
        course_query = m.group(1).strip().strip('"\'')
        course_id, course_name = find_course_id(course_query, df_course)
        if course_id is None:
            return False, None, None, None

    mask = df_emp['completed_course_ids'].astype(str).str.contains(course_id, na=False)
    matched = df_emp.loc[mask].copy()

    # Include manager_name if the prompt asks for it
    kpi_cols = ['employee_id', 'employee_name']
    if 'manager' in p and 'manager_name' in matched.columns:
        kpi_cols.append('manager_name')
    kpi_cols += ['effectiveness_score', 'engagement_score', 'skill_gap_index',
                  'completeness_rate', 'voucher_eligibility']
    existing_cols = [c for c in kpi_cols if c in matched.columns]
    table_df = matched[existing_cols].reset_index(drop=True)

    voucher_counts = matched['voucher_eligibility'].value_counts()
    voucher_str = ", ".join(f"{status}: {count}" for status, count in voucher_counts.items())

    text_answer = (
        f"{len(table_df)} employees have completed '{course_name}' ({course_id}). "
        f"Voucher eligibility breakdown: {voucher_str}. "
        f"The full list with employee ID, name, manager and KPIs is included in the table below / PDF."
    )

    # Voucher eligibility bar chart if requested
    chart_spec = None
    if re.search(r'\b(bar|chart|graph|plot|visuali[sz]e)\b', p) and 'voucher' in p:
        chart_spec = {
            "chart_type": "bar",
            "title": f"Voucher Eligibility - '{course_name}' Completers",
            "x_label": "Voucher Eligibility",
            "y_label": "Count",
            "categories": voucher_counts.index.tolist(),
            "values": voucher_counts.values.tolist(),
        }

    return True, text_answer, table_df, chart_spec


def try_handle_completeness_kpi_request(prompt, df_emp):
    """
    Deterministically handle requests like:
    'employees with completeness_rate of exactly 100 ... with kpi/details, grouped by department'
    Returns (handled: bool, text_answer: str, table_df: pd.DataFrame or None, chart_spec: dict or None).
    """
    p = prompt.lower()
    if "completeness_rate" not in p and "completeness rate" not in p:
        return False, None, None, None
    if not re.search(r'\b(kpi|details|table|list employees|employee id|employee_id)\b', p):
        return False, None, None, None

    # Extract the target completeness value (default 100)
    m = re.search(r'(\d+(?:\.\d+)?)', p)
    target_value = float(m.group(1)) if m else 100.0

    matched = df_emp[df_emp['completeness_rate'] == target_value].copy()

    kpi_cols = ['employee_id', 'employee_name', 'department', 'effectiveness_score',
                'engagement_score', 'skill_gap_index', 'completeness_rate', 'voucher_eligibility']
    existing_cols = [c for c in kpi_cols if c in matched.columns]
    table_df = matched[existing_cols].reset_index(drop=True)

    dept_counts = matched['department'].value_counts()
    breakdown_str = ", ".join(f"{dept}: {count}" for dept, count in dept_counts.items())

    text_answer = (
        f"Total: {len(table_df)} employees have a completeness_rate of exactly {target_value:g}. "
        f"By department: {breakdown_str}. "
        f"The full list with employee ID, name and KPIs is included in the table below / PDF."
    )

    chart_spec = None
    wants_pie = bool(re.search(r'\bpie\b', p))
    wants_bar = bool(re.search(r'\b(bar|chart|graph|plot|visuali[sz]e)\b', p))
    if wants_pie or wants_bar:
        chart_spec = {
            "chart_type": "pie" if wants_pie else "bar",
            "title": f"Employees with completeness_rate = {target_value:g} by Department",
            "x_label": "Department",
            "y_label": "Count",
            "categories": dept_counts.index.tolist(),
            "values": dept_counts.values.tolist(),
        }

    return True, text_answer, table_df, chart_spec


# ==========================================
# 3. PDF GENERATION ENGINE
# ==========================================
def generate_pdf_report(user_query, text_response, graph_filename="temp_graph.png", msg_id=None):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt="Enterprise AI Analytics Report", ln=True, align='C')
    pdf.ln(5)

    # Query Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="User Query:", ln=True)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 8, txt=user_query)
    pdf.ln(5)

    # Response Section
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="AI Analysis:", ln=True)
    pdf.set_font("Arial", '', 11)

    # Clean text for PDF encoding (prevents character errors)
    clean_text = str(text_response).encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 8, txt=clean_text)
    pdf.ln(10)

    # Insert Graph if one was generated
    if os.path.exists(graph_filename):
        pdf.image(graph_filename, x=15, w=180)

    if msg_id is None:
        msg_id = abs(hash(user_query))
    output_path = f"report_{msg_id}.pdf"
    pdf.output(output_path)
    return output_path


# ==========================================
# 4. AI AGENT (THE BRAIN) - Computes numbers only
# ==========================================
@st.cache_resource
def get_ai_agent():
    llm = ChatOllama(model="llama3.1", temperature=0)

    PREFIX = """
    You are an Enterprise Analytics AI. You have access to TWO pandas dataframes:

    - df1: Employee / Learning Agent Context data (one row per employee). Columns include
      employee_name, department, designation, manager_name, location, effectiveness_score,
      engagement_score, skill_gap_index, completeness_rate, voucher_eligibility,
      completed_course_ids (a string containing course IDs like 'C1172_MS', NOT course names),
      completed_course_domains, etc.

    - df2: Course Effectiveness data (one row per course). Columns include
      course_id (e.g. 'C1172_MS'), course_name (e.g. 'Azure Data Factory Basics Lab 2'),
      category, vendor, delivery_mode, difficulty_level, total_learners,
      completion_rate_pct, pass_rate_pct, avg_score_passed, avg_feedback_rating,
      avg_days_to_complete, etc.

    CRITICAL RULES:
    1. YOU MUST EXECUTE THE CODE: Use your python tools to calculate the actual answer from df1 and/or df2. Do NOT guess.
    2. CHOOSE THE RIGHT DATAFRAME: Employee/team/department/engagement/effectiveness questions use df1.
       Course/vendor/category/completion/pass-rate questions use df2. If the question needs both, use both.
    3. NEVER PRINT FULL DATAFRAMES: Never run code that returns a full dataframe with all columns
       (e.g. df1[df1['manager_name'] == 'X']). This wastes time and breaks your reasoning.
       ALWAYS select only the specific column(s) you need and aggregate immediately, e.g.:
       df1[df1['manager_name'] == 'Meera Reddy']['engagement_score'].mean()
       Every Action Input must return a single number, a short Series, or a small grouped table - never raw rows with all 41 columns.
    4. COURSE COMPLETION LOOKUPS (VERY IMPORTANT): df1's completed_course_ids column contains
       comma-separated course IDs (like 'C1431_CR, C1151_MS'), NOT course names. If the user asks
       "how many employees completed [course name]", you MUST do a two-step lookup:
       (a) Find the course_id in df2 where course_name matches (case-insensitive, allow partial match).
       (b) Count rows in df1 where completed_course_ids contains that course_id
           (e.g. df1['completed_course_ids'].astype(str).str.contains(course_id, na=False).sum()).
       NEVER search for the course name string directly inside completed_course_ids - it will not match.
    5. DO NOT GENERATE ANY CHARTS OR USE MATPLOTLIB. Only compute numeric results with pandas.
    6. NO CODE IN FINAL ANSWER: Your final response must NOT contain any raw Python code or triple backticks.
    7. STRICT TOOL FORMAT: Every step MUST be exactly two lines in this format, with no deviation:
       Action: python_repl_ast
       Action Input: <a single line of valid python/pandas code>
       NEVER write "Action: Use python_repl_ast to..." or describe the action in words.
       NEVER put code in triple-backtick blocks inside Action Input. One line of code only.
    8. NEVER INVENT NEW DATAFRAME NAMES OR COLUMN NAMES. The only dataframes are df1 and df2 as
       described above. Do not use 'df', 'Manager', 'Engagement', or any name not listed in the schemas.
       Use the EXACT employee/manager names given in the user's question - do not substitute similar-sounding names.
    9. WHEN A GROUPED BREAKDOWN IS REQUESTED (e.g. "grouped by department", "by team", "per X"):
       your Final Answer text MUST explicitly state the total AND list each group's value
       (e.g. "Total: 196. By department: Ai & Data Science: 12, Bi & Analytics: 15, ...").
       Do not only put the breakdown in the chart JSON - it must also appear in the readable sentence.
    10. MANDATORY FORMATTING: When you have calculated the result and are ready to respond to the user,
       you MUST begin your response with the exact phrase "Final Answer: ". If you do not use this phrase,
       the system will crash.
    """

    agent = create_pandas_dataframe_agent(
        llm,
        [df_emp, df_course],  # df1 = df_emp, df2 = df_course
        verbose=True,
        agent_type="zero-shot-react-description",
        agent_executor_kwargs={"handle_parsing_errors": True},
        max_iterations=15,
        max_execution_time=120,
        prefix=PREFIX,
        allow_dangerous_code=True
    )
    return agent


def extract_json_block(text):
    """Try to find and parse a JSON object embedded in text."""
    matches = re.findall(r'\{[^{}]*\}', text, flags=re.DOTALL)
    for m in reversed(matches):  # try last block first (often the final structured answer)
        try:
            return json.loads(m)
        except Exception:
            continue
    return None


# ==========================================
# 5. STREAMLIT USER INTERFACE (HYBRID)
# ==========================================
st.title("🧠 Enterprise L&D AI Hub")

# ---------------------------------------------------------
# SECTION A: FAST DATA EXPLORER (DETERMINISTIC / NO AI)
# ---------------------------------------------------------
st.subheader("🔍 Fast Data Explorer (100% Accurate Data Retrieval)")
st.markdown("Use this section to look up exact lists of employees, check individual KPIs, or filter by manager/department without waiting for the AI.")

tab_emp, tab_course = st.tabs(["👤 Employees", "📚 Courses"])

with tab_emp:
    with st.expander("Click to Open Employee Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)

        sel_dept = col1.selectbox("Department:", ["All"] + sorted(df_emp['department'].dropna().unique().tolist()))
        sel_mgr = col2.selectbox("Manager Name:", ["All"] + sorted(df_emp['manager_name'].dropna().unique().tolist()))
        sel_role = col3.selectbox("Designation:", ["All"] + sorted(df_emp['designation'].dropna().unique().tolist()))
        sel_loc = col4.selectbox("Location:", ["All"] + sorted(df_emp['location'].dropna().unique().tolist()))

        filtered_df = df_emp.copy()
        if sel_dept != "All": filtered_df = filtered_df[filtered_df['department'] == sel_dept]
        if sel_mgr != "All": filtered_df = filtered_df[filtered_df['manager_name'] == sel_mgr]
        if sel_role != "All": filtered_df = filtered_df[filtered_df['designation'] == sel_role]
        if sel_loc != "All": filtered_df = filtered_df[filtered_df['location'] == sel_loc]

        st.caption(f"Showing {len(filtered_df)} matching employees.")

        display_cols = ['employee_name', 'department', 'designation', 'manager_name', 'completeness_rate', 'effectiveness_score', 'engagement_score', 'skill_gap_index', 'voucher_eligibility']
        existing_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(filtered_df[existing_cols], use_container_width=True)

with tab_course:
    with st.expander("Click to Open Course Filters", expanded=False):
        ccol1, ccol2, ccol3 = st.columns(3)

        sel_cat = ccol1.selectbox("Category:", ["All"] + sorted(df_course['category'].dropna().unique().tolist()))
        sel_vendor = ccol2.selectbox("Vendor:", ["All"] + sorted(df_course['vendor'].dropna().unique().tolist()))
        sel_diff = ccol3.selectbox("Difficulty:", ["All"] + sorted(df_course['difficulty_level'].dropna().unique().tolist()))

        filtered_course_df = df_course.copy()
        if sel_cat != "All": filtered_course_df = filtered_course_df[filtered_course_df['category'] == sel_cat]
        if sel_vendor != "All": filtered_course_df = filtered_course_df[filtered_course_df['vendor'] == sel_vendor]
        if sel_diff != "All": filtered_course_df = filtered_course_df[filtered_course_df['difficulty_level'] == sel_diff]

        st.caption(f"Showing {len(filtered_course_df)} matching courses.")

        course_display_cols = ['course_name', 'category', 'vendor', 'delivery_mode', 'difficulty_level', 'completion_rate_pct', 'pass_rate_pct', 'avg_score_passed', 'avg_feedback_rating']
        existing_course_cols = [c for c in course_display_cols if c in filtered_course_df.columns]
        st.dataframe(filtered_course_df[existing_course_cols], use_container_width=True)

st.divider()

# ---------------------------------------------------------
# SECTION B: AI ANALYTICS AGENT
# ---------------------------------------------------------
st.subheader("💬 Ask the AI for Deep Analytics & Charts")
st.markdown("Ask complex questions to generate graphs, find correlations, and download PDF reports. The AI can use **employee data** (df1) and **course effectiveness data** (df2).")

st.markdown("**Try asking:**")
scol1, scol2, scol3 = st.columns(3)


def set_prompt(text):
    st.session_state.prompt_trigger = text


if scol1.button("📚 Average effectiveness by Dept? (Bar chart)"):
    set_prompt("Using df1 (employee data), group the employees by department and calculate the exact average effectiveness score for each department. Generate a bar chart.")
if scol2.button("🎟️ Compare Engagement: Meera vs Rohan? (Bar chart)"):
    set_prompt("Using df1 (employee data), calculate the exact average engagement score for Meera Reddy's team and Rohan Joshi's team. Generate a bar chart comparing them.")
if scol3.button("📊 Avg Pass Rate by Difficulty? (Bar chart)"):
    set_prompt("Using df2 (course effectiveness data), calculate the exact average pass_rate_pct grouped by difficulty_level. Generate a bar chart.")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'prompt_trigger' not in st.session_state:
    st.session_state.prompt_trigger = None

agent = get_ai_agent()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_path") and os.path.exists(msg["image_path"]):
            st.image(msg["image_path"])
        if msg.get("pdf_path") and os.path.exists(msg["pdf_path"]):
            with open(msg["pdf_path"], "rb") as f:
                st.download_button("📄 Download PDF", f, file_name=os.path.basename(msg["pdf_path"]), mime="application/pdf", key=msg["pdf_path"])

user_input = st.chat_input("E.g., 'Compare average pass rate for Beginner vs Advanced courses. Generate a chart.'")
active_prompt = user_input or st.session_state.prompt_trigger

if active_prompt:
    st.session_state.prompt_trigger = None

    st.session_state.chat_history.append({"role": "user", "content": active_prompt})
    with st.chat_message("user"):
        st.markdown(active_prompt)

    with st.chat_message("assistant"):
        with st.spinner("AI is analyzing the dataset..."):
            try:
                # Unique chart filename for THIS message (avoids stale/shared image across history)
                msg_id = abs(hash(active_prompt + str(len(st.session_state.chat_history))))
                chart_filename = f"chart_{msg_id}.png"
                if os.path.exists(chart_filename):
                    os.remove(chart_filename)

                # --- TRY DETERMINISTIC HANDLERS FIRST ---
                handled, det_text, det_table, det_chart_spec = try_handle_employee_list_request(active_prompt, df_emp, df_course)
                if not handled:
                    handled, det_text, det_table, det_chart_spec = try_handle_completeness_kpi_request(active_prompt, df_emp)

                if handled:
                    text_answer = det_text
                    image_path = None

                    # Generate chart if a chart spec was produced
                    if det_chart_spec:
                        if generate_chart(det_chart_spec, chart_filename):
                            image_path = chart_filename

                    # Build PDF with table (and chart, if generated)
                    pdf_file = f"report_{msg_id}.pdf"
                    generate_table_pdf(text_answer, det_table, pdf_file, graph_filename=chart_filename)

                    st.markdown(text_answer)
                    if image_path and os.path.exists(image_path):
                        st.image(image_path)
                    st.dataframe(det_table, use_container_width=True)

                    with open(pdf_file, "rb") as f:
                        st.download_button("📄 Download AI PDF Report", f, file_name="analytics_report.pdf", mime="application/pdf", key=f"dl_{pdf_file}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": text_answer,
                        "image_path": image_path,
                        "pdf_path": pdf_file
                    })

                else:
                    wants_chart = bool(re.search(r'\b(chart|graph|plot|visuali[sz]e|bar|scatter|histogram|compare)\b', active_prompt, flags=re.IGNORECASE))

                    if wants_chart:
                        structured_prompt = (
                            active_prompt
                            + "\n\nCalculate the exact numeric results using pandas on df1 and/or df2 as appropriate."
                            + " Do NOT use matplotlib or generate any chart yourself."
                            + " After computing, your Final Answer MUST include TWO parts:"
                            + " (1) A short plain-English sentence stating the calculated numbers."
                            + " (2) On a new line, a JSON object (and ONLY valid JSON, no markdown formatting)"
                            + " describing the chart to draw, in this exact schema:\n"
                            + '{"chart_type": "bar", "title": "...", "x_label": "...", "y_label": "...",'
                            + ' "categories": ["Label1", "Label2"], "values": [12.34, 56.78]}\n'
                            + "Use chart_type 'pie' (with \"categories\" and \"values\" arrays, no x_label/y_label needed)"
                            + " if the user explicitly asks for a pie chart."
                            + " Use chart_type 'bar' for comparisons/grouped counts/grouped averages,"
                            + ' \'scatter\' with "x_values" and "y_values" arrays for correlation requests,'
                            + " and 'hist' with a \"values\" array for distribution requests."
                            + " The JSON must contain the REAL computed numbers, not placeholders."
                        )
                    else:
                        structured_prompt = active_prompt + "\n\nCalculate the exact numbers using pandas on df1 and/or df2 as appropriate."

                    # Execute AI
                    response = agent.invoke({"input": structured_prompt})
                    raw_output = response["output"]

                    if raw_output.strip().lower().startswith("agent stopped"):
                        raw_output = (
                            "I had trouble fully completing this analysis (it required too many steps). "
                            "Please try rephrasing the question more simply, or break it into smaller parts."
                        )

                    # Strip any raw python/markdown code blocks
                    cleaned = re.sub(r'```python.*?```', '', raw_output, flags=re.DOTALL)
                    cleaned = re.sub(r'```.*?```', '', cleaned, flags=re.DOTALL)
                    cleaned = cleaned.strip()

                    chart_created = False
                    image_path = None

                    if wants_chart:
                        chart_spec = extract_json_block(cleaned)
                        if chart_spec:
                            chart_created = generate_chart(chart_spec, chart_filename)
                            if chart_created:
                                image_path = chart_filename
                            # Remove the raw JSON block from the displayed text
                            cleaned = re.sub(r'\{[^{}]*\}', '', cleaned, flags=re.DOTALL).strip()

                        # --- RETRY ONCE IF CHART SPEC WAS MISSING/INVALID ---
                        if not chart_created:
                            retry_prompt = (
                                "Your previous response did not include a valid JSON chart specification. "
                                "Re-answer this request: " + active_prompt + "\n\n"
                                "Compute the exact numbers with pandas, then respond with a short sentence "
                                "followed by ONLY a valid JSON object (no markdown, no code fences) in this schema:\n"
                                '{"chart_type": "bar", "title": "...", "x_label": "...", "y_label": "...", '
                                '"categories": ["Label1", "Label2"], "values": [12.34, 56.78]}\n'
                                "Use real computed numbers."
                            )
                            try:
                                retry_response = agent.invoke({"input": retry_prompt})
                                retry_output = retry_response["output"]
                                retry_cleaned = re.sub(r'```.*?```', '', retry_output, flags=re.DOTALL).strip()
                                retry_spec = extract_json_block(retry_cleaned)
                                if retry_spec:
                                    chart_created = generate_chart(retry_spec, chart_filename)
                                    if chart_created:
                                        image_path = chart_filename
                                        cleaned = re.sub(r'\{[^{}]*\}', '', retry_cleaned, flags=re.DOTALL).strip()
                            except Exception:
                                pass

                    text_answer = cleaned if cleaned else "Analysis complete."

                    # Generate PDF (embeds chart if one was created)
                    pdf_file = generate_pdf_report(active_prompt, text_answer, chart_filename, msg_id)

                    # Show results in UI
                    st.markdown(text_answer)
                    if image_path and os.path.exists(image_path):
                        st.image(image_path)

                    if os.path.exists(pdf_file):
                        with open(pdf_file, "rb") as f:
                            st.download_button("📄 Download AI PDF Report", f, file_name="analytics_report.pdf", mime="application/pdf", key=f"dl_{pdf_file}")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": text_answer,
                        "image_path": image_path,
                        "pdf_path": pdf_file
                    })

            except Exception as e:
                error_msg = f"**Error during AI Execution:**\n\n`{str(e)}`\n\n*Tip: Ensure you are running 'llama3.1' in Ollama.*"
                st.error(error_msg)
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg, "image_path": None, "pdf_path": None})