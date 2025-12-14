import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# ------------------------------------------------------
# 1. PREPROCESSING & SAFETY
# ------------------------------------------------------
def preprocess_function(expr):
    """
    Prepares mathematical strings for evaluation.
    - Converts caret (^) to Python power (**).
    - Adds implicit multiplication for coefficients (e.g., 2x -> 2*x).
    """
    # Replace ^ with ** for exponent
    expr = expr.replace("^", "**")
    
    # Insert * between a number and a variable, e.g., 4x → 4*x
    expr = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", expr)

    # NOTE: We removed the rule that converts "x2" -> "x*2" 
    # This ensures functions like "log10" or variables like "x1" don't break.
    return expr

def safe_eval(func, x):
    """
    Evaluates a mathematical function string safely.
    """
    allowed = {
        # Basic Math
        "np": np,
        "pi": np.pi,
        "e": np.e,
        "x": x,
        # Trigonometry
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        # Exponentials & Logs
        "exp": np.exp,
        "sqrt": np.sqrt,
        "log": np.log,    # Natural log (ln)
        "log10": np.log10, # Base-10 log
        "abs": np.abs,
    }

    try:
        processed = preprocess_function(func)
        return eval(processed, {"__builtins__": {}}, allowed)
    except Exception as e:
        raise ValueError(f"Could not evaluate function: {e}")

# ------------------------------------------------------
# 2. ALGORITHM: GOLDEN SECTION SEARCH
# ------------------------------------------------------
def golden_section_search(func, a, b, tol):
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    resphi = 2 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    # Wrap initial eval in try-except to catch domain errors early (e.g. log of negative)
    try:
        f1 = safe_eval(func, x1)
        f2 = safe_eval(func, x2)
    except Exception as e:
        return None, f"Math Domain Error: {e}"

    iterations = []

    # Iteration loop
    while abs(b - a) > tol:
        iterations.append([a, b, x1, x2, f1, f2, abs(b - a)])

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = safe_eval(func, x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = safe_eval(func, x2)

    return (b + a) / 2, iterations

# ------------------------------------------------------
# 3. STREAMLIT UI LAYOUT
# ------------------------------------------------------
st.set_page_config(page_title="Optimization Tool", layout="wide")

# --- SIDEBAR: Configuration & Team Info ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Interval Settings")
    # Using columns in sidebar for a compact look
    col1, col2 = st.columns(2)
    a = col1.number_input("Left (a):", value=-5.0)
    b = col2.number_input("Right (b):", value=5.0)
    
    st.subheader("Precision")
    tol = st.number_input("Tolerance:", value=0.0001, format="%.6f")
    
    st.markdown("---")
    st.subheader("👥 Group Members")
    st.text("1. Regino Z. Balogo Jr.")
    st.text("2. Rhazel Jay V. Gumacal")
    st.text("3. Christian Angelo R. Panique")

# --- MAIN PAGE ---
st.title("🌟 Golden Section Search Calculator")
st.write("Find the minimum of a unimodal function over an interval $[a, b]$.")

# Documentation Expander (Satisfies Documentation Requirement)
with st.expander("ℹ️ How this Algorithm Works"):
    st.markdown("""
    **Golden Section Search (GSS)** is a technique for finding the extremum (minimum or maximum) 
    of a function by successively narrowing the range of values inside which the extremum exists.
    
    1. **Initialize**: The interval $[a, b]$ is divided using the golden ratio $\phi \\approx 1.618$.
    2. **Evaluate**: The function is evaluated at two test points $x_1$ and $x_2$.
    3. **Reduce**: The sub-interval that cannot contain the minimum is discarded.
    4. **Repeat**: Steps 2-3 are repeated until the interval width is less than the **Tolerance**.
    """)

# Input Section
st.subheader("1. Define Function")
func_input = st.text_input("Enter function f(x):", "x^2 + 2x + 1")

if st.button("🚀 Run Optimization"):
    # Run the algorithm
    xmin, result_data = golden_section_search(func_input, a, b, tol)

    # Check if result is an error message (string) or valid data
    if isinstance(result_data, str):
        st.error(result_data)
    else:
        iter_data = result_data
        min_val = safe_eval(func_input, xmin)

        # --- 2. RESULTS DISPLAY ---
        st.markdown("---")
        st.subheader("2. Optimization Results")
        
        # Using Streamlit Metrics for a cleaner look
        m1, m2, m3 = st.columns(3)
        m1.metric("Approximate Minimum (x)", f"{xmin:.6f}")
        m2.metric("Function Value f(x)", f"{min_val:.6f}")
        m3.metric("Total Iterations", f"{len(iter_data)}")

        # --- 3. VISUALIZATION (Graph) ---
        st.subheader("3. Graphical Analysis")
        
        # Create Graph
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Generate points for plotting
        xs = np.linspace(a, b, 400)
        try:
            ys = [safe_eval(func_input, x) for x in xs]
            
            # Plot the function line
            ax.plot(xs, ys, label=f"f(x) = {func_input}", color="#4da6ff", linewidth=2)
            
            # Plot the minimum point
            ax.scatter([xmin], [min_val], color="red", s=100, zorder=5, label=f"Min (x={xmin:.4f})")
            
            # Graph styling
            ax.set_title("Function Plot & Minimum Point")
            ax.set_xlabel("x")
            ax.set_ylabel("f(x)")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend()
            
            st.pyplot(fig)
            
        except Exception as e:
            st.warning(f"Could not generate graph due to domain errors: {e}")

        # --- 4. DATA TABLE ---
        st.subheader("4. Iteration History")
        with st.expander("View Detailed Iteration Table", expanded=False):
            df = pd.DataFrame(iter_data, columns=["a", "b", "x1", "x2", "f(x1)", "f(x2)", "Interval Width"])
            st.dataframe(df.style.format("{:.6f}"))