"""
Root entry point for Portfolio VaR Dashboard.
Allows running `streamlit run app.py` directly from the workspace root directory.
"""
import sys
import os

# Delegate directly to portfolio_var_dashboard/app.py
dashboard_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portfolio_var_dashboard")
if dashboard_dir not in sys.path:
    sys.path.insert(0, dashboard_dir)

# Load and execute portfolio_var_dashboard/app.py
app_path = os.path.join(dashboard_dir, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = f.read()

# Execute in current module's namespace
exec(code, globals())
