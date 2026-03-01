"""
__main__.py — Entry point for package execution

Enables running the pipeline as:
    python -m run_all
    python run_all.py (direct execution)

This file delegates to the main() function in run_all.py
"""

if __name__ == "__main__":
    from run_all import main

    main()
