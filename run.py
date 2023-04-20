"""
Use: cx_Freeze to create an executable

cxfreeze -c run.py

creates a new folder build
add your streamlit file + all required data or models

click run.exe

"""

# Libaries
import streamlit as st
import pandas as pd
import pickle
import joblib
import subprocess

if __name__ == '__main__':
    subprocess.run("streamlit run pizzaprognosetool.py")