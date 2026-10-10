# import necessary libraries.
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
warnings.filterwarnings("ignore")
import yfinance as yf

# Define the symbols for the financial instruments to be analyzed.
SYMBOLS     = ["SOXX", "QQQ"]
START, END  = "2015-01-01", "2026-10-10"
TRADING_DAYS = 252
CHART_DIR   = "D:/Abhay_New/Abhay/CQF/Module 4/Exam3/charts/"
os.makedirs(CHART_DIR, exist_ok=True)

# Download the historical data for the specified symbols using yfinance.
raw = yf.download(SYMBOLS, start=START, end=END, auto_adjust=True, progress=False)
close  = raw["Close"]
volume = raw["Volume"]



