import unittest
import pandas as pd
import os
import yfinance as yf
from data_loader import load_price_data, load_multi_asset_data, DATA_DIR

# --- Configuration for testing ---
TICKER = "SPY"
BAD_TICKER = "INVALID_TEST_TICKER"
START = "2023-01-01"
END = "2023-01-10"
# NOTE: Ensure DATA_DIR is defined in data_loader.py (it should be 'data/')
CACHE_FILE = os.path.join(DATA_DIR, f"{TICKER}_data.csv")

class TestDataLoader(unittest.TestCase):
    """
    Tests for the functions in data_loader.py
    """
    
    # 1. Setup/Teardown: Run before and after each test
    def setUp(self): # <--- Correctly indented
        # Clean up any cached file from previous runs
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            
    def tearDown(self): # <--- Correctly indented
        # Ensure the test data is removed after the test is complete
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)

    # --- Test Cases ---

    def test_load_price_data_download(self): # <--- Correctly indented
        """Test that the function downloads, cleans, and saves data."""
        df = load_price_data(TICKER, START, END)
        
        # 1. Check if the output is a non-empty DataFrame
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        
        # 2. Check Data Standardization (Lowercase columns)
        expected_cols = ['open', 'high', 'low', 'close', 'volume', 'adj close'] 
        self.assertTrue(all(col in df.columns for col in expected_cols))
        
        # 3. Check Data Cleaning (No NaNs)
        self.assertFalse(df.isnull().values.any(), "DataFrame contains NaN values")
        
        # 4. Check Caching (File was created)
        self.assertTrue(os.path.exists(CACHE_FILE))

    def test_load_price_data_cache(self): # <--- Correctly indented
        """Test that the function prioritizes the cached file."""
        # First call downloads and saves (creates the cache file)
        load_price_data(TICKER, START, END)
        
        # Capture the modification time of the cache file
        mod_time_initial = os.path.getmtime(CACHE_FILE)
        
        # Second call loads from cache (should NOT redownload)
        df_cached = load_price_data(TICKER, START, END)
        
        # Check that the file modification time is the same (i.e., it wasn't overwritten by a new download)
        mod_time_final = os.path.getmtime(CACHE_FILE)
        self.assertEqual(mod_time_initial, mod_time_final, "Cache file was overwritten, caching failed.")
        self.assertFalse(df_cached.empty)

    def test_load_price_data_bad_ticker(self): # <--- Correctly indented
        """Test graceful failure when an invalid ticker is provided."""
        df = load_price_data(BAD_TICKER, START, END)
        
        # Expect an empty DataFrame to be returned
        self.assertTrue(df.empty)

    def test_load_multi_asset_data(self): # <--- Correctly indented
        """Test that the multi-asset loader returns a dictionary of DataFrames."""
        tickers = [TICKER, BAD_TICKER]
        data_dict = load_multi_asset_data(tickers, START, END)
        
        # 1. Check if the output is a dictionary
        self.assertIsInstance(data_dict, dict)
        
        # 2. Check that only the good ticker was included (graceful failure success)
        self.assertIn(TICKER, data_dict)
        self.assertNotIn(BAD_TICKER, data_dict)
        
        # 3. Check the format of the resulting DataFrame
        self.assertIsInstance(data_dict[TICKER], pd.DataFrame)

# This block allows you to run the tests directly from the command line
if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)