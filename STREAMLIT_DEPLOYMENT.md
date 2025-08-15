# Streamlit Cloud Deployment Guide

## 🚀 Quick Setup for Streamlit Cloud

### 1. **Data Files Setup**

For Streamlit Cloud compatibility, your data files need to be accessible. We've set up multiple fallback paths:

**Option A: Include data files in repository (Recommended)**
```bash
# Run the copy script to put files in root directory
python copy_data_for_streamlit.py
```

This copies:
- `scraped_data/playwright/unified_articles_complete.json` → `unified_articles_complete.json`
- `data/price_data/combined_daily_prices.csv` → `combined_daily_prices.csv`

**Option B: Upload via dashboard**
- Use the file uploader in the sidebar when data files are missing

### 2. **Repository Structure for Deployment**

Ensure these files are in your repository:
```
├── dashboard/
│   ├── streamlit_dashboard.py          # Main dashboard
│   ├── utils/data_processor.py         # Updated with fallback paths
│   └── ...
├── requirements_streamlit.txt          # Lighter dependencies
├── .streamlit/config.toml             # Streamlit configuration
├── unified_articles_complete.json     # (Copied data file)
├── combined_daily_prices.csv          # (Copied data file)
└── ...
```

### 3. **Streamlit Cloud Settings**

**Main file path:** `dashboard/streamlit_dashboard.py`

**Python version:** 3.11

**Requirements file:** `requirements_streamlit.txt`

### 4. **Environment Variables (Optional)**

No environment variables are required for basic functionality. All data is loaded from files.

### 5. **Debugging**

If deployment fails:

1. **Enable Debug Mode**: Check the "🔍 Debug Mode" in the sidebar
2. **Check File Paths**: The debug mode will show exactly what files are found
3. **Upload Manually**: Use the sidebar file uploader if needed

### 6. **File Size Considerations**

- `unified_articles_complete.json`: ~114MB (large file, may take time to deploy)
- `combined_daily_prices.csv`: ~1MB

Streamlit Cloud supports files up to 200MB, so you're within limits.

### 7. **Performance Tips**

- Data loading is cached in session state
- First load will take time due to large file size
- Subsequent navigation is fast due to caching

## 🔧 Troubleshooting

### Common Issues:

**"Data files not found"**
- Enable debug mode to see file search results
- Ensure files are committed to your repository
- Try the manual upload option

**"Module not found"**
- Check that `requirements_streamlit.txt` includes all dependencies
- Verify Python version compatibility

**"Out of memory"**
- The 126 articles and 1,714 price records should fit comfortably
- If issues persist, consider data pagination

### Debug Information:

Your data structure:
- ✅ **126 articles** (2022-09-07 to 2025-08-11)
- ✅ **1,714 price records** (BTC/ETH, 2021-01-01 to 2025-08-12)
- ✅ **47 technical indicators** per price record
- ✅ **25 article features** per article

## 🎯 Expected Functionality

Once deployed, your dashboard will show:
- Real option flows analysis from 126 articles
- Actual BTC/ETH price correlations over 4+ years
- Comprehensive technical indicators and sentiment analysis
- Interactive visualizations and strategy backtesting

## 📝 Notes

- The `.gitignore` has been updated to ignore the copied data files in root
- Debug utilities (`debug_paths.py`, `copy_data_for_streamlit.py`) are also ignored
- Original data files in subdirectories remain for local development