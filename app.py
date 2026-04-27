import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
# Import ML Engines
try:
    from ml_engine import MLEngine
    from sentiment_engine import SentimentEngine
except ImportError:
    pass # Will handle gracefully if modules missing
import os


# --- Page Configuration ---
st.set_page_config(
    page_title="FinAI - Indian Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Dark Theme & Styling ---
st.markdown("""
<style>
    /* General Settings */
    [data-testid="stAppViewContainer"] {
        background-color: #0e0e0e;
        color: #e0e0e0;
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem !important;
        font-weight: 600;
        color: #ffffff;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 1rem !important;
        color: #a0a0a0;
    }
    
    /* Input Field */
    .stTextInput > div > div > input {
        background-color: #1e1e1e;
        color: white;
        border: 1px solid #333;
        border-radius: 5px;
    }
    
    /* Cards/Containers (simulated) */
    .css-1r6slb0 {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    
    /* Plotly Chart Background */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---

def format_indian_currency(value):
    """Format large numbers into Lakhs/Crores for Indian context."""
    if value is None:
        return "N/A"
    
    try:
        val = float(value)
    except:
        return value

    if val >= 10000000:
        return f"₹{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"₹{val/100000:.2f} L"
    else:
        return f"₹{val:,.2f}"

def format_volume(value):
    """Format volume into Indian numeric system."""
    if value is None:
        return "N/A"
    
    try:
        val = float(value)
    except:
        return value

    if val >= 10000000:
        return f"{val/10000000:.2f} Cr"
    elif val >= 100000:
        return f"{val/100000:.2f} L"
    else:
        return f"{val:,.0f}"

@st.cache_data(ttl=60) # Cache for 1 min
def fetch_stock_data(ticker, period="5y"):
    """Fetch stock history and info."""
    stock = yf.Ticker(ticker)
    # Fetch history
    history = stock.history(period=period)
    # Fetch info (use fast_info for more reliability on some attrs if needed, but info is standard)
    info = stock.info
    return history, info

def fetch_indices():
    """Fetch NIFTY 50 and SENSEX real-time data using fast_info."""
    
    def _get_live(symbol):
        """Get live price via fast_info (real-time, not EOD)."""
        try:
            fi = yf.Ticker(symbol).fast_info
            price = float(fi.get("lastPrice", 0))
            prev_close = float(fi.get("previousClose", 0))
            if prev_close > 0:
                change = price - prev_close
                pct_change = (change / prev_close) * 100
            else:
                change, pct_change = 0, 0
            return price, change, pct_change
        except Exception:
            return 0, 0, 0

    n_price, n_change, n_pct = _get_live("^NSEI")
    s_price, s_change, s_pct = _get_live("^BSESN")
    
    return (n_price, n_change, n_pct), (s_price, s_change, s_pct)

@st.cache_resource
def get_ml_engine(ticker):
    return MLEngine(ticker)

@st.cache_resource
def get_sentiment_engine():
    return SentimentEngine()

# --- Main Layout ---

# Top Bar: Indices
st.title("FinAI 🇮🇳")
st.markdown("### AI-Powered Real-Time Indian Stock Market Analysis")

@st.fragment(run_every=5)
def live_indices_ticker():
    """Auto-refreshing indices strip — polls every 5 seconds."""
    try:
        (n_price, n_change, n_pct), (s_price, s_change, s_pct) = fetch_indices()
        
        col1, col2, col3, col4 = st.columns([2, 2, 6, 2])
        
        with col1:
            st.metric("NIFTY 50", f"{n_price:,.2f}", f"{n_change:+.2f} ({n_pct:+.2f}%)")
            
        with col2:
            st.metric("SENSEX", f"{s_price:,.2f}", f"{s_change:+.2f} ({s_pct:+.2f}%)")

    except Exception as e:
        st.error(f"Failed to load indices: {e}")

live_indices_ticker()


st.markdown("---")

# Input Section
col_input, col_spacing = st.columns([1, 2])
with col_input:
    ticker_input = st.text_input("Enter Symbol (e.g. INFY, RELIANCE)", value="INFY").upper().strip()

# Logic to append .NS
if ticker_input:
    if not (ticker_input.endswith(".NS") or ticker_input.endswith(".BO")):
        ticker_full = f"{ticker_input}.NS"
    else:
        ticker_full = ticker_input
else:
    ticker_full = "INFY.NS"


# Main Content
if ticker_full:
    try:
        history, info = fetch_stock_data(ticker_full)
        
        if history.empty:
            st.warning(f"No data found for {ticker_full}. Please check the symbol.")
        else:
            # Header
            st.header(info.get('longName', ticker_full))
            st.caption(f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}")
            
            # Current Data
            current_price = history['Close'].iloc[-1]
            prev_close = history['Close'].iloc[-2] if len(history) > 1 else current_price
            price_change = current_price - prev_close
            pct_change = (price_change / prev_close) * 100
            
            day_high = history['High'].iloc[-1]
            day_low = history['Low'].iloc[-1]
            volume = history['Volume'].iloc[-1]
            
            # KPI Metrics
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Current Price (LTP)", f"₹{current_price:,.2f}", f"{price_change:+.2f} ({pct_change:+.2f}%)")
            with kpi2:
                st.metric("Day High", f"₹{day_high:,.2f}")
            with kpi3:
                st.metric("Day Low", f"₹{day_low:,.2f}")
            with kpi4:
                st.metric("Volume", format_volume(volume))

            # Charts
            st.subheader("Price History")
            
            # Chart controls
            chart_type = "Candlestick" # Could add line option
            
            # Create Plotly Figure
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=history.index,
                open=history['Open'],
                high=history['High'],
                low=history['Low'],
                close=history['Close'],
                name=ticker_full
            ))
            
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=600,
                xaxis_rangeslider_visible=False,
                title=f"{ticker_full} - 5 Year History"
            )
            
            # Add range selector
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]),
                    bgcolor="#1e1e1e",
                    activecolor="#4caf50"
                )
            )
            
            st.plotly_chart(fig, width="stretch")
            
            # Fundamental Data (Optional Bonus)
            st.subheader("Fundamentals")
            f1, f2, f3 = st.columns(3)
            with f1:
                st.write(f"**Market Cap:** {format_indian_currency(info.get('marketCap'))}")
                st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
            with f2:
                st.write(f"**52 Week High:** ₹{info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.write(f"**52 Week Low:** ₹{info.get('fiftyTwoWeekLow', 'N/A')}")
            with f3: 
                st.write(f"**Book Value:** {info.get('bookValue', 'N/A')}")
                st.write(f"**Dividend Yield:** {info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "N/A")

            st.markdown("---")
            st.subheader("🤖 AI Price Prediction (Regression)")
            
            if st.button("Generate AI Forecast"):
                with st.spinner("Analyzing market sentiment, categorizing news, and training LSTM models..."):
                    try:
                        # 1. Prediction (New Regression Architecture)
                        ml = MLEngine(ticker_full) 
                        future_df = None
                        training_raw_df = None
                        
                        # Check if model exists
                        if ml.model_exists():
                            st.info("Loading existing model for fast inference...")
                            future_df, training_raw_df = ml.predict_realtime()
                        else:
                            st.warning("No model found. Training initial model (this takes ~1 min)...")
                            with st.spinner("Training LSTM on 10 years of historical data..."):
                                success = ml.train_offline()
                                if success:
                                    st.success("Training complete! Generating forecast...")
                                    future_df, training_raw_df = ml.predict_realtime()
                                else:
                                    st.error("Failed to train model (insufficient data).")

                        # 2. Sentiment
                        sent_engine = get_sentiment_engine()
                        sentiment_score, sent_stats, news_list = sent_engine.run(ticker_full)
                        
                        # 3. Enhanced Blending for Regression
                        # Logic: Use sentiment to adjust the TRAJECTORY slightly, 
                        # but trust the model's price level.
                        
                        if future_df is not None:
                            # Apply sentiment adjustment
                            volatility_factor = 0.02 # 2% max swing based on news for price adj
                            adjustment = sentiment_score * volatility_factor
                            
                            # Original LSTM for comparison
                            future_df['Base LSTM'] = future_df['Predicted Price']
                            
                            # Explicit Merging Logic
                            final_forecast = future_df['Predicted Price'] * (1 + adjustment)
                            future_df['Predicted Price'] = final_forecast
                            
                            # Calculate implied Direction
                            initial_price = future_df['Predicted Price'].iloc[0]
                            final_price = future_df['Predicted Price'].iloc[-1]
                            change = final_price - current_price
                            
                            # Market Mood Badge
                            if sentiment_score > 0.15:
                                mood = "BULLISH"
                                mood_color = "#00e676" # Bright Green
                                bg_color = "rgba(0, 230, 118, 0.2)"
                            elif sentiment_score < -0.15:
                                mood = "BEARISH"
                                mood_color = "#ff1744" # Bright Red
                                bg_color = "rgba(255, 23, 68, 0.2)"
                            else:
                                mood = "NEUTRAL"
                                mood_color = "#2979ff" # Blue
                                bg_color = "rgba(41, 121, 255, 0.2)"

                            # Top Mood Section
                            st.markdown(f"""
                            <div style="display: flex; align-items: center; justify-content: space-between; 
                                        padding: 20px; background-color: {bg_color}; border: 2px solid {mood_color}; 
                                        border-radius: 12px; margin-bottom: 20px;">
                                <div>
                                    <h3 style="margin:0; color: #fff;">Market Mood: <span style="color:{mood_color}">{mood}</span></h3>
                                    <p style="margin:5px 0 0 0; color: #ddd;">Composite Sentiment Score: <b>{sentiment_score:.2f}</b></p>
                                </div>
                                <div style="text-align: right;">
                                    <span style="color:#00e676">▲ {sent_stats['positive']} Pos</span> &nbsp;|&nbsp; 
                                    <span style="color:#ff1744">▼ {sent_stats['negative']} Neg</span> &nbsp;|&nbsp; 
                                    <span style="color:#2979ff">● {sent_stats['neutral']} Neu</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display Forecast Chart with Comparison
                            st.write("### Past & Future 5-Day Price Forecast")
                            
                            # Comparison Metrics (Since index 0 is 5 days ago, index 5 is Tomorrow)
                            latest_pred = future_df['Predicted Price'].iloc[5] if len(future_df) > 5 else future_df['Predicted Price'].iloc[0]
                            st.metric("Tomorrow's Predicted Close", f"₹{latest_pred:,.2f}", 
                                      f"{latest_pred - current_price:+.2f} ({(latest_pred - current_price)/current_price:.2%})")

                            fig_pred = go.Figure()
                            # Historical (Last 60 days)
                            hist_subset = history.iloc[-60:]
                            fig_pred.add_trace(go.Scatter(x=hist_subset.index, y=hist_subset['Close'], name='Historical', line=dict(color='gray', width=1)))
                            
                            # Base LSTM (Optional)
                            # fig_pred.add_trace(go.Scatter(x=future_df.index, y=future_df['Base LSTM'], name='Base LSTM', line=dict(color='white', width=1, dash='dot')))

                            # Final Prediction
                            fig_pred.add_trace(go.Scatter(x=future_df.index, y=future_df['Predicted Price'], name='AI Forecast', line=dict(color=mood_color, width=3)))
                            
                            fig_pred.update_layout(
                                template="plotly_dark", 
                                paper_bgcolor='rgba(0,0,0,0)', 
                                plot_bgcolor='rgba(0,0,0,0)', 
                                title=f"Projected Price Trend: {ticker_full} (Regression Model)", 
                                hovermode="x unified"
                            )
                            fig_pred.update_xaxes(showspikes=False)
                            fig_pred.update_yaxes(showspikes=False)
                            st.plotly_chart(fig_pred, width="stretch")
                            
                            # Training Dataset Display (New Requirement)
                            st.subheader("Training Data Inspection")
                            
                            col_d1, col_d2 = st.columns(2)
                            
                            with col_d1:
                                with st.expander("View Raw OHLCV Data (Source)", expanded=False):
                                    if training_raw_df is not None:
                                        st.dataframe(training_raw_df.tail(100), height=300)
                                        st.download_button("Download Raw CSV", training_raw_df.to_csv(), "raw_data.csv")
                            
                            with col_d2:
                                with st.expander("View Processed Feature Data (Engineered)", expanded=False):
                                    # Load processed data from disk to show exactly what was used
                                    processed_path = os.path.join("saved_data", f"{ticker_full}_processed.csv")
                                    if os.path.exists(processed_path):
                                        proc_df = pd.read_csv(processed_path, index_col=0)
                                        st.dataframe(proc_df.tail(100), height=300)
                                        st.download_button("Download Processed CSV", proc_df.to_csv(), "processed_data.csv")
                                    else:
                                        st.info("Processed data not found on disk.")

                            # --- MODEL PERFORMANCE SECTION ---
                            st.markdown("---")
                            st.subheader("Model Performance (Regression Metrics)")
                            
                            # Primary: saved_data/ metrics (includes MAPE); fallback: saved_models/
                            _sd_metrics_path = os.path.join("saved_data", f"{ticker_full}_metrics.json")
                            _sm_metrics_path = ml.metrics_path
                            metrics_path = _sd_metrics_path if os.path.exists(_sd_metrics_path) else _sm_metrics_path
                            training_plot_path = os.path.join(ml.model_dir, f"{ticker_full}_training_plot.png")

                            if os.path.exists(metrics_path):
                                import json
                                with open(metrics_path, 'r') as f:
                                    metrics = json.load(f)
                                
                                # Row 1: Core metrics
                                m1, m2, m3, m4 = st.columns(4)
                                with m1: st.metric("MSE", f"{metrics.get('mse', 0):.4f}")
                                with m2: st.metric("MAE", f"{metrics.get('mae', 0):.4f}")
                                with m3: st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
                                with m4: st.metric("R2 Score", f"{metrics.get('r2_score', 0):.4f}")

                                # Row 2: MAPE + Quality Grade
                                mape_val = metrics.get('mape', None)
                                if mape_val is not None:
                                    m5, m6, m7 = st.columns([1, 1, 2])
                                    with m5:
                                        st.metric("MAPE", f"{mape_val:.2f}%")
                                    with m6:
                                        # Quality grade badge
                                        if mape_val < 5:
                                            grade_label, grade_color = "Excellent", "#00e676"
                                        elif mape_val < 10:
                                            grade_label, grade_color = "Good", "#ffab00"
                                        else:
                                            grade_label, grade_color = "Poor", "#ff1744"
                                        st.markdown(f"""
                                        <div style="margin-top: 6px;">
                                            <span style="font-size: 0.9rem; color: #a0a0a0;">Model Quality</span><br/>
                                            <span style="font-size: 1.6rem; font-weight: 700; color: {grade_color};">{grade_label}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with m7:
                                        # Improvement log summary (if exists)
                                        _log_path = os.path.join("saved_data", f"{ticker_full}_improvement_log.json")
                                        if os.path.exists(_log_path):
                                            with open(_log_path, 'r') as lf:
                                                imp_log = json.load(lf)
                                            if imp_log:
                                                last = imp_log[-1]
                                                st.markdown(f"""
                                                <div style="margin-top: 6px;">
                                                    <span style="font-size: 0.9rem; color: #a0a0a0;">Last Auto-Improvement</span><br/>
                                                    <span style="font-size: 1rem; color: #ccc;">Status: <b>{last['status']}</b> &nbsp;|&nbsp; Before MAPE: {last['before'].get('mape','N/A')}% → After MAPE: {last['after'].get('mape','N/A')}%</span>
                                                </div>
                                                """, unsafe_allow_html=True)
                                
                                # Display Plots
                                if os.path.exists(training_plot_path):
                                    st.image(training_plot_path, caption="Training History & Validation fit")
                            else:
                                st.info("Performance metrics not found.")

                            # Categorized News
                            st.write("### Latest Market News Analysis")
                            
                            if news_list:
                                tab1, tab2, tab3, tab4 = st.tabs(["All News", "Positive", "Negative", "Neutral"])
                                
                                def render_news(n_list):
                                    for news in n_list:
                                        s_label = news.get_label('sentiment_label', 'Neutral') if isinstance(news, dict) and 'get_label' in dir(news) else news.get('sentiment_label', 'Neutral')
                                        s_score = news.get('sentiment_score', 0)
                                        
                                        # Color code
                                        c = "#2979ff"
                                        if s_label == "Positive": c = "#00e676"
                                        if s_label == "Negative": c = "#ff1744"
                                        
                                        st.markdown(f"""
                                        <div style="margin-bottom: 12px; padding: 12px; background-color: #1a1a1a; border-left: 4px solid {c}; border-radius: 4px;">
                                            <a href="{news['link']}" target="_blank" style="text-decoration: none; color: #fff; font-weight: 600; font-size: 1.1em;">{news['title']}</a>
                                            <div style="margin-top: 6px; display: flex; justify-content: space-between; font-size: 0.9em; color: #aaa;">
                                                <span>{news['published']}</span>
                                                <span style="color: {c}; font-weight: bold;">{s_label} ({s_score:.2f})</span>
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                                with tab1:
                                    render_news(news_list)
                                with tab2:
                                    pos_news = [n for n in news_list if n.get('sentiment_label') == 'Positive']
                                    if pos_news: render_news(pos_news)
                                    else: st.info("No positive news detected.")
                                with tab3:
                                    neg_news = [n for n in news_list if n.get('sentiment_label') == 'Negative']
                                    if neg_news: render_news(neg_news)
                                    else: st.info("No negative news detected.")
                                with tab4:
                                    neu_news = [n for n in news_list if n.get('sentiment_label') == 'Neutral']
                                    if neu_news: render_news(neu_news)
                                    else: st.info("No neutral news detected.")

                        else:
                            st.error("Not enough data to generate prediction.")
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")
                        # Print full traceback for debugging if needed
                        import traceback
                        st.text(traceback.format_exc())

    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
