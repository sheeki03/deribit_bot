import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import json
from pathlib import Path
import os
from typing import Dict, List, Optional
import base64

# Configure Streamlit
st.set_page_config(
    page_title="Deribit Option Flows Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules (using relative imports instead of sys.path manipulation)
from app.core.logging import logger
from app.scoring.flow_scorer import flow_scorer
from app.ml.ensemble_scorer import ensemble_scorer
from app.training.training_pipeline import training_pipeline
from app.validation.data_validator import data_validator
from app.ml.xgboost_model import xgboost_model
from app.ml.finbert_model import finbert_model


class DeribitFlowsDashboard:
    """
    Advanced Streamlit dashboard for Deribit Option Flows Intelligence System.
    
    Features:
    - Real-time FlowScore monitoring
    - Multi-tab interface with comprehensive analytics
    - Image gallery with AI analysis
    - Model performance tracking
    - Interactive backtesting
    - System health monitoring
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'recent_scores' not in st.session_state:
            st.session_state.recent_scores = []
        
        if 'dashboard_data' not in st.session_state:
            st.session_state.dashboard_data = {}
        
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def run(self):
        """Main dashboard entry point."""
        # Custom CSS for better styling
        self.inject_custom_css()
        
        # Sidebar
        self.render_sidebar()
        
        # Main dashboard tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📊 Live Feed", "🎯 Analytics", "🖼️ Images", "🤖 Models", "📈 Backtest", "⚙️ System", "📰 Sentiment vs Price"
        ])
        
        with tab1:
            self.render_live_feed_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_images_tab()
        
        with tab4:
            self.render_models_tab()
        
        with tab5:
            self.render_backtest_tab()
        
        with tab6:
            self.render_system_tab()
        
        with tab7:
            self.render_sentiment_vs_price_tab()
    
    def inject_custom_css(self):
        """Inject custom CSS for better UI."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
            margin-bottom: 1rem;
        }
        
        .bullish-score {
            color: #28a745;
            font-weight: bold;
        }
        
        .bearish-score {
            color: #dc3545;
            font-weight: bold;
        }
        
        .neutral-score {
            color: #6c757d;
            font-weight: bold;
        }
        
        .confidence-high {
            color: #28a745;
        }
        
        .confidence-medium {
            color: #ffc107;
        }
        
        .confidence-low {
            color: #dc3545;
        }
        
        .image-analysis {
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar with controls and status."""
        st.sidebar.markdown("## 🎛️ Controls")
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=False)
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 10, 300, 60)
            
            # Auto refresh logic
            if (datetime.now() - st.session_state.last_refresh).total_seconds() > refresh_interval:
                st.session_state.last_refresh = datetime.now()
                st.rerun()
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            self.refresh_dashboard_data()
            st.rerun()
        
        # Asset selection
        st.sidebar.markdown("## 📈 Asset Selection")
        selected_assets = st.sidebar.multiselect(
            "Select Assets",
            options=['BTC', 'ETH'],
            default=['BTC', 'ETH']
        )
        
        # Time range selection
        st.sidebar.markdown("## 📅 Time Range")
        time_range = st.sidebar.selectbox(
            "Select Range",
            options=['Last 24 Hours', 'Last 3 Days', 'Last Week', 'Last Month'],
            index=1
        )
        
        # System status
        st.sidebar.markdown("## 🔧 System Status")
        
        # Model status indicators
        xgb_status = "✅" if xgboost_model.is_trained else "❌"
        finbert_status = "✅" if finbert_model.is_loaded else "❌"
        
        st.sidebar.markdown(f"""
        **Models:**
        - XGBoost: {xgb_status}
        - FinBERT: {finbert_status}
        
        **Last Update:** {st.session_state.last_refresh.strftime('%H:%M:%S')}
        """)
        
        return selected_assets, time_range
    
    def render_live_feed_tab(self):
        """Render live feed of recent FlowScores."""
        st.markdown('<div class="main-header">📊 Live Option Flows Feed</div>', unsafe_allow_html=True)
        
        # Get recent scores
        recent_scores = flow_scorer.get_recent_scores(limit=20)
        
        if not recent_scores:
            st.info("No recent FlowScores available. The system may still be processing data.")
            return
        
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate aggregated metrics
        all_btc_scores = [score['scores'].get('BTC', {}).get('score', 0) for score in recent_scores]
        all_eth_scores = [score['scores'].get('ETH', {}).get('score', 0) for score in recent_scores]
        
        with col1:
            btc_filtered = [s for s in all_btc_scores if s != 0]
            avg_btc_score = np.mean(btc_filtered) if btc_filtered else 0
            st.metric(
                "BTC Avg FlowScore",
                f"{avg_btc_score:.3f}",
                help="Average FlowScore for BTC over recent articles"
            )
        
        with col2:
            eth_filtered = [s for s in all_eth_scores if s != 0]
            avg_eth_score = np.mean(eth_filtered) if eth_filtered else 0
            st.metric(
                "ETH Avg FlowScore",
                f"{avg_eth_score:.3f}",
                help="Average FlowScore for ETH over recent articles"
            )
        
        with col3:
            total_articles = len(recent_scores)
            st.metric(
                "Articles Processed",
                total_articles,
                help="Total articles processed recently"
            )
        
        with col4:
            # Calculate confidence average
            all_confidences = []
            for score in recent_scores:
                for asset_data in score['scores'].values():
                    conf = asset_data.get('confidence', 0)
                    if conf > 0:
                        all_confidences.append(conf)
            
            avg_confidence = np.mean(all_confidences) if all_confidences else 0
            st.metric(
                "Avg Confidence",
                f"{avg_confidence:.1%}",
                help="Average confidence across all predictions"
            )
        
        # Recent articles table
        st.markdown("### 📰 Recent Articles")
        
        # Prepare data for display
        display_data = []
        for article in recent_scores:
            for asset, score_data in article['scores'].items():
                score = score_data.get('score', 0)
                confidence = score_data.get('confidence', 0)
                
                # Format score with color
                if score > 0.1:
                    score_class = 'bullish-score'
                    sentiment = '🔵 Bullish'
                elif score < -0.1:
                    score_class = 'bearish-score'
                    sentiment = '🔴 Bearish'
                else:
                    score_class = 'neutral-score'
                    sentiment = '⚪ Neutral'
                
                display_data.append({
                    'Time': article.get('published_at', 'Unknown'),
                    'Title': article.get('title', 'Untitled')[:50] + '...',
                    'Asset': asset,
                    'FlowScore': f"{score:.3f}",
                    'Confidence': f"{confidence:.1%}",
                    'Sentiment': sentiment,
                    'URL': article.get('url', '')
                })
        
        if display_data:
            df = pd.DataFrame(display_data)
            df = df.sort_values('Time', ascending=False)
            
            # Display with styling
            st.dataframe(
                df[['Time', 'Title', 'Asset', 'FlowScore', 'Confidence', 'Sentiment']],
                use_container_width=True,
                height=400
            )
        
        # Real-time chart
        self.render_flowscore_chart(recent_scores)
    
    def render_flowscore_chart(self, scores_data: List[Dict]):
        """Render interactive FlowScore time series chart."""
        if not scores_data:
            return
        
        st.markdown("### 📈 FlowScore Timeline")
        
        # Prepare data for plotting
        chart_data = []
        for article in scores_data:
            pub_time = pd.to_datetime(article.get('published_at', datetime.now()))
            title = article.get('title', 'Unknown')
            
            for asset, score_data in article['scores'].items():
                chart_data.append({
                    'timestamp': pub_time,
                    'asset': asset,
                    'flowscore': score_data.get('score', 0),
                    'confidence': score_data.get('confidence', 0),
                    'title': title
                })
        
        if not chart_data:
            st.info("No chart data available")
            return
        
        df = pd.DataFrame(chart_data)
        
        # Create interactive plot
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxis=True,
            subplot_titles=['FlowScore Over Time', 'Confidence Levels'],
            vertical_spacing=0.1
        )
        
        # FlowScore plot
        for asset in df['asset'].unique():
            asset_data = df[df['asset'] == asset]
            
            color = '#1f77b4' if asset == 'BTC' else '#ff7f0e'
            
            fig.add_trace(
                go.Scatter(
                    x=asset_data['timestamp'],
                    y=asset_data['flowscore'],
                    mode='lines+markers',
                    name=f'{asset} FlowScore',
                    line=dict(color=color),
                    hovertemplate=f'<b>{asset}</b><br>' +
                                'Time: %{x}<br>' +
                                'FlowScore: %{y:.3f}<br>' +
                                '<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Confidence plot
            fig.add_trace(
                go.Scatter(
                    x=asset_data['timestamp'],
                    y=asset_data['confidence'],
                    mode='lines+markers',
                    name=f'{asset} Confidence',
                    line=dict(color=color, dash='dash'),
                    hovertemplate=f'<b>{asset}</b><br>' +
                                'Time: %{x}<br>' +
                                'Confidence: %{y:.1%}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
        
        # Add horizontal lines for reference
        fig.add_hline(y=0.3, line_dash="dot", line_color="green", 
                     annotation_text="Bullish Threshold", row=1, col=1)
        fig.add_hline(y=-0.3, line_dash="dot", line_color="red", 
                     annotation_text="Bearish Threshold", row=1, col=1)
        fig.add_hline(y=0.7, line_dash="dot", line_color="gray", 
                     annotation_text="High Confidence", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            height=600,
            title="FlowScore and Confidence Timeline",
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="FlowScore", row=1, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_sentiment_vs_price_tab(self, base_dir: Optional[str] = None):
        """Render Weekly Notes sentiment vs price analysis visuals."""
        st.markdown('<div class="main-header">📰 Weekly Notes: Sentiment vs Price</div>', unsafe_allow_html=True)
        
        # File paths
        default_base = os.getenv("WEEKLY_NOTES_RESULTS_DIR", "test_results")
        base = Path(base_dir or default_base)
        base.mkdir(parents=True, exist_ok=True)
        summary_path = base / "weekly_notes_summary.json"
        per_article_csv = base / "weekly_notes_analysis.csv"
        
        # Controls
        colA, colB, colC = st.columns([1,1,2])
        with colA:
            asset = st.selectbox("Asset", options=["BTC", "ETH"], index=0)
        with colB:
            horizon = st.selectbox("Horizon (hours)", options=[24, 72, 168], index=2)
        with colC:
            st.caption("Data sources: per-article analysis CSV and aggregated summary JSON.")
        
        # Load per-article rows
        rows_df = None
        if per_article_csv.exists():
            try:
                rows_df = pd.read_csv(per_article_csv)
            except Exception as e:
                st.warning(f"Failed to read per-article CSV: {e}")
        else:
            st.info(f"Per-article CSV not found at {per_article_csv}")
        
        # Load summary JSON
        summary = None
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception as e:
                st.warning(f"Failed to read summary JSON: {e}")
        else:
            st.info(f"Summary JSON not found at {summary_path}")
        
        # Bucket stats and hit rates
        st.markdown("### 📊 Bucket Stats and Hit Rates")
        if summary and "assets" in summary and asset in summary["assets"]:
            try:
                selected_h = int(horizon)
            except Exception:
                selected_h = horizon
            try:
                summary_h = int(summary.get("horizon_hours")) if summary.get("horizon_hours") is not None else None
            except Exception:
                summary_h = summary.get("horizon_hours")
            if summary_h != selected_h:
                st.info("Summary horizon does not match the selected horizon. Adjust selection or regenerate summary.")
            
            if summary_h == selected_h:
                sec = summary["assets"][asset]
                bs = pd.DataFrame(sec.get("bucket_stats", []))
                hr = pd.DataFrame(sec.get("hit_rates", []))
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Bucket Stats")
                    if not bs.empty:
                        st.dataframe(bs, use_container_width=True)
                    else:
                        st.info("No bucket stats available.")
                with col2:
                    st.markdown("#### Hit Rates")
                    if not hr.empty:
                        st.dataframe(hr, use_container_width=True)
                    else:
                        st.info("No hit rates available.")
        else:
            st.info("Summary not available or horizon mismatch. Run the analysis pipeline first.")
        
        # Rolling trends chart
        st.markdown("### 📈 4-Week Rolling Average Returns by Sentiment Bucket")
        if summary and "assets" in summary and asset in summary["assets"]:
            trends = pd.DataFrame(summary["assets"][asset].get("rolling_4w_trends", []))
            if not trends.empty:
                # Filter horizon implicitly matched by compute (uses selected horizon file)
                fig = px.line(
                    trends,
                    x="week",
                    y="rolling_avg_return",
                    color="sentiment_bucket",
                    markers=True,
                    title=f"{asset} Rolling Avg Returns ({horizon}h)"
                )
                fig.update_layout(height=400, legend_title_text="Sentiment")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rolling trends available.")
        else:
            st.info("Rolling trends not available.")
        
        # Per-article table (optional)
        st.markdown("### 📰 Per-Article Results (preview)")
        if rows_df is not None and not rows_df.empty:
            ret_col = f"{asset}_ret_{horizon}h"
            cols = [c for c in ["published_at_utc", "title", "sentiment", "confidence", ret_col, "url"] if c in rows_df.columns]
            view = rows_df[cols].copy() if cols else rows_df.head(50)
            st.dataframe(view.head(200), use_container_width=True)
        else:
            st.info("Per-article CSV not loaded.")

        # Image display (URLs and/or local paths)
        st.markdown("### 🖼️ Image Previews")
        def _is_url(s: str) -> bool:
            return isinstance(s, str) and (s.startswith("http://") or s.startswith("https://"))

        def _resolve_local(p: str | Path) -> Optional[Path]:
            try:
                pth = Path(p)
                if pth.is_file():
                    return pth
                # try relative to base dir
                cand = base / pth
                if cand.is_file():
                    return cand
                # try relative to dashboard file location
                here = Path(__file__).parent
                cand2 = (here / pth).resolve()
                return cand2 if cand2.is_file() else None
            except (OSError, PermissionError, ValueError) as e:
                logger.warning(f"Error resolving path {p}: {e}")
                return None

        def _from_summary(summary, asset) -> List[str | Path]:
            """Extract image strings/Paths from the summary asset section."""
            imgs: List[str | Path] = []
            try:
                if summary and isinstance(summary, dict):
                    asset_sec = summary.get("assets", {}).get(asset, {})
                    for key in ["example_images", "images", "sample_images"]:
                        vals = asset_sec.get(key, []) or []
                        if isinstance(vals, list):
                            imgs.extend([v for v in vals if isinstance(v, (str, Path))])
                            break
            except Exception as e:
                logger.warning(f"Error extracting images from summary: {e}")
            return imgs
        
        def _from_rows(rows_df) -> List[str | Path]:
            """Iterate the first 50 rows and collect image values."""
            imgs: List[str | Path] = []
            if rows_df is None or rows_df.empty:
                return imgs
                
            candidates = [
                "image", "image_url", "image_path", "thumbnail", "images"
            ]
            present = [c for c in candidates if c in rows_df.columns]
            
            try:
                for _, r in rows_df.head(50).iterrows():
                    for c in present:
                        v = r.get(c)
                        if pd.isna(v):
                            continue
                        if c == "images":
                            # Handle special "images" column
                            parsed = _parse_images(v)
                            imgs.extend(parsed)
                        else:
                            # Handle single-value columns
                            imgs.append(v)
            except Exception as e:
                logger.warning(f"Error extracting images from rows: {e}")
            return imgs
        
        def _parse_images(v) -> List[str | Path]:
            """Parse a JSON array fallback or pipe-delimited string into a list of strings/Paths."""
            imgs: List[str | Path] = []
            try:
                if isinstance(v, str) and v.strip():
                    # Try JSON first
                    try:
                        arr = json.loads(v)
                        if isinstance(arr, list):
                            imgs.extend([x for x in arr if isinstance(x, (str, Path))])
                    except json.JSONDecodeError:
                        # Not JSON, try pipe-delimited
                        parts = [p.strip() for p in str(v).split("|") if p.strip()]
                        imgs.extend(parts)
            except Exception as e:
                logger.warning(f"Error parsing images value {v}: {e}")
            return imgs
        
        def _collect_images() -> List[str | Path]:
            """Collect images from summary and rows, deduplicate while preserving order."""
            # Collect from both sources
            summary_imgs = _from_summary(summary, asset)
            rows_imgs = _from_rows(rows_df)
            
            # Merge outputs
            all_imgs = summary_imgs + rows_imgs
            
            # Deduplicate while preserving order
            seen = set()
            uniq: List[str | Path] = []
            for it in all_imgs:
                key = str(it)
                if key not in seen:
                    seen.add(key)
                    uniq.append(it)
            
            # Return first 12 entries
            return uniq[:12]

        images = _collect_images()
        if images:
            cols = st.columns(4)
            shown = 0
            for i, it in enumerate(images):
                src: Optional[str | Path] = None
                if isinstance(it, Path):
                    src = _resolve_local(it)
                elif _is_url(str(it)):
                    src = str(it)
                else:
                    # treat as local path string
                    loc = _resolve_local(str(it))
                    src = loc if loc is not None else None
                with cols[i % 4]:
                    if isinstance(src, Path) and src is not None:
                        try:
                            st.image(str(src), use_container_width=True)
                            shown += 1
                        except Exception:
                            st.caption(f"Failed to load image: {src}")
                    elif isinstance(src, str) and _is_url(src):
                        try:
                            st.image(src, use_container_width=True)
                            shown += 1
                        except Exception:
                            st.caption(f"Failed to load URL image: {src}")
                    else:
                        st.caption(f"Image not found or unsupported: {it}")
            if shown == 0:
                st.info("No valid images could be rendered from available sources.")
        else:
            st.info("No image references found in summary or per-article data.")
    
    def render_analytics_tab(self):
        """Render analytics and statistics tab."""
        st.markdown('<div class="main-header">🎯 Analytics & Statistics</div>', unsafe_allow_html=True)
        
        # Get scoring statistics
        scoring_stats = flow_scorer.get_scoring_statistics()
        ensemble_stats = ensemble_scorer.get_scoring_stats()
        
        if not scoring_stats or scoring_stats.get('total_articles', 0) == 0:
            st.info("No analytics data available yet. Process some articles first.")
            return
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📊 Total Articles</h3>
                <h2>{}</h2>
            </div>
            """.format(scoring_stats.get('total_articles', 0)), unsafe_allow_html=True)
        
        with col2:
            btc_count = scoring_stats.get('assets', {}).get('BTC', {}).get('count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>₿ BTC Analyses</h3>
                <h2>{btc_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            eth_count = scoring_stats.get('assets', {}).get('ETH', {}).get('count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Ξ ETH Analyses</h3>
                <h2>{eth_count}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Asset-specific analytics
        st.markdown("### 📈 Asset Performance")
        
        asset_tabs = st.tabs(['BTC Analytics', 'ETH Analytics'])
        
        for i, asset in enumerate(['BTC', 'ETH']):
            with asset_tabs[i]:
                self.render_asset_analytics(asset, scoring_stats.get('assets', {}).get(asset, {}))
        
        # Ensemble model performance
        st.markdown("### 🤖 Model Performance")
        
        if 'recent_stats' in ensemble_stats:
            recent_stats = ensemble_stats['recent_stats']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Mean Score",
                    f"{recent_stats.get('mean_score', 0):.3f}",
                    help="Average FlowScore across all predictions"
                )
            
            with col2:
                st.metric(
                    "Mean Confidence",
                    f"{recent_stats.get('mean_confidence', 0):.1%}",
                    help="Average confidence across all predictions"
                )
            
            with col3:
                st.metric(
                    "Score Volatility",
                    f"{recent_stats.get('score_volatility', 0):.3f}",
                    help="Standard deviation of FlowScores"
                )
            
            with col4:
                bullish_ratio = recent_stats.get('bullish_ratio', 0)
                st.metric(
                    "Bullish Ratio",
                    f"{bullish_ratio:.1%}",
                    help="Percentage of bullish predictions"
                )
        
        # Component performance breakdown
        if 'component_stats' in ensemble_stats:
            self.render_component_performance(ensemble_stats['component_stats'])
    
    def render_asset_analytics(self, asset: str, asset_stats: Dict):
        """Render analytics for a specific asset."""
        if not asset_stats:
            st.info(f"No {asset} data available")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mean_score = asset_stats.get('mean_score', 0)
            score_color = 'bullish-score' if mean_score > 0 else 'bearish-score' if mean_score < 0 else 'neutral-score'
            st.markdown(f"""
            **Mean FlowScore:** <span class="{score_color}">{mean_score:.3f}</span>
            """, unsafe_allow_html=True)
            
            st.metric(
                "Score Std Dev",
                f"{asset_stats.get('std_score', 0):.3f}",
                help="Standard deviation of FlowScores"
            )
        
        with col2:
            st.metric(
                "Min Score",
                f"{asset_stats.get('min_score', 0):.3f}",
                help="Most bearish score recorded"
            )
            
            st.metric(
                "Max Score",
                f"{asset_stats.get('max_score', 0):.3f}",
                help="Most bullish score recorded"
            )
        
        with col3:
            st.metric(
                "Mean Confidence",
                f"{asset_stats.get('mean_confidence', 0):.1%}",
                help="Average prediction confidence"
            )
        
        # Sentiment distribution
        st.markdown(f"#### {asset} Sentiment Distribution")
        
        bullish_ratio = asset_stats.get('bullish_ratio', 0)
        bearish_ratio = asset_stats.get('bearish_ratio', 0)
        neutral_ratio = asset_stats.get('neutral_ratio', 0)
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=['Bullish', 'Bearish', 'Neutral'],
            values=[bullish_ratio, bearish_ratio, neutral_ratio],
            marker_colors=['#28a745', '#dc3545', '#6c757d'],
            hole=0.4
        )])
        
        fig.update_layout(
            title=f"{asset} Sentiment Distribution",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_component_performance(self, component_stats: Dict):
        """Render model component performance breakdown."""
        st.markdown("#### 🔧 Model Component Performance")
        
        if not component_stats:
            st.info("No component performance data available")
            return
        
        # Create comparison chart
        components = list(component_stats.keys())
        means = [component_stats[comp]['mean'] for comp in components]
        stds = [component_stats[comp]['std'] for comp in components]
        
        fig = go.Figure()
        
        # Add bars for mean scores
        fig.add_trace(go.Bar(
            name='Mean Score',
            x=components,
            y=means,
            error_y=dict(type='data', array=stds),
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        ))
        
        fig.update_layout(
            title="Model Component Mean Scores (with Standard Deviation)",
            yaxis_title="Score",
            xaxis_title="Component"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Component details table
        comp_df = pd.DataFrame({
            'Component': components,
            'Mean': [f"{m:.3f}" for m in means],
            'Std Dev': [f"{s:.3f}" for s in stds],
            'Min': [f"{component_stats[comp]['min']:.3f}" for comp in components],
            'Max': [f"{component_stats[comp]['max']:.3f}" for comp in components]
        })
        
        st.dataframe(comp_df, use_container_width=True)
    
    def render_images_tab(self):
        """Render image gallery and analysis tab."""
        st.markdown('<div class="main-header">🖼️ Image Gallery & Analysis</div>', unsafe_allow_html=True)

        # Controls
        colA, colB = st.columns([1,1])
        with colA:
            limit = st.number_input("Max items to display", 1, 200, 20)
        with colB:
            show_raw = st.checkbox("Show raw JSON", value=False)

        # Load analyzed results
        analysis_path = Path("test_results/image_analysis.json")
        if not analysis_path.exists():
            st.info("No image analysis found. Run the image scanner to generate test_results/image_analysis.json.")
            return
        try:
            data = json.loads(analysis_path.read_text())
        except Exception as e:
            st.error(f"Failed to load image analysis results: {e}")
            return

        images = data.get("images", [])
        if not images:
            st.info("image_analysis.json has no entries.")
            return

        # Grid display
        shown = 0
        for rec in images:
            if shown >= limit:
                break
            with st.container():
                cols = st.columns([1,2])
                with cols[0]:
                    p = rec.get("path")
                    caption = rec.get("article") or "image"
                    if isinstance(p, str) and (p.startswith("http://") or p.startswith("https://")):
                        try:
                            st.image(p, caption=caption)
                        except Exception as e:
                            st.text(f"(failed to load URL image: {e})")
                    else:
                        # Resolve local path relative to project if not absolute
                        if isinstance(p, str) and not Path(p).is_absolute():
                            resolved = (Path(__file__).parent / p).resolve()
                        else:
                            resolved = Path(p) if p else None
                        if resolved and resolved.exists():
                            st.image(str(resolved), caption=caption)
                        else:
                            st.text("(image not found)")
                    st.caption(p or "")
                with cols[1]:
                    # Parsed GPT-5 JSON
                    gpt5 = rec.get("gpt5")
                    if gpt5:
                        # Key fields if present
                        chart_type = gpt5.get("chart_type")
                        assets = gpt5.get("assets")
                        timeframe = gpt5.get("timeframe")
                        sentiment = gpt5.get("sentiment")
                        reasoning = gpt5.get("reasoning") or gpt5.get("explanation")
                        st.markdown("#### Vision Analysis")
                        if chart_type:
                            st.write(f"Chart: {chart_type}")
                        if assets:
                            st.write(f"Assets: {assets}")
                        if timeframe:
                            st.write(f"Timeframe: {timeframe}")
                        if sentiment:
                            st.write(f"Sentiment: {sentiment}")
                        key_points = gpt5.get("key_points") or []
                        if isinstance(key_points, list) and key_points:
                            st.write("Key Points:")
                            for kp in key_points[:8]:
                                if isinstance(kp, dict):
                                    st.write(f"- {kp.get('label', 'point')}: {kp.get('value')}")
                                else:
                                    st.write(f"- {kp}")
                        key_text = gpt5.get("key_text")
                        if key_text:
                            with st.expander("Extracted Text (model)"):
                                st.write(key_text)
                        if reasoning and show_raw:
                            with st.expander("Model Reasoning"):
                                st.write(reasoning)
                    else:
                        st.warning("No parsed GPT-5 JSON. See raw output below.")

                    # OCR text
                    ocr = rec.get("ocr_text")
                    if ocr:
                        with st.expander("OCR Text"):
                            st.write(ocr)

                    # Raw output
                    if show_raw:
                        raw = rec.get("gpt5_raw")
                        if raw:
                            with st.expander("Raw Model Output"):
                                st.code(raw)
                    if rec.get("error"):
                        st.error(rec["error"])
            st.divider()
            shown += 1
    
    def render_models_tab(self):
        """Render model status and performance tab."""
        st.markdown('<div class="main-header">🤖 Model Status & Performance</div>', unsafe_allow_html=True)
        
        # Model status overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### XGBoost Model")
            
            if xgboost_model.is_trained:
                model_info = xgboost_model.get_model_info()
                
                st.success("✅ Model is trained and ready")
                
                st.markdown(f"""
                **Model Details:**
                - Version: {model_info.get('model_version', 'Unknown')}
                - Features: {model_info.get('feature_count', 0)}
                - Last Training: {model_info.get('last_training', 'Unknown')}
                """)
                
                # Feature importance
                top_features = model_info.get('top_features', {})
                if top_features:
                    st.markdown("**Top Features:**")
                    for feature, importance in list(top_features.items())[:5]:
                        st.write(f"- {feature}: {importance:.4f}")
            else:
                st.error("❌ XGBoost model not trained")
                st.info("Train the model using the System tab")
        
        with col2:
            st.markdown("### FinBERT Model")
            
            if finbert_model.is_loaded:
                model_info = finbert_model.get_model_info()
                
                st.success("✅ Model is loaded and ready")
                
                st.markdown(f"""
                **Model Details:**
                - Model: {model_info.get('model_name', 'Unknown')}
                - Device: {model_info.get('device', 'Unknown')}
                - Total Predictions: {model_info.get('total_predictions', 0)}
                """)
                
                # Recent performance
                recent_confidence = model_info.get('recent_avg_confidence', 0)
                st.metric("Recent Avg Confidence", f"{recent_confidence:.1%}")
            else:
                st.error("❌ FinBERT model not loaded")
                st.info("Load the model using the System tab")
        
        # Ensemble configuration
        st.markdown("### 🎛️ Ensemble Configuration")
        
        current_weights = ensemble_scorer.model_weights
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            xgb_weight = st.slider(
                "XGBoost Weight",
                0.0, 1.0, current_weights['xgboost'],
                step=0.05,
                help="Weight for XGBoost predictions"
            )
        
        with col2:
            finbert_weight = st.slider(
                "FinBERT Weight", 
                0.0, 1.0, current_weights['finbert'],
                step=0.05,
                help="Weight for FinBERT predictions"
            )
        
        with col3:
            vision_weight = st.slider(
                "Vision Weight",
                0.0, 1.0, current_weights['vision'],
                step=0.05,
                help="Weight for Vision AI predictions"
            )
        
        with col4:
            market_weight = st.slider(
                "Market Context Weight",
                0.0, 1.0, current_weights['market_context'],
                step=0.05,
                help="Weight for market context"
            )
        
        # Weight validation and update
        total_weight = xgb_weight + finbert_weight + vision_weight + market_weight
        
        if abs(total_weight - 1.0) > 0.01:
            st.warning(f"⚠️ Weights must sum to 1.0 (current: {total_weight:.2f})")
        else:
            if st.button("Update Ensemble Weights"):
                new_weights = {
                    'xgboost': xgb_weight,
                    'finbert': finbert_weight,
                    'vision': vision_weight,
                    'market_context': market_weight
                }
                
                try:
                    ensemble_scorer.update_model_weights(new_weights)
                    st.success("✅ Ensemble weights updated successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to update weights: {e}")
    
    def render_backtest_tab(self):
        """Render backtesting interface."""
        st.markdown('<div class="main-header">📈 Backtesting & Performance</div>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🧪 Backtesting Engine
        
        This tab will provide:
        - **Historical Performance**: FlowScore accuracy vs actual returns
        - **Strategy Backtesting**: Simulate trading based on FlowScores
        - **Risk Metrics**: Sharpe ratio, maximum drawdown, hit rates
        - **Regime Analysis**: Performance across different market conditions
        - **Parameter Optimization**: Find optimal thresholds and weights
        
        *Backtesting functionality will be available after sufficient historical data is collected...*
        """)
        
        # Placeholder metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Hit Rate", "0.0%", help="Directional accuracy")
        
        with col2:
            st.metric("Sharpe Ratio", "0.00", help="Risk-adjusted returns")
        
        with col3:
            st.metric("Max Drawdown", "0.0%", help="Largest loss from peak")
        
        with col4:
            st.metric("Total Return", "0.0%", help="Cumulative strategy return")
    
    def render_system_tab(self):
        """Render system administration and monitoring tab."""
        st.markdown('<div class="main-header">⚙️ System Administration</div>', unsafe_allow_html=True)
        
        # Training section
        st.markdown("### 🎓 Model Training")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Start Training Pipeline", type="primary"):
                with st.spinner("Training models... This may take several minutes."):
                    try:
                        # Create new event loop for async training
                        import asyncio
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        try:
                            training_results = loop.run_until_complete(
                                training_pipeline.run_complete_training_pipeline()
                            )
                            st.success("Training completed successfully!")
                            st.json(training_results)
                        finally:
                            loop.close()
                            
                    except Exception as e:
                        st.error(f"Training failed: {e}")
        
        with col2:
            if st.button("📊 Load Training Summary"):
                try:
                    training_summary = training_pipeline.get_training_summary()
                    if training_summary.get('status') != 'no_training_data':
                        st.json(training_summary)
                    else:
                        st.info("No training data available. Run training pipeline first.")
                except Exception as e:
                    st.error(f"Failed to load training summary: {e}")
        
        # Data management
        st.markdown("### 💾 Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Clear Caches"):
                try:
                    ensemble_scorer.clear_cache()
                    data_validator.clear_cache()
                    flow_scorer.clear_processed_articles(keep_recent=50)
                    st.success("Caches cleared successfully!")
                except Exception as e:
                    st.error(f"Cache clearing failed: {e}")
        
        with col2:
            if st.button("📈 Load Models"):
                try:
                    # Load FinBERT
                    finbert_success = finbert_model.load_model()
                    
                    if finbert_success:
                        st.success("✅ FinBERT loaded successfully!")
                    else:
                        st.error("❌ FinBERT loading failed")
                        
                except Exception as e:
                    st.error(f"Model loading failed: {e}")
        
        with col3:
            if st.button("🔄 Reset Session"):
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.rerun()

        # Scoring utilities
        st.markdown("### 🧪 Sample Scoring (Cleaned Articles)")
        col1, col2 = st.columns(2)
        with col1:
            sample_size = st.slider("Sample Size", min_value=5, max_value=50, value=15, step=5,
                                    help="Number of cleaned articles to score from scraped_data/cleaned")
        with col2:
            no_market = st.checkbox("Skip Market Enrichment", value=True,
                                     help="Avoids external price lookups for faster scoring")
        if st.button("⚡ Score Sample Cleaned Articles"):
            try:
                cleaned_path = Path("scraped_data/cleaned/articles_cleaned.json")
                if not cleaned_path.exists():
                    st.error(f"Cleaned file not found: {cleaned_path}")
                else:
                    with cleaned_path.open("r", encoding="utf-8") as f:
                        articles = json.load(f)
                    # Use a subset and map cleaned body_text to expected field
                    subset = []
                    for a in articles[:sample_size]:
                        a = dict(a)
                        if not a.get('body_markdown') and a.get('body_text'):
                            a['body_markdown'] = a['body_text']
                        subset.append(a)

                    # Load FinBERT if not loaded
                    if not finbert_model.is_loaded:
                        finbert_model.load_model()

                    # Run scoring synchronously
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(
                            flow_scorer.score_articles_batch(
                                subset,
                                assets=['BTC','ETH'],
                                max_concurrent=4,
                                enrich_market_data=not no_market
                            )
                        )
                    finally:
                        loop.close()

                    st.success(f"Scored {len(subset)} articles. Check Live Feed and Analytics tabs.")
                    st.session_state.last_refresh = datetime.now()
                    st.rerun()
            except Exception as e:
                st.error(f"Scoring failed: {e}")
        
        # System information
        st.markdown("### 📊 System Information")
        
        system_info = {
            "Models Status": {
                "XGBoost Trained": xgboost_model.is_trained,
                "FinBERT Loaded": finbert_model.is_loaded,
            },
            "Data Validation": data_validator.get_validation_stats(),
            "Flow Scorer": {
                "Processed Articles": len(flow_scorer.processed_articles),
                "Alert Thresholds": flow_scorer.alert_thresholds
            },
            "Session State": {
                "Last Refresh": st.session_state.last_refresh.isoformat(),
                "Dashboard Data Keys": list(st.session_state.dashboard_data.keys())
            }
        }
        
        st.json(system_info)
    
    def refresh_dashboard_data(self):
        """Refresh dashboard data from all sources."""
        try:
            st.session_state.dashboard_data = {
                'last_refresh': datetime.now(),
                'recent_scores': flow_scorer.get_recent_scores(20),
                'scoring_stats': flow_scorer.get_scoring_statistics(),
                'ensemble_stats': ensemble_scorer.get_scoring_stats()
            }
            st.session_state.last_refresh = datetime.now()
        except Exception as e:
            st.error(f"Data refresh failed: {e}")


# Main dashboard execution
if __name__ == "__main__":
    dashboard = DeribitFlowsDashboard()
    dashboard.run()