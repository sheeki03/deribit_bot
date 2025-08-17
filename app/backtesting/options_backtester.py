"""
Options Strategy Backtesting Framework
Backtests options strategies using historical price data and analysis results.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple, Any
from datetime import datetime, date, timedelta
import logging
import threading
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from app.analytics.unified_options_analyzer import unified_analyzer
from app.market_data.price_data_loader import price_loader

logger = logging.getLogger(__name__)

def _normalize_datetime(dt_obj):
    """Convert datetime to timezone-naive to avoid timezone conflicts."""
    if isinstance(dt_obj, str):
        dt_obj = pd.to_datetime(dt_obj)
    elif hasattr(dt_obj, 'tz') and dt_obj.tz is not None:
        dt_obj = dt_obj.tz_localize(None)
    return dt_obj

@dataclass
class OptionsContract:
    """Represents an options contract."""
    symbol: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    premium: float
    quantity: int = 1
    
@dataclass 
class BacktestTrade:
    """Represents a backtest trade."""
    entry_date: str
    exit_date: Optional[str]
    symbol: str
    strategy: str
    contracts: List[OptionsContract]
    entry_analysis: Dict[str, Any]
    exit_analysis: Optional[Dict[str, Any]] = None
    pnl: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    status: str = 'open'  # 'open', 'closed', 'expired'

@dataclass
class BacktestResults:
    """Results of a backtest."""
    strategy_name: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    max_drawdown: float
    win_rate: float
    avg_winning_trade: float
    avg_losing_trade: float
    sharpe_ratio: Optional[float] = None
    trades: List[BacktestTrade] = None
    
    @property
    def total_return(self) -> float:
        """Calculate total return as percentage.
        
        For options strategies, we calculate return relative to total premium paid.
        If no premium data is available, we use a conservative estimate.
        """
        if not self.trades:
            return 0.0
        
        # Calculate total premium paid (options cost)
        total_premium_paid = 0.0
        for trade in self.trades:
            if trade.contracts:
                for contract in trade.contracts:
                    total_premium_paid += abs(contract.premium * contract.quantity)
        
        # If no premium data, estimate based on average trade size
        if total_premium_paid == 0.0:
            # Conservative estimate: assume each trade risked $1000 on average
            estimated_capital = self.total_trades * 1000.0 if self.total_trades > 0 else 10000.0
            return self.total_pnl / estimated_capital if estimated_capital > 0 else 0.0
        
        # Return as percentage of premium paid
        return self.total_pnl / total_premium_paid if total_premium_paid > 0 else 0.0

class OptionsStrategy(ABC):
    """Abstract base class for options strategies."""
    
    @abstractmethod
    def should_enter(self, context: Dict[str, Any]) -> bool:
        """Determine if strategy should enter a position."""
        pass
    
    @abstractmethod
    def create_position(self, context: Dict[str, Any]) -> List[OptionsContract]:
        """Create the options position."""
        pass
    
    @abstractmethod
    def should_exit(self, context: Dict[str, Any], trade: BacktestTrade) -> bool:
        """Determine if strategy should exit a position."""
        pass
    
    @abstractmethod
    def calculate_pnl(self, trade: BacktestTrade, current_context: Dict[str, Any]) -> float:
        """Calculate current PnL of the position."""
        pass

class LongCallStrategy(OptionsStrategy):
    """Simple long call strategy for backtesting."""
    
    def __init__(self, 
                 volatility_threshold: float = 30.0,
                 max_days_to_expiry: int = 30,
                 profit_target: float = 2.0,
                 stop_loss: float = -0.5,
                 randomize_params: bool = False,
                 random_seed: Optional[int] = None):
        """Initialize strategy with optional parameter randomization.
        
        Args:
            volatility_threshold: Maximum volatility for entry
            max_days_to_expiry: Maximum days to expiration
            profit_target: Profit target multiplier
            stop_loss: Stop loss threshold (negative)
            randomize_params: If True, add random variation to parameters
            random_seed: Random seed for reproducible randomization
        """
        if randomize_params:
            if random_seed is not None:
                np.random.seed(random_seed)
            
            # Add ±10% random variation to parameters
            vol_var = np.random.uniform(0.9, 1.1)
            profit_var = np.random.uniform(0.9, 1.1)
            stop_var = np.random.uniform(0.9, 1.1)
            days_var = int(np.random.uniform(0.8, 1.2))
            
            self.volatility_threshold = volatility_threshold * vol_var
            self.profit_target = profit_target * profit_var
            self.stop_loss = stop_loss * stop_var
            self.max_days_to_expiry = max(7, max_days_to_expiry * days_var)
            
            logger.info(f"Randomized strategy params: vol_thresh={self.volatility_threshold:.1f}, "
                       f"profit_target={self.profit_target:.2f}, stop_loss={self.stop_loss:.2f}, "
                       f"max_days={self.max_days_to_expiry}")
        else:
            self.volatility_threshold = volatility_threshold
            self.max_days_to_expiry = max_days_to_expiry
            self.profit_target = profit_target
            self.stop_loss = stop_loss
    
    def should_enter(self, context: Dict[str, Any]) -> bool:
        """Enter when volatility is low and trend is bullish."""
        analysis = context.get('analysis')
        if not analysis:
            return False
        
        # Low volatility regime
        if analysis.vol_regime not in ['low', 'very_low']:
            return False
        
        # Bullish or neutral sentiment
        if analysis.chart_sentiment == 'bearish':
            return False
        
        # Price near support
        if analysis.distance_to_support_pct > 5:
            return False
            
        return True
    
    def create_position(self, context: Dict[str, Any]) -> List[OptionsContract]:
        """Create ATM call position."""
        analysis = context.get('analysis')
        spot_price = analysis.spot_price
        
        # Use ATM strike
        strike = round(spot_price / 50) * 50  # Round to nearest $50
        
        # Estimate premium (simplified)
        volatility = analysis.realized_vol_30d / 100
        days_to_expiry = self.max_days_to_expiry
        premium = self._estimate_option_premium(spot_price, strike, volatility, days_to_expiry, 'call')
        
        expiry = (_normalize_datetime(pd.to_datetime(context['date'])) + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')
        
        return [OptionsContract(
            symbol=f"{analysis.asset}-{expiry}-C-{strike}",
            strike=strike,
            expiration=expiry,
            option_type='call',
            premium=premium,
            quantity=1
        )]
    
    def should_exit(self, context: Dict[str, Any], trade: BacktestTrade) -> bool:
        """Exit based on profit target, stop loss, or time decay."""
        current_pnl = self.calculate_pnl(trade, context)
        
        # Profit target (absolute threshold)
        if current_pnl >= self.profit_target:
            return True
        
        # Stop loss (absolute threshold, normalize with abs())
        if current_pnl <= -abs(self.stop_loss):
            return True
        
        # Time-based exit (7 days before expiry)
        entry_date = _normalize_datetime(pd.to_datetime(trade.entry_date))
        current_date = _normalize_datetime(pd.to_datetime(context['date']))
        expiry_date = _normalize_datetime(pd.to_datetime(trade.contracts[0].expiration))
        
        if (expiry_date - current_date).days <= 7:
            return True
        
        return False
    
    def calculate_pnl(self, trade: BacktestTrade, current_context: Dict[str, Any]) -> float:
        """Calculate PnL based on Black-Scholes approximation."""
        if not trade.contracts:
            return 0.0
        
        contract = trade.contracts[0]
        current_analysis = current_context.get('analysis')
        
        if not current_analysis:
            return 0.0
        
        spot_price = current_analysis.spot_price
        current_date = _normalize_datetime(pd.to_datetime(current_context['date']))
        expiry_date = _normalize_datetime(pd.to_datetime(contract.expiration))
        days_to_expiry = (expiry_date - current_date).days
        
        if days_to_expiry <= 0:
            # Expired - intrinsic value only
            if contract.option_type == 'call':
                intrinsic = max(spot_price - contract.strike, 0)
            else:
                intrinsic = max(contract.strike - spot_price, 0)
            return (intrinsic - contract.premium) * contract.quantity
        
        # Estimate current premium
        volatility = current_analysis.realized_vol_30d / 100
        current_premium = self._estimate_option_premium(
            spot_price, contract.strike, volatility, days_to_expiry, contract.option_type
        )
        
        return (current_premium - contract.premium) * contract.quantity
    
    def _estimate_option_premium(self, spot: float, strike: float, volatility: float, 
                                days: int, option_type: str) -> float:
        """Simplified option pricing using Black-Scholes approximation."""
        if days <= 0:
            if option_type == 'call':
                return max(spot - strike, 0)
            else:
                return max(strike - spot, 0)
        
        # Simplified Black-Scholes
        time_to_expiry = days / 365.0
        moneyness = spot / strike
        
        # Basic approximation
        if option_type == 'call':
            intrinsic = max(spot - strike, 0)
            time_value = spot * volatility * np.sqrt(time_to_expiry) * 0.4
        else:
            intrinsic = max(strike - spot, 0)
            time_value = spot * volatility * np.sqrt(time_to_expiry) * 0.4
        
        return max(intrinsic + time_value, intrinsic)

class OptionsBacktester:
    """Options strategy backtesting engine."""
    
    def __init__(self):
        self.trades: List[BacktestTrade] = []
        self.results: Optional[BacktestResults] = None
    
    def backtest_strategy(self,
                         strategy: OptionsStrategy,
                         asset: str,
                         start_date: str,
                         end_date: str,
                         strategy_name: str = None) -> BacktestResults:
        """Run backtest for a strategy over a date range.
        
        Args:
            strategy: Options strategy to backtest
            asset: Asset to trade (BTC, ETH)
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_name: Name of the strategy
            
        Returns:
            Backtest results
        """
        if strategy_name is None:
            strategy_name = strategy.__class__.__name__
        
        logger.info(f"Starting backtest for {strategy_name} on {asset} from {start_date} to {end_date}")
        
        # Generate trading dates - ensure timezone-naive processing
        start_dt = _normalize_datetime(pd.to_datetime(start_date))
        end_dt = _normalize_datetime(pd.to_datetime(end_date))
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        trading_dates = [d.strftime('%Y-%m-%d') for d in date_range]
        
        open_trades: List[BacktestTrade] = []
        closed_trades: List[BacktestTrade] = []
        
        for date_str in trading_dates:
            try:
                # Get analysis context for current date
                analysis = unified_analyzer.analyze_options_context(asset, date_str, include_image_analysis=False)
                context = {'date': date_str, 'analysis': analysis}
                
                # Check exits for open trades
                trades_to_close = []
                for trade in open_trades:
                    if strategy.should_exit(context, trade):
                        trades_to_close.append(trade)
                
                # Close trades
                for trade in trades_to_close:
                    trade.exit_date = date_str
                    trade.exit_analysis = asdict(analysis)
                    trade.pnl = strategy.calculate_pnl(trade, context)
                    trade.status = 'closed'
                    closed_trades.append(trade)
                    open_trades.remove(trade)
                
                # Check for new entry
                if strategy.should_enter(context) and len(open_trades) < 5:  # Max 5 concurrent positions
                    contracts = strategy.create_position(context)
                    
                    # Validate contracts before creating trade
                    if (contracts is not None and 
                        isinstance(contracts, list) and 
                        len(contracts) > 0 and 
                        hasattr(contracts[0], 'symbol') and 
                        contracts[0].symbol):
                        
                        trade = BacktestTrade(
                            entry_date=date_str,
                            exit_date=None,
                            symbol=contracts[0].symbol,
                            strategy=strategy_name,
                            contracts=contracts,
                            entry_analysis=asdict(analysis)
                        )
                        open_trades.append(trade)
                    else:
                        logger.warning(f"Invalid contracts returned by strategy.create_position on {date_str}: {contracts}")
                
            except Exception as e:
                logger.warning(f"Error processing {date_str}: {e}")
                continue
        
        # Close any remaining open trades
        final_date = end_date
        final_analysis = unified_analyzer.analyze_options_context(asset, final_date, include_image_analysis=False)
        final_context = {'date': final_date, 'analysis': final_analysis}
        
        for trade in open_trades:
            trade.exit_date = final_date
            trade.exit_analysis = asdict(final_analysis)
            trade.pnl = strategy.calculate_pnl(trade, final_context)
            trade.status = 'closed'
            closed_trades.append(trade)
        
        # Calculate results
        results = self._calculate_results(strategy_name, start_date, end_date, closed_trades)
        
        logger.info(f"Backtest completed: {results.total_trades} trades, {results.win_rate:.1f}% win rate, ${results.total_pnl:.2f} total PnL")
        
        self.trades = closed_trades
        self.results = results
        
        return results
    
    def _calculate_results(self, strategy_name: str, start_date: str, end_date: str, 
                          trades: List[BacktestTrade]) -> BacktestResults:
        """Calculate backtest results from completed trades."""
        if not trades:
            return BacktestResults(
                strategy_name=strategy_name,
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                avg_winning_trade=0.0,
                avg_losing_trade=0.0,
                trades=trades
            )
        
        # Basic statistics
        total_trades = len(trades)
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        
        winning_trades = sum(1 for pnl in pnls if pnl > 0)
        losing_trades = sum(1 for pnl in pnls if pnl < 0)
        
        total_pnl = sum(pnls)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        winning_pnls = [pnl for pnl in pnls if pnl > 0]
        losing_pnls = [pnl for pnl in pnls if pnl < 0]
        
        avg_winning_trade = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_losing_trade = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        
        # Calculate maximum drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = cumulative_pnl - running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        
        # Calculate Sharpe ratio (annualized)
        sharpe_ratio = None
        if len(pnls) > 1:
            daily_returns = np.array(pnls)
            mean_daily = np.mean(daily_returns)  # Assuming risk-free rate ~ 0 for simplicity
            std_daily = np.std(daily_returns, ddof=1)
            if std_daily > 0:
                sharpe_ratio = (mean_daily / std_daily) * np.sqrt(252)  # Annualized using trading days
        
        return BacktestResults(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            sharpe_ratio=sharpe_ratio,
            trades=trades
        )
    
    def get_trade_summary(self) -> pd.DataFrame:
        """Get summary of all trades as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'entry_date': trade.entry_date,
                'exit_date': trade.exit_date,
                'symbol': trade.symbol,
                'strategy': trade.strategy,
                'pnl': trade.pnl,
                'status': trade.status,
                'entry_spot': trade.entry_analysis.get('spot_price', 0),
                'exit_spot': trade.exit_analysis.get('spot_price', 0) if trade.exit_analysis else 0,
                'entry_vol': trade.entry_analysis.get('realized_vol_30d', 0),
                'contracts': len(trade.contracts)
            })
        
        return pd.DataFrame(trade_data)


# Thread-local storage for backtester instances
_thread_local = threading.local()


def get_backtester(fresh_instance: bool = False) -> OptionsBacktester:
    """Get a thread-local OptionsBacktester instance.
    
    Args:
        fresh_instance: If True, creates a new instance and clears caches
    """
    if fresh_instance or not hasattr(_thread_local, 'backtester'):
        # Clear analysis caches for fresh results
        from app.analytics.unified_options_analyzer import unified_analyzer
        unified_analyzer.clear_cache()
        
        # Clear price data cache too
        price_loader.clear_cache()
        
        # Create new instance
        _thread_local.backtester = OptionsBacktester()
        logger.info("Created fresh backtester instance with cleared caches")
    return _thread_local.backtester


def run_simple_backtest(asset: str = 'BTC', 
                       start_date: str = '2025-07-01', 
                       end_date: str = '2025-08-12',
                       fresh_analysis: bool = True,
                       randomize_strategy: bool = False,
                       random_seed: Optional[int] = None,
                       **strategy_params) -> BacktestResults:
    """Run a simple long call backtest.
    
    Args:
        asset: Asset to backtest ('BTC' or 'ETH')
        start_date: Start date for backtest
        end_date: End date for backtest
        fresh_analysis: If True, clears caches for fresh analysis
        randomize_strategy: If True, adds random variation to strategy parameters
        random_seed: Random seed for reproducible randomization
        **strategy_params: Additional strategy parameters (volatility_threshold, profit_target, etc.)
    """
    strategy = LongCallStrategy(
        randomize_params=randomize_strategy,
        random_seed=random_seed,
        **strategy_params
    )
    return get_backtester(fresh_instance=fresh_analysis).backtest_strategy(strategy, asset, start_date, end_date)