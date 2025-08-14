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
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from app.analytics.unified_options_analyzer import unified_analyzer
from app.market_data.price_data_loader import price_loader

logger = logging.getLogger(__name__)

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
                 stop_loss: float = -0.5):
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
        
        expiry = (pd.to_datetime(context['date']) + timedelta(days=days_to_expiry)).strftime('%Y-%m-%d')
        
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
        
        # Profit target
        if current_pnl >= self.profit_target:
            return True
        
        # Stop loss
        if current_pnl <= self.stop_loss:
            return True
        
        # Time-based exit (7 days before expiry)
        entry_date = pd.to_datetime(trade.entry_date)
        current_date = pd.to_datetime(context['date'])
        expiry_date = pd.to_datetime(trade.contracts[0].expiration)
        
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
        current_date = pd.to_datetime(current_context['date'])
        expiry_date = pd.to_datetime(contract.expiration)
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
        
        # Generate trading dates
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
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
                    if contracts:
                        trade = BacktestTrade(
                            entry_date=date_str,
                            exit_date=None,
                            symbol=contracts[0].symbol,
                            strategy=strategy_name,
                            contracts=contracts,
                            entry_analysis=asdict(analysis)
                        )
                        open_trades.append(trade)
                
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


# Global instance
backtester = OptionsBacktester()


def run_simple_backtest(asset: str = 'BTC', 
                       start_date: str = '2025-07-01', 
                       end_date: str = '2025-08-12') -> BacktestResults:
    """Run a simple long call backtest."""
    strategy = LongCallStrategy()
    return backtester.backtest_strategy(strategy, asset, start_date, end_date)