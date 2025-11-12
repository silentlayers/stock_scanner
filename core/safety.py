"""
Production Safety Controls for Automated Trading System
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class SafetyLimits:
    """Production trading limits"""
    max_position_size: int = 5  # Max contracts per trade
    max_daily_trades: int = 3  # Max trades per day
    max_open_positions: int = 5  # Max concurrent positions
    max_daily_loss: float = 500.0  # Max loss per day ($)
    max_account_deployment: float = 0.10  # Max 10% of account
    min_account_balance: float = 5000.0  # Minimum account balance to trade
    require_confirmation: bool = True  # Require manual confirmation before orders
    dry_run_mode: bool = False  # If True, no actual orders placed


class SafetyManager:
    """Manages all production trading safety controls"""

    def __init__(self, limits: Optional[SafetyLimits] = None):
        self.limits = limits or SafetyLimits()
        self.trade_log_file = "logs/trade_history.jsonl"
        self._ensure_log_directory()

    def _ensure_log_directory(self):
        """Create logs directory if it doesn't exist"""
        os.makedirs("logs", exist_ok=True)

    def validate_trade(
        self,
        quantity: int,
        spread_price: float,
        account_balance: float,
        current_positions: int,
        daily_trades_count: int,
        max_loss_per_contract: float = 500.0
    ) -> Tuple[bool, List[str]]:
        """
        Validate if a trade meets all safety requirements

        Args:
            quantity: Number of contracts
            spread_price: Credit received per contract
            account_balance: Current account balance
            current_positions: Number of open positions
            daily_trades_count: Number of trades executed today
            max_loss_per_contract: Maximum loss per contract (default: 500)

        Returns:
            (is_valid, list_of_violations)
        """
        violations = []

        # Check position size
        if quantity > self.limits.max_position_size:
            violations.append(
                f"Position size {quantity} exceeds maximum {self.limits.max_position_size} contracts"
            )

        # Check daily trade limit
        if daily_trades_count >= self.limits.max_daily_trades:
            violations.append(
                f"Daily trade limit reached: {daily_trades_count}/{self.limits.max_daily_trades}"
            )

        # Check open positions limit
        if current_positions >= self.limits.max_open_positions:
            violations.append(
                f"Maximum open positions reached: {current_positions}/{self.limits.max_open_positions}"
            )

        # Check account balance
        if account_balance < self.limits.min_account_balance:
            violations.append(
                f"Account balance ${account_balance:,.2f} below minimum ${self.limits.min_account_balance:,.2f}"
            )

        # Check capital deployment using actual max loss
        capital_at_risk = quantity * max_loss_per_contract
        deployment_pct = (capital_at_risk /
                          account_balance) if account_balance > 0 else 1.0

        if deployment_pct > self.limits.max_account_deployment:
            violations.append(
                f"Capital deployment {deployment_pct*100:.1f}% exceeds maximum {self.limits.max_account_deployment*100:.1f}%"
            )

        is_valid = len(violations) == 0
        return is_valid, violations

    def check_daily_loss_limit(self, account_balance: float, starting_balance: float) -> Tuple[bool, str]:
        """
        Check if daily loss limit has been exceeded

        Returns:
            (within_limits, message)
        """
        daily_pnl = account_balance - starting_balance

        if daily_pnl < -self.limits.max_daily_loss:
            return False, f"Daily loss limit exceeded: ${daily_pnl:,.2f} (limit: ${-self.limits.max_daily_loss:,.2f})"

        return True, ""

    def log_trade(
        self,
        trade_type: str,
        spread_details: Dict,
        order_result: Dict,
        account_balance: float,
        dry_run: bool = False
    ):
        """Log trade execution to file"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade_type": trade_type,
            "dry_run": dry_run,
            "spread": spread_details,
            "order_result": order_result,
            "account_balance": account_balance
        }

        try:
            with open(self.trade_log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.info(
                f"Trade logged: {trade_type} - {spread_details.get('symbol', 'UNKNOWN')}")
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def get_today_trades(self) -> List[Dict]:
        """Get all trades executed today"""
        today = datetime.now().date()
        trades = []

        if not os.path.exists(self.trade_log_file):
            return trades

        try:
            with open(self.trade_log_file, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    trade_date = datetime.fromisoformat(
                        entry["timestamp"]).date()
                    if trade_date == today and not entry.get("dry_run", False):
                        trades.append(entry)
        except Exception as e:
            logger.error(f"Failed to read trade log: {e}")

        return trades

    def get_daily_stats(self) -> Dict:
        """Get today's trading statistics"""
        today_trades = self.get_today_trades()

        return {
            "trades_count": len(today_trades),
            "trades_remaining": max(0, self.limits.max_daily_trades - len(today_trades)),
            "total_contracts": sum(
                t.get("spread", {}).get("quantity", 0) for t in today_trades
            ),
            "timestamp": datetime.now().isoformat()
        }

    def format_safety_report(
        self,
        quantity: int,
        spread_price: float,
        account_balance: float,
        current_positions: int
    ) -> str:
        """Generate human-readable safety report"""
        daily_stats = self.get_daily_stats()
        is_valid, violations = self.validate_trade(
            quantity, spread_price, account_balance,
            current_positions, daily_stats["trades_count"]
        )

        capital_at_risk = quantity * 500
        deployment_pct = (capital_at_risk / account_balance) * \
            100 if account_balance > 0 else 100

        report = f"""
### 🛡️ Production Safety Check

**Trade Parameters:**
- Contracts: {quantity}
- Capital at Risk: ${capital_at_risk:,.2f}
- Account Deployment: {deployment_pct:.1f}%

**Daily Limits:**
- Trades Today: {daily_stats['trades_count']}/{self.limits.max_daily_trades}
- Open Positions: {current_positions}/{self.limits.max_open_positions}

**Status:** {"✅ APPROVED" if is_valid else "❌ BLOCKED"}
"""

        if violations:
            report += "\n**Violations:**\n"
            for v in violations:
                report += f"- ⚠️ {v}\n"

        return report


class DryRunManager:
    """Simulates order execution without placing real orders"""

    def __init__(self):
        self.simulated_orders = []

    def simulate_otoco_order(self, spread: Dict, quantity: int) -> Dict:
        """Simulate OTOCO bracket order"""
        order_id = len(self.simulated_orders) + 1

        simulated_result = {
            "dry_run": True,
            "complex_order_id": f"DRY-{order_id}",
            "trigger_order": {
                "id": f"DRY-TRIGGER-{order_id}",
                "status": "Simulated",
                "quantity": quantity,
                "price": spread["spread_price"]
            },
            "take_profit": {
                "id": f"DRY-TP-{order_id}",
                "status": "Simulated",
                "price": spread["take_profit"]
            },
            "stop_loss": {
                "id": f"DRY-SL-{order_id}",
                "status": "Simulated",
                "price": spread["stop_loss"]
            },
            "timestamp": datetime.now().isoformat()
        }

        self.simulated_orders.append(simulated_result)
        logger.info(f"DRY RUN: Simulated OTOCO order #{order_id}")

        return simulated_result
