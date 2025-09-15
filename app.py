import os
import sys
import json
import time
import random
import asyncio
import logging
import argparse
import pyfiglet
import csv
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from functools import wraps
import locale
from pyquotex.utils.indicators import TechnicalIndicators
import numpy as np
try:
    import tensorflow as tf
except ImportError:
    tf = None

from pyquotex.expiration import (
    timestamp_to_date,
    get_timestamp_days_ago
)
from pyquotex.utils.processor import (
    process_candles,
    get_color,
    aggregate_candle
)
from pyquotex.config import credentials
from pyquotex.stable_api import Quotex

__author__ = "Team Jund Al Nabi"
__version__ = "2.0"

USER_AGENT = "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/119.0"


class Monitor:
    """Monitor class for tracking rolling accuracy and profits."""
    
    def __init__(self, rolling_window=50, csv_file="trade_metrics.csv"):
        self.rolling_window = rolling_window
        self.results_window = deque(maxlen=rolling_window)
        self.profits = []
        self.csv_file = csv_file

        # Write CSV header if file does not exist
        try:
            with open(self.csv_file, 'x', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "rolling_accuracy", "total_profit", "last_trade_profit",
                    "trade_count", "win_streak", "loss_streak", "strategy", "learning_confidence"
                ])
        except FileExistsError:
            pass

    def log_trade(self, correct_prediction, profit, trade_count=0, win_streak=0, loss_streak=0, 
                  strategy="neutral", learning_confidence=0.5):
        """
        Log a trade with comprehensive metrics.
        
        Args:
            correct_prediction: 1 if the trade prediction was correct, 0 if wrong
            profit: amount gained/lost on this trade
            trade_count: total number of trades
            win_streak: current win streak
            loss_streak: current loss streak
            strategy: current trading strategy
            learning_confidence: current learning confidence
        """
        # Update rolling accuracy
        self.results_window.append(correct_prediction)
        rolling_accuracy = sum(self.results_window) / len(self.results_window)

        # Update total profit
        self.profits.append(profit)
        total_profit = sum(self.profits)

        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Print summary
        print(f"📊 [{timestamp}] Rolling Accuracy (last {len(self.results_window)} trades): {rolling_accuracy:.2%}")
        print(f"💰 [{timestamp}] Last Trade Profit: R$ {profit:.2f}, Total Profit: R$ {total_profit:.2f}")
        print(f"📈 [{timestamp}] Trade #{trade_count} | Win Streak: {win_streak} | Loss Streak: {loss_streak}")
        print(f"🧠 [{timestamp}] Strategy: {strategy.upper()} | Learning Confidence: {learning_confidence:.2%}\n")

        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, rolling_accuracy, total_profit, profit,
                trade_count, win_streak, loss_streak, strategy, learning_confidence
            ])

    def get_stats(self):
        """Get current monitoring statistics."""
        if not self.results_window:
            return {
                'rolling_accuracy': 0.0,
                'total_profit': 0.0,
                'total_trades': 0,
                'recent_trades': 0
            }
        
        return {
            'rolling_accuracy': sum(self.results_window) / len(self.results_window),
            'total_profit': sum(self.profits),
            'total_trades': len(self.profits),
            'recent_trades': len(self.results_window)
        }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pyquotex.log')
    ]
)
logger = logging.getLogger(__name__)

LANGUAGE_MESSAGES = {
    "en_US": {
        "private_version_ad": (
            "🌟✨ This is the COMMUNITY version of PyQuotex! ✨🌟\n"
            "🔐  Unlock full power and extra features with our PRIVATE version.\n"
            "📤  For more functionalities and exclusive support, please consider donating to the project.\n"
            "➡️ Contact for donations and private version access: https://t.me/pyquotex/852"
        )
    }
}


def detect_user_language() -> str:
    """Attempts to detect the user's system language."""
    try:
        system_lang = locale.getlocale()[0]
        if system_lang and system_lang.startswith("pt"):
            return "pt_BR"
        return "en_US"
    except Exception:
        return "en_US"


def ensure_connection(max_attempts: int = 5):
    """Decorator to ensure connection before executing function."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            if not self.client:
                logger.error("Quotex API client not initialized.")
                raise RuntimeError("Quotex API client not initialized.")

            if await self.client.check_connect():
                logger.debug("Already connected. Proceeding with operation.")
                return await func(self, *args, **kwargs)

            logger.info("Establishing connection...")
            check, reason = await self._connect_with_retry(max_attempts)

            if not check:
                logger.error(f"Failed to connect after multiple attempts: {reason}")
                raise ConnectionError(f"Failed to connect: {reason}")

            try:
                result = await func(self, *args, **kwargs)
                return result
            finally:
                if self.client and await self.client.check_connect():
                    await self.client.close()
                    logger.debug("Connection closed after operation.")

        return wrapper

    return decorator


class PyQuotexCLI:
    def save_model(self, version_suffix=""):
        """Save the model with versioning and backup capabilities."""
        if self.model is not None:
            import os
            from datetime import datetime
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            # Save with timestamp and version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"trade_model_{timestamp}{version_suffix}.keras"
            model_path = os.path.join('models', model_name)
            
            # Save the model
            self.model.save(model_path)
            
            # Also save as latest
            latest_path = os.path.join('models', 'trade_model_latest.keras')
            self.model.save(latest_path)
            
            # Save training data and metadata
            self.save_training_data()
            
            print(f'[TensorFlow] Model saved to {model_path}')
            print(f'[TensorFlow] Latest model saved to {latest_path}')
            
            # Keep only last 5 model versions
            self.cleanup_old_models()

    def load_model(self):
        """Always create a fresh model for real-time learning."""
        if tf is not None:
            print('[TensorFlow] Creating fresh model for real-time learning')
                self.setup_model()
        else:
            print('[TensorFlow] TensorFlow not available, using rule-based learning')
            self.setup_rule_based_learning()
        
        # Try to load existing patterns
        self.load_learning_patterns()

    def save_training_data(self):
        """Save training data and model metadata."""
        import json
        import os
        
        training_data = {
            'train_data': self.train_data,
            'train_labels': self.train_labels,
            'feature_size': self.feature_size,
            'prev_result': self.prev_result,
            'prev_profit': self.prev_profit,
            'total_trades': len(self.train_data),
            'wins': sum(self.train_labels),
            'losses': len(self.train_labels) - sum(self.train_labels)
        }
        
        os.makedirs('models', exist_ok=True)
        with open(os.path.join('models', 'training_data.json'), 'w') as f:
            json.dump(training_data, f, indent=2)

    def load_training_data(self):
        """Load training data and model metadata."""
        import json
        import os
        
        data_path = os.path.join('models', 'training_data.json')
        if os.path.exists(data_path):
            try:
                with open(data_path, 'r') as f:
                    training_data = json.load(f)
                
                self.train_data = training_data.get('train_data', [])
                self.train_labels = training_data.get('train_labels', [])
                self.feature_size = training_data.get('feature_size', self.feature_size)
                self.prev_result = training_data.get('prev_result', 0)
                self.prev_profit = training_data.get('prev_profit', 0.0)
                
                total_trades = training_data.get('total_trades', 0)
                wins = training_data.get('wins', 0)
                losses = training_data.get('losses', 0)
                
                print(f'[TensorFlow] Loaded training data: {total_trades} trades ({wins} wins, {losses} losses)')
            except Exception as e:
                print(f'[TensorFlow] Failed to load training data: {e}')
                self.train_data = []
                self.train_labels = []

    def cleanup_old_models(self):
        """Keep only the last 5 model versions to save disk space."""
        import os
        import glob
        
        model_files = glob.glob(os.path.join('models', 'trade_model_*.keras'))
        # Remove the latest model from the list since we want to keep it
        model_files = [f for f in model_files if not f.endswith('trade_model_latest.keras')]
        
        if len(model_files) > 5:
            # Sort by modification time and keep only the 5 most recent
            model_files.sort(key=os.path.getmtime, reverse=True)
            for old_model in model_files[5:]:
                try:
                    os.remove(old_model)
                    print(f'[TensorFlow] Cleaned up old model: {os.path.basename(old_model)}')
                except Exception as e:
                    print(f'[TensorFlow] Failed to remove old model {old_model}: {e}')

    def get_model_performance(self):
        """Get current model performance statistics."""
        if not self.train_data or not self.train_labels:
            return "No training data available"
        
        total_trades = len(self.train_labels)
        wins = sum(self.train_labels)
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        return f"Trades: {total_trades} | Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%"

    def immediate_learning_update(self, features, result):
        """Immediate learning update after each trade for real-time adaptation."""
        if self.model is None or len(features) != self.feature_size:
            return
        
        try:
            # Single sample learning with high learning rate for immediate adaptation
            x = np.array([features])
            y = np.array([result])
            
            # Use a higher learning rate for immediate updates
            original_lr = self.model.optimizer.learning_rate
            self.model.optimizer.learning_rate = 0.01  # 10x higher for immediate learning
            
            # Single epoch training on the new sample
            self.model.fit(x, y, epochs=1, verbose=0)
            
            # Restore original learning rate
            self.model.optimizer.learning_rate = original_lr
            
            # Track immediate learning
            if not hasattr(self, 'immediate_updates'):
                self.immediate_updates = 0
            self.immediate_updates += 1
            
            if self.immediate_updates % 10 == 0:  # Show progress every 10 immediate updates
                print(f"[TensorFlow] Immediate learning: {self.immediate_updates} real-time updates applied")
            
            # Save model after each immediate update
            self.save_model("_immediate")
                
        except Exception as e:
            print(f"[TensorFlow] Immediate learning failed: {e}")

    def show_learning_progress(self):
        """Show detailed learning progress and patterns."""
        if not self.train_data or len(self.train_data) < 5:
            return
        
        print(f"\n📈 Learning Progress Analysis:")
        print(f"   Total Trades: {len(self.train_data)}")
        
        # Recent performance analysis
        recent_10 = self.train_labels[-10:] if len(self.train_labels) >= 10 else self.train_labels
        recent_5 = self.train_labels[-5:] if len(self.train_labels) >= 5 else self.train_labels
        
        recent_10_rate = sum(recent_10) / len(recent_10) if recent_10 else 0
        recent_5_rate = sum(recent_5) / len(recent_5) if recent_5 else 0
        
        print(f"   Recent 10 trades: {recent_10_rate:.1%} win rate")
        print(f"   Recent 5 trades: {recent_5_rate:.1%} win rate")
        
        # Pattern detection
        if len(self.train_labels) >= 3:
            last_3 = self.train_labels[-3:]
            if all(x == 0 for x in last_3):
                print(f"   🚨 Pattern: 3 consecutive losses detected!")
            elif all(x == 1 for x in last_3):
                print(f"   🎯 Pattern: 3 consecutive wins detected!")
        
        # Learning trend
        if len(self.train_labels) >= 10:
            first_half = self.train_labels[:len(self.train_labels)//2]
            second_half = self.train_labels[len(self.train_labels)//2:]
            
            first_half_rate = sum(first_half) / len(first_half)
            second_half_rate = sum(second_half) / len(second_half)
            
            if second_half_rate > first_half_rate + 0.1:
                print(f"   📈 Trend: Learning improving (+{(second_half_rate - first_half_rate):.1%})")
            elif second_half_rate < first_half_rate - 0.1:
                print(f"   📉 Trend: Performance declining ({(second_half_rate - first_half_rate):.1%})")
            else:
                print(f"   ➡️ Trend: Stable performance")

    def get_learning_insights(self):
        """Get detailed learning insights and recommendations."""
        if not self.train_data or len(self.train_data) < 5:
            return "Insufficient data for insights"
        
        insights = []
        
        # Win streak analysis
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_type = None
        
        for result in self.train_labels:
            if result == 1:  # Win
                if current_type == "win":
                    current_streak += 1
                else:
                    if current_type == "loss":
                        max_loss_streak = max(max_loss_streak, current_streak)
                    current_streak = 1
                    current_type = "win"
            else:  # Loss
                if current_type == "loss":
                    current_streak += 1
                else:
                    if current_type == "win":
                        max_win_streak = max(max_win_streak, current_streak)
                    current_streak = 1
                    current_type = "loss"
        
        # Update final streaks
        if current_type == "win":
            max_win_streak = max(max_win_streak, current_streak)
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
        
        insights.append(f"Max win streak: {max_win_streak}")
        insights.append(f"Max loss streak: {max_loss_streak}")
        
        # Performance consistency
        if len(self.train_labels) >= 20:
            recent_20 = self.train_labels[-20:]
            win_rate_20 = sum(recent_20) / len(recent_20)
            if win_rate_20 > 0.6:
                insights.append("🔥 High performance mode!")
            elif win_rate_20 < 0.4:
                insights.append("⚠️ Low performance - consider strategy adjustment")
        
        return " | ".join(insights)

    def add_to_memory_buffer(self, features, result, metadata=None):
        """Add experience to memory buffer for incremental learning."""
        experience = {
            'features': features,
            'result': result,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        self.memory_buffer.append(experience)
        
        # Maintain buffer size
        if len(self.memory_buffer) > self.memory_buffer_size:
            self.memory_buffer.pop(0)  # Remove oldest experience

    def add_to_learning_memory(self, features, result, performance_metrics):
        """Add recent learning experience to memory."""
        learning_experience = {
            'features': features,
            'result': result,
            'performance': performance_metrics,
            'timestamp': time.time()
        }
        
        self.learning_memory.append(learning_experience)
        
        # Maintain memory size
        if len(self.learning_memory) > self.learning_memory_size:
            self.learning_memory.pop(0)

    def incremental_learning_update(self, features, result):
        """Perform incremental learning update using mini-batches."""
        if self.model is None or len(features) != self.feature_size:
            return
        
        try:
            # Add to memory buffers
            self.add_to_memory_buffer(features, result)
            
            # Prepare mini-batch for incremental learning
            if len(self.memory_buffer) >= self.incremental_batch_size:
                # Sample from memory buffer
                batch_indices = np.random.choice(
                    len(self.memory_buffer), 
                    size=min(self.incremental_batch_size, len(self.memory_buffer)),
                    replace=False
                )
                
                batch_features = []
                batch_labels = []
                
                for idx in batch_indices:
                    experience = self.memory_buffer[idx]
                    batch_features.append(experience['features'])
                    batch_labels.append(experience['result'])
                
                # Convert to numpy arrays
                x_batch = np.array(batch_features)
                y_batch = np.array(batch_labels)
                
                # Perform incremental learning
                original_lr = self.model.optimizer.learning_rate
                self.model.optimizer.learning_rate = 0.001  # Lower LR for stability
                
                # Single epoch incremental training
                history = self.model.fit(
                    x_batch, y_batch,
                    epochs=1,
                    verbose=0,
                    validation_split=0.0  # No validation for incremental
                )
                
                # Restore learning rate
                self.model.optimizer.learning_rate = original_lr
                
                # Track incremental learning
                if not hasattr(self, 'incremental_updates'):
                    self.incremental_updates = 0
                self.incremental_updates += 1
                
                # Add to learning memory
                performance_metrics = {
                    'accuracy': history.history['accuracy'][-1],
                    'loss': history.history['loss'][-1]
                }
                self.add_to_learning_memory(features, result, performance_metrics)
                
                if self.incremental_updates % 20 == 0:
                    print(f"[TensorFlow] Incremental learning: {self.incremental_updates} mini-batch updates")
                
                # Save model after each incremental update
                self.save_model("_incremental")
                
        except Exception as e:
            print(f"[TensorFlow] Incremental learning failed: {e}")

    def detect_catastrophic_forgetting(self):
        """Detect if the model is experiencing catastrophic forgetting."""
        if len(self.learning_memory) < 20:
            return False, "Insufficient data"
        
        # Compare recent performance with historical performance
        recent_experiences = self.learning_memory[-10:]
        historical_experiences = self.learning_memory[:-10]
        
        if len(historical_experiences) < 10:
            return False, "Insufficient historical data"
        
        recent_accuracy = np.mean([exp['performance']['accuracy'] for exp in recent_experiences])
        historical_accuracy = np.mean([exp['performance']['accuracy'] for exp in historical_experiences])
        
        performance_drop = historical_accuracy - recent_accuracy
        
        if performance_drop > self.catastrophic_forgetting_threshold:
            return True, f"Performance drop: {performance_drop:.3f}"
        
        return False, f"Performance stable: {performance_drop:.3f}"

    def experience_replay_training(self, replay_ratio=0.3):
        """Perform experience replay training to prevent forgetting."""
        if len(self.memory_buffer) < self.incremental_batch_size:
            return False
        
        try:
            # Sample experiences from different time periods
            total_experiences = len(self.memory_buffer)
            recent_size = int(total_experiences * replay_ratio)
            historical_size = total_experiences - recent_size
            
            # Recent experiences
            recent_indices = np.random.choice(
                range(total_experiences - recent_size, total_experiences),
                size=min(recent_size, self.incremental_batch_size // 2),
                replace=False
            )
            
            # Historical experiences
            historical_indices = np.random.choice(
                range(historical_size),
                size=min(historical_size, self.incremental_batch_size // 2),
                replace=False
            )
            
            # Combine experiences
            all_indices = np.concatenate([recent_indices, historical_indices])
            
            batch_features = []
            batch_labels = []
            
            for idx in all_indices:
                experience = self.memory_buffer[idx]
                batch_features.append(experience['features'])
                batch_labels.append(experience['result'])
            
            # Train on mixed experience
            x_batch = np.array(batch_features)
            y_batch = np.array(batch_labels)
            
            original_lr = self.model.optimizer.learning_rate
            self.model.optimizer.learning_rate = 0.0005  # Lower LR for stability
            
            history = self.model.fit(
                x_batch, y_batch,
                epochs=3,  # More epochs for experience replay
                verbose=0
            )
            
            self.model.optimizer.learning_rate = original_lr
            
            print(f"[TensorFlow] Experience replay: {len(all_indices)} experiences, accuracy: {history.history['accuracy'][-1]:.3f}")
            
            # Save model after experience replay
            self.save_model("_replay")
            return True
            
        except Exception as e:
            print(f"[TensorFlow] Experience replay failed: {e}")
            return False

    def adaptive_incremental_learning(self, features, result):
        """Adaptive incremental learning that adjusts based on performance."""
        if self.model is None or len(features) != self.feature_size:
            return
        
        # Perform incremental learning
        self.incremental_learning_update(features, result)
        
        # Check for catastrophic forgetting
        is_forgetting, message = self.detect_catastrophic_forgetting()
        
        if is_forgetting:
            print(f"[TensorFlow] Catastrophic forgetting detected: {message}")
            print("[TensorFlow] Performing experience replay training...")
            
            # Perform experience replay to recover
            success = self.experience_replay_training(replay_ratio=0.4)
            
            if success:
                print("[TensorFlow] Experience replay completed successfully")
            else:
                print("[TensorFlow] Experience replay failed, performing full retraining...")
                # Fallback to full retraining if experience replay fails
                if len(self.train_data) >= 10:
                    x = np.array(self.train_data)
                    y = np.array(self.train_labels)
                    self.model.fit(x, y, epochs=5, verbose=0)
                    print("[TensorFlow] Full retraining completed")
                    
                    # Save model after full retraining
                    self.save_model("_full_retrain")

    def get_incremental_learning_stats(self):
        """Get statistics about incremental learning performance."""
        stats = {
            'memory_buffer_size': len(self.memory_buffer),
            'learning_memory_size': len(self.learning_memory),
            'incremental_updates': getattr(self, 'incremental_updates', 0),
            'memory_utilization': len(self.memory_buffer) / self.memory_buffer_size * 100
        }
        
        if len(self.learning_memory) >= 10:
            recent_accuracies = [exp['performance']['accuracy'] for exp in self.learning_memory[-10:]]
            stats['recent_accuracy'] = np.mean(recent_accuracies)
            stats['accuracy_std'] = np.std(recent_accuracies)
        
        return stats

    def setup_rule_based_learning(self):
        """Setup rule-based learning system that learns during trades."""
        print('[Learning] Setting up rule-based real-time learning system')
        
        # Learning patterns and rules
        self.learning_patterns = {
            'candle_patterns': {},  # Track candle patterns and their success rates
            'indicator_patterns': {},  # Track indicator combinations
            'time_patterns': {},  # Track time-based patterns
            'streak_patterns': {},  # Track win/loss streaks
            'market_conditions': {}  # Track market conditions
        }
        
        # Real-time learning parameters
        self.learning_confidence = 0.5  # Start with neutral confidence
        self.learning_rate = 0.1  # How fast to adapt
        self.min_samples = 3  # Minimum samples before trusting a pattern
        
        # Current strategy state
        self.current_strategy = 'neutral'
        self.strategy_confidence = 0.5
        self.adaptation_speed = 0.1
        
        print('[Learning] Rule-based learning system ready')

    def learn_from_trade(self, features, result, market_data):
        """Learn from each trade in real-time."""
        if not hasattr(self, 'learning_patterns'):
            self.setup_rule_based_learning()
        
        # Extract patterns from the trade
        patterns = self.extract_patterns(features, market_data)
        
        # Update pattern success rates
        for pattern_type, pattern_data in patterns.items():
            if pattern_type not in self.learning_patterns:
                self.learning_patterns[pattern_type] = {}
            
            pattern_key = str(pattern_data)
            if pattern_key not in self.learning_patterns[pattern_type]:
                self.learning_patterns[pattern_type][pattern_key] = {
                    'wins': 0,
                    'total': 0,
                    'success_rate': 0.0,
                    'confidence': 0.0
                }
            
            # Update statistics
            self.learning_patterns[pattern_type][pattern_key]['total'] += 1
            if result == 1:  # Win
                self.learning_patterns[pattern_type][pattern_key]['wins'] += 1
            
            # Calculate success rate
            stats = self.learning_patterns[pattern_type][pattern_key]
            stats['success_rate'] = stats['wins'] / stats['total']
            
            # Calculate confidence based on sample size
            stats['confidence'] = min(1.0, stats['total'] / self.min_samples)
        
        # Update overall learning confidence
        self.update_learning_confidence()
        
        # Adapt strategy based on recent performance
        self.adapt_strategy()
        
        print(f'[Learning] Pattern learned: {len(patterns)} patterns updated')
        
        # Save rule-based learning patterns
        self.save_learning_patterns()

    def extract_patterns(self, features, market_data):
        """Extract learning patterns from trade data."""
        patterns = {}
        
        # Candle patterns
        if len(features) >= 25:  # 5 candles * 5 features
            candle_data = features[:25]
            patterns['candle_patterns'] = self.analyze_candle_patterns(candle_data)
        
        # Indicator patterns
        if len(features) >= 35:
            indicator_data = features[25:35]  # Technical indicators
            patterns['indicator_patterns'] = self.analyze_indicator_patterns(indicator_data)
        
        # Time patterns
        current_hour = time.localtime().tm_hour
        patterns['time_patterns'] = {
            'hour': current_hour,
            'time_of_day': 'morning' if 6 <= current_hour < 12 else 
                          'afternoon' if 12 <= current_hour < 18 else 
                          'evening' if 18 <= current_hour < 24 else 'night'
        }
        
        # Market conditions
        if market_data:
            patterns['market_conditions'] = self.analyze_market_conditions(market_data)
        
        return patterns

    def analyze_candle_patterns(self, candle_data):
        """Analyze candle patterns for learning."""
        patterns = {}
        
        # Reshape to 5 candles with 5 features each
        candles = np.array(candle_data).reshape(5, 5)
        
        # Pattern 1: Trend direction
        opens = candles[:, 0]
        closes = candles[:, 1]
        highs = candles[:, 2]
        lows = candles[:, 3]
        
        green_candles = np.sum(closes > opens)
        red_candles = np.sum(closes < opens)
        
        patterns['trend'] = 'bullish' if green_candles > red_candles else 'bearish' if red_candles > green_candles else 'neutral'
        patterns['green_ratio'] = green_candles / 5.0
        patterns['red_ratio'] = red_candles / 5.0
        
        # Pattern 2: Volatility
        price_ranges = highs - lows
        avg_range = np.mean(price_ranges)
        patterns['volatility'] = 'high' if avg_range > np.std(price_ranges) * 1.5 else 'low'
        
        # Pattern 3: Momentum
        price_changes = closes - opens
        momentum = np.sum(price_changes)
        patterns['momentum'] = 'positive' if momentum > 0 else 'negative'
        
        return patterns

    def analyze_indicator_patterns(self, indicator_data):
        """Analyze technical indicator patterns."""
        patterns = {}
        
        if len(indicator_data) >= 10:
            # RSI, MACD, Stochastic patterns
            rsi = indicator_data[2] if len(indicator_data) > 2 else 50
            macd = indicator_data[3] if len(indicator_data) > 3 else 0
            stoch_k = indicator_data[7] if len(indicator_data) > 7 else 50
            stoch_d = indicator_data[8] if len(indicator_data) > 8 else 50
            
            patterns['rsi_signal'] = 'overbought' if rsi > 70 else 'oversold' if rsi < 30 else 'neutral'
            patterns['macd_signal'] = 'bullish' if macd > 0 else 'bearish'
            patterns['stoch_signal'] = 'overbought' if stoch_k > 80 else 'oversold' if stoch_k < 20 else 'neutral'
            
            # Combined signal strength
            bullish_signals = sum([
                patterns['rsi_signal'] == 'oversold',
                patterns['macd_signal'] == 'bullish',
                patterns['stoch_signal'] == 'oversold'
            ])
            
            bearish_signals = sum([
                patterns['rsi_signal'] == 'overbought',
                patterns['macd_signal'] == 'bearish',
                patterns['stoch_signal'] == 'overbought'
            ])
            
            patterns['signal_strength'] = 'strong_bullish' if bullish_signals >= 2 else \
                                        'strong_bearish' if bearish_signals >= 2 else \
                                        'weak_bullish' if bullish_signals > bearish_signals else \
                                        'weak_bearish' if bearish_signals > bullish_signals else 'neutral'
        
        return patterns

    def analyze_market_conditions(self, market_data):
        """Analyze current market conditions."""
        conditions = {}
        
        # This would analyze real market data
        # For now, we'll use basic patterns
        conditions['trend'] = 'unknown'
        conditions['volatility'] = 'unknown'
        conditions['volume'] = 'unknown'
        
        return conditions

    def update_learning_confidence(self):
        """Update overall learning confidence based on pattern performance."""
        if not hasattr(self, 'learning_patterns'):
            return
        
        total_patterns = 0
        confident_patterns = 0
        
        for pattern_type, patterns in self.learning_patterns.items():
            for pattern_key, stats in patterns.items():
                total_patterns += 1
                if stats['confidence'] > 0.7:  # High confidence threshold
                    confident_patterns += 1
        
        if total_patterns > 0:
            self.learning_confidence = confident_patterns / total_patterns
        else:
            self.learning_confidence = 0.5

    def adapt_strategy(self):
        """Adapt trading strategy based on learned patterns."""
        if not hasattr(self, 'learning_patterns'):
            return
        
        # Analyze recent performance
        recent_performance = self.get_recent_performance()
        
        # Adapt strategy based on performance
        if recent_performance > 0.7:  # High performance
            self.current_strategy = 'aggressive'
            self.strategy_confidence = min(1.0, self.strategy_confidence + self.adaptation_speed)
        elif recent_performance < 0.3:  # Low performance
            self.current_strategy = 'conservative'
            self.strategy_confidence = max(0.1, self.strategy_confidence - self.adaptation_speed)
        else:  # Medium performance
            self.current_strategy = 'balanced'
            # Keep current confidence
        
        print(f'[Learning] Strategy adapted: {self.current_strategy} (confidence: {self.strategy_confidence:.2f})')

    def get_recent_performance(self):
        """Get recent trading performance."""
        if not hasattr(self, 'train_labels') or len(self.train_labels) < 5:
            return 0.5  # Neutral if no data
        
        recent_trades = self.train_labels[-10:]  # Last 10 trades
        return sum(recent_trades) / len(recent_trades)

    def predict_direction_learned(self, features, market_data):
        """Predict direction using learned patterns."""
        if not hasattr(self, 'learning_patterns'):
            return 'call'  # Default if no learning
        
        # Extract current patterns
        patterns = self.extract_patterns(features, market_data)
        
        # Calculate prediction confidence for each pattern type
        predictions = []
        confidences = []
        
        for pattern_type, pattern_data in patterns.items():
            if pattern_type in self.learning_patterns:
                pattern_key = str(pattern_data)
                if pattern_key in self.learning_patterns[pattern_type]:
                    stats = self.learning_patterns[pattern_type][pattern_key]
                    
                    if stats['confidence'] > 0.5:  # Only use confident patterns
                        if stats['success_rate'] > 0.6:
                            predictions.append('call')
                            confidences.append(stats['success_rate'] * stats['confidence'])
                        elif stats['success_rate'] < 0.4:
                            predictions.append('put')
                            confidences.append((1 - stats['success_rate']) * stats['confidence'])
        
        # Make final prediction
        if predictions and confidences:
            # Weighted voting based on confidence
            call_weight = sum(conf for pred, conf in zip(predictions, confidences) if pred == 'call')
            put_weight = sum(conf for pred, conf in zip(predictions, confidences) if pred == 'put')
            
            if call_weight > put_weight:
                return 'call'
            elif put_weight > call_weight:
                return 'put'
        
        # Fallback to basic analysis
        return self.basic_direction_analysis(features)

    def basic_direction_analysis(self, features):
        """Basic direction analysis when learning is insufficient."""
        if len(features) >= 25:
            # Use last 5 candles
            candles = np.array(features[:25]).reshape(5, 5)
            opens = candles[:, 0]
            closes = candles[:, 1]
            
            green_count = np.sum(closes > opens)
            red_count = np.sum(closes < opens)
            
            if green_count > red_count:
                return 'call'
            elif red_count > green_count:
                return 'put'
        
        return 'call'  # Default fallback

    def predict_with_confidence(self, features):
        """Make prediction with confidence score using neural network."""
        if self.model is None or len(features) != self.feature_size:
            return 'call', 0.5  # Default prediction with neutral confidence
        
        try:
            # Get prediction probability
            prediction_proba = self.model.predict(np.array([features]), verbose=0)[0][0]
            
            # Determine direction and confidence
            direction = 'call' if prediction_proba > 0.5 else 'put'
            confidence = abs(prediction_proba - 0.5) * 2  # Convert to 0-1 confidence scale
            
            return direction, confidence
            
        except Exception as e:
            print(f"[Neural Network] Prediction failed: {e}")
            return 'call', 0.5

    def get_neural_network_insights(self, features):
        """Get detailed insights from neural network layers."""
        if self.model is None or len(features) != self.feature_size:
            return {}
        
        try:
            # Create intermediate model to get layer outputs
            layer_outputs = []
            for layer in self.model.layers[:-1]:  # Exclude output layer
                layer_outputs.append(layer.output)
            
            # Create model to get intermediate outputs
            intermediate_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer_outputs
            )
            
            # Get layer activations
            activations = intermediate_model.predict(np.array([features]), verbose=0)
            
            insights = {
                'input_normalized': activations[0][0].tolist() if len(activations) > 0 else [],
                'market_patterns': activations[1][0].tolist() if len(activations) > 1 else [],
                'technical_analysis': activations[2][0].tolist() if len(activations) > 2 else [],
                'pattern_recognition': activations[3][0].tolist() if len(activations) > 3 else [],
                'decision_making': activations[4][0].tolist() if len(activations) > 4 else []
            }
            
            return insights
            
        except Exception as e:
            print(f"[Neural Network] Insights failed: {e}")
            return {}

    def analyze_neural_network_performance(self):
        """Analyze neural network performance and provide insights."""
        if not self.train_data or len(self.train_data) < 10:
            return "Insufficient data for analysis"
        
        try:
            # Get recent predictions vs actual results
            recent_data = self.train_data[-20:] if len(self.train_data) >= 20 else self.train_data
            recent_labels = self.train_labels[-20:] if len(self.train_labels) >= 20 else self.train_labels
            
            predictions = []
            confidences = []
            
            for features in recent_data:
                direction, confidence = self.predict_with_confidence(features)
                predictions.append(1 if direction == 'call' else 0)
                confidences.append(confidence)
            
            # Calculate metrics
            correct_predictions = sum(1 for p, l in zip(predictions, recent_labels) if p == l)
            accuracy = correct_predictions / len(predictions)
            avg_confidence = sum(confidences) / len(confidences)
            
            # Analyze confidence vs accuracy
            high_conf_predictions = [(p, l, c) for p, l, c in zip(predictions, recent_labels, confidences) if c > 0.7]
            if high_conf_predictions:
                high_conf_accuracy = sum(1 for p, l, c in high_conf_predictions if p == l) / len(high_conf_predictions)
            else:
                high_conf_accuracy = 0
            
            insights = {
                'recent_accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'high_confidence_accuracy': high_conf_accuracy,
                'total_predictions': len(predictions),
                'high_confidence_predictions': len(high_conf_predictions)
            }
            
            return insights
            
        except Exception as e:
            print(f"[Neural Network] Performance analysis failed: {e}")
            return {}

    def update_streaks(self, result):
        """Update win/loss streaks based on trade result."""
        if result == "WIN":
            self.win_streak += 1
            self.loss_streak = 0
        else:  # LOSS
            self.loss_streak += 1
            self.win_streak = 0

    def log_trade_to_monitor(self, predicted_direction, actual_result, profit):
        """Log trade to monitoring system."""
        # Determine if prediction was correct
        correct_prediction = 1 if (predicted_direction.upper() == "CALL" and actual_result == "WIN") or \
                                  (predicted_direction.upper() == "PUT" and actual_result == "WIN") else 0
        
        # Update streaks
        self.update_streaks(actual_result)
        
        # Get current strategy and learning confidence
        strategy = getattr(self, 'current_strategy', 'neutral')
        learning_confidence = getattr(self, 'learning_confidence', 0.5)
        
        # Log to monitor
        self.monitor.log_trade(
            correct_prediction=correct_prediction,
            profit=profit,
            trade_count=self.trade_count,
            win_streak=self.win_streak,
            loss_streak=self.loss_streak,
            strategy=strategy,
            learning_confidence=learning_confidence
        )

    def save_learning_patterns(self):
        """Save rule-based learning patterns to file."""
        if not hasattr(self, 'learning_patterns'):
            return
        
        try:
            import json
            import os
            
            # Create patterns directory if it doesn't exist
            os.makedirs('patterns', exist_ok=True)
            
            # Prepare patterns data for saving
            patterns_data = {
                'learning_patterns': self.learning_patterns,
                'learning_confidence': getattr(self, 'learning_confidence', 0.5),
                'current_strategy': getattr(self, 'current_strategy', 'neutral'),
                'strategy_confidence': getattr(self, 'strategy_confidence', 0.5),
                'adaptation_speed': getattr(self, 'adaptation_speed', 0.1),
                'min_samples': getattr(self, 'min_samples', 3),
                'timestamp': time.time()
            }
            
            # Save patterns
            patterns_file = os.path.join('patterns', 'learning_patterns.json')
            with open(patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            print(f'[Learning] Patterns saved to {patterns_file}')
            
        except Exception as e:
            print(f'[Learning] Failed to save patterns: {e}')

    def load_learning_patterns(self):
        """Load rule-based learning patterns from file."""
        try:
            import json
            import os
            
            patterns_file = os.path.join('patterns', 'learning_patterns.json')
            if os.path.exists(patterns_file):
                with open(patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                
                self.learning_patterns = patterns_data.get('learning_patterns', {})
                self.learning_confidence = patterns_data.get('learning_confidence', 0.5)
                self.current_strategy = patterns_data.get('current_strategy', 'neutral')
                self.strategy_confidence = patterns_data.get('strategy_confidence', 0.5)
                self.adaptation_speed = patterns_data.get('adaptation_speed', 0.1)
                self.min_samples = patterns_data.get('min_samples', 3)
                
                print(f'[Learning] Patterns loaded from {patterns_file}')
                print(f'[Learning] Loaded {sum(len(patterns) for patterns in self.learning_patterns.values())} patterns')
                return True
            else:
                print(f'[Learning] No patterns file found, starting fresh')
                return False
                
        except Exception as e:
            print(f'[Learning] Failed to load patterns: {e}')
            return False
    def setup_model(self):
        if tf is None:
            print("TensorFlow not installed. Self-learning is disabled.")
            self.model = None
            return
        
        print("[Neural Network] Setting up advanced neural network architecture...")
        
        # --- Feature engineering ---
        # Features: [open, close, high, low, volume] for last 5 candles, SMA, EMA, RSI, MACD, Stochastic, previous trade result, profit/loss, time of day
        self.feature_size = 5 * 5 + 3 + 3 + 3 + 2  # 5 candles * 5 features + SMA + EMA + RSI + MACD + Stoch(K,D) + prev_result + prev_profit + time_of_day
        
        # Advanced Neural Network Architecture
        self.model = tf.keras.Sequential([
            # Input layer
            tf.keras.layers.Input(shape=(self.feature_size,), name='market_data'),
            
            # Feature normalization
            tf.keras.layers.BatchNormalization(name='input_normalization'),
            
            # First hidden layer - Market pattern recognition
            tf.keras.layers.Dense(128, activation='relu', name='market_patterns'),
            tf.keras.layers.Dropout(0.3, name='dropout_1'),
            tf.keras.layers.BatchNormalization(name='bn_1'),
            
            # Second hidden layer - Technical analysis
            tf.keras.layers.Dense(64, activation='relu', name='technical_analysis'),
            tf.keras.layers.Dropout(0.2, name='dropout_2'),
            tf.keras.layers.BatchNormalization(name='bn_2'),
            
            # Third hidden layer - Pattern recognition
            tf.keras.layers.Dense(32, activation='relu', name='pattern_recognition'),
            tf.keras.layers.Dropout(0.2, name='dropout_3'),
            tf.keras.layers.BatchNormalization(name='bn_3'),
            
            # Fourth hidden layer - Decision making
            tf.keras.layers.Dense(16, activation='relu', name='decision_making'),
            tf.keras.layers.Dropout(0.1, name='dropout_4'),
            
            # Output layer - Binary classification (call/put)
            tf.keras.layers.Dense(1, activation='sigmoid', name='prediction')
        ])
        
        # Advanced optimizer with learning rate scheduling
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with advanced metrics
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
        # Initialize training data
        self.train_data = []
        self.train_labels = []
        self.prev_result = 0  # 1 for win, 0 for loss
        self.prev_profit = 0.0
        
        # Neural network training parameters
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        self.model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        print(f"[Neural Network] Model created with {self.model.count_params():,} parameters")
        print(f"[Neural Network] Feature size: {self.feature_size}")
        print(f"[Neural Network] Architecture: Input -> 128 -> 64 -> 32 -> 16 -> 1")
        print(f"[Neural Network] Optimizer: Adam with exponential decay")
        print(f"[Neural Network] Regularization: Dropout + BatchNorm")

    async def auto_trade(self, amount: float = 50, asset: str = "EURUSD_otc", direction: str = "call", duration: int = 60, interval: int = 60, use_model: bool = True) -> None:
        """Automatically executes trades every interval (seconds), predicts direction using last candle, and shows profit/loss clearly."""
        logger.info(f"Starting auto-trade: {amount} {asset} {duration}s every {interval}s, predicting direction.")
        # Connect once
        if not self.client:
            logger.error("Quotex API client not initialized.")
            raise RuntimeError("Quotex API client not initialized.")
        if not await self.client.check_connect():
            logger.info("Establishing connection...")
            check, reason = await self._connect_with_retry()
            if not check:
                logger.error(f"Failed to connect: {reason}")
                raise ConnectionError(f"Failed to connect: {reason}")
        self.load_model()
        try:
            total_profit = 0.0
            trade_count = 0
            base_amount = amount
            next_amount = amount
            while True:
                # --- Feature engineering ---
                end_from_time = time.time()
                candles_needed = 30
                candles = await self.client.get_candles(asset, end_from_time, candles_needed * 60, 60)
                direction_pred = direction
                green_count = 0
                red_count = 0
                features = []
                # Use last 5 candles, each with open, close, high, low, volume
                if candles and len(candles) >= 5:
                    last_candles = candles[-5:]
                    print("[DEBUG] Last 5 candles:")
                    for idx, candle in enumerate(last_candles):
                        print(f"  Candle {idx+1}: {candle}")
                    for candle in last_candles:
                        features.extend([
                            candle.get('open', 0),
                            candle.get('close', 0),
                            candle.get('high', 0),
                            candle.get('low', 0),
                            candle.get('volume', 0)
                        ])
                        if 'open' in candle and 'close' in candle:
                            if candle['close'] > candle['open']:
                                green_count += 1
                            elif candle['close'] < candle['open']:
                                red_count += 1
                    # Technical indicators
                    # Use last 30 candles for all indicator calculations
                    closes_30 = [c.get('close', 0) for c in candles[-30:]] if len(candles) >= 30 else []
                    highs_30 = [c.get('high', 0) for c in candles[-30:]] if len(candles) >= 30 else []
                    lows_30 = [c.get('low', 0) for c in candles[-30:]] if len(candles) >= 30 else []
                    closes = closes_30
                    highs = highs_30
                    lows = lows_30
                    if len(set(closes)) <= 1:
                        print("[WARNING] Candle close prices are flat or insufficient for indicator calculation.")
                    if len(closes) < 30:
                        print(f"[WARNING] Only {len(closes)} closes available, indicators may not be accurate (needs >=30).")
                    sma_val = TechnicalIndicators.calculate_sma(closes, period=5)
                    ema_val = TechnicalIndicators.calculate_ema(closes, period=5)
                    rsi_val = TechnicalIndicators.calculate_rsi(closes, period=14)
                    # MACD needs >=30 closes
                    if len(closes) >= 30:
                        macd_vals = TechnicalIndicators.calculate_macd(closes)
                        macd_list = macd_vals.get('macd', [])
                        signal_list = macd_vals.get('signal', [])
                        hist_list = macd_vals.get('histogram', [])
                        last_macd = macd_list[-1] if macd_list else None
                        last_signal = signal_list[-1] if signal_list else None
                        last_hist = hist_list[-1] if hist_list else None
                    else:
                        print(f"[WARNING] Only {len(closes)} closes available, MACD not calculated (needs >=30).")
                        last_macd = last_signal = last_hist = None
                    # Stochastic needs >=14 closes
                    if len(closes) >= 14:
                        stoch_vals = TechnicalIndicators.calculate_stochastic(closes, highs, lows, k_period=14, d_period=3)
                        last_k = stoch_vals['k'][-1] if stoch_vals['k'] else None
                        last_d = stoch_vals['d'][-1] if stoch_vals['d'] else None
                    else:
                        print(f"[WARNING] Only {len(closes)} closes available, Stochastic not calculated.")
                        last_k = last_d = None
                    last_rsi = rsi_val[-1] if rsi_val else None
                    features.append(sma_val[-1] if sma_val else 0)
                    features.append(ema_val[-1] if ema_val else 0)
                    features.append(last_rsi if last_rsi is not None else 0)
                    features.append(last_macd if last_macd is not None else 0)
                    features.append(last_signal if last_signal is not None else 0)
                    features.append(last_hist if last_hist is not None else 0)
                    features.append(last_k if last_k is not None else 0)
                    features.append(last_d if last_d is not None else 0)
                    # RSI status for overbought/oversold
                    rsi_status = "Neutral"
                    if last_rsi is not None:
                        if last_rsi >= 70:
                            rsi_status = "Overbought"
                        elif last_rsi <= 30:
                            rsi_status = "Oversold"
                    # MACD status for trend/momentum
                    macd_status = "Neutral"
                    if last_macd is not None and last_signal is not None:
                        if last_macd > last_signal:
                            macd_status = "Bullish"
                        elif last_macd < last_signal:
                            macd_status = "Bearish"
                    # Stochastic status
                    stoch_status = "Neutral"
                    if last_k is not None:
                        if last_k >= 80:
                            stoch_status = "Overbought"
                        elif last_k <= 20:
                            stoch_status = "Oversold"
                    # Add previous trade result, previous profit, time of day
                    features.append(self.prev_result)
                    features.append(self.prev_profit)
                    features.append(time.localtime().tm_hour / 24.0)  # normalized hour
                # Use advanced neural network for prediction
                if use_model and len(features) == self.feature_size and self.model is not None:
                    # Neural network prediction with confidence
                    direction_pred, nn_confidence = self.predict_with_confidence(features)
                    print(f"[Neural Network] Prediction: {direction_pred.upper()} (confidence: {nn_confidence:.2%})")
                elif use_model and len(features) == self.feature_size:
                    # Fallback to learned pattern prediction
                    direction_pred = self.predict_direction_learned(features, candles)
                    else:
                    # Fallback to basic analysis
                        if green_count > red_count:
                            direction_pred = "call"
                        elif red_count > green_count:
                            direction_pred = "put"
                        else:
                            direction_pred = direction
                # Use buy_and_check_win for accurate result and profit
                balance_before = await self.client.get_balance()
                await self.buy_and_check_win.__wrapped__(self, next_amount, asset, direction_pred, duration)
                balance_after = await self.client.get_balance()
                profit = balance_after - balance_before
                total_profit += profit
                trade_count += 1
                result = "WIN" if profit > 0 else ("LOSS" if profit < 0 else "EVEN")
                
                # Update trade count and log to monitor
                self.trade_count = trade_count
                self.log_trade_to_monitor(direction_pred, result, profit)
                
                # Real-time recovery: increase amount after loss, reset after win
                if result == "LOSS":
                    next_amount *= 2  # Martingale: double after loss
                else:
                    next_amount = base_amount  # Reset after win or even
                # REAL-TIME LEARNING: Learn from every trade immediately
                if candles and len(features) == self.feature_size:
                    self.train_data.append(features)
                    self.train_labels.append(1 if result == "WIN" else 0)
                    
                    # Learn from this trade using pattern recognition
                    self.learn_from_trade(features, 1 if result == "WIN" else 0, candles)
                    
                    # Also do TensorFlow learning if available
                    if self.model is not None:
                        self.immediate_learning_update(features, 1 if result == "WIN" else 0)
                        self.adaptive_incremental_learning(features, 1 if result == "WIN" else 0)
                    
                    # Enhanced retraining logic with more aggressive triggers
                    should_retrain = False
                    retrain_reason = ""
                    retrain_urgency = "normal"
                    
                    # URGENT: Retrain after 3 consecutive losses
                    if len(self.train_labels) >= 3:
                        last_3 = self.train_labels[-3:]
                        if all(x == 0 for x in last_3):  # All losses
                            should_retrain = True
                            retrain_reason = "URGENT: 3 consecutive losses detected!"
                            retrain_urgency = "urgent"
                    
                    # URGENT: Retrain after 5 consecutive wins (might be overfitting)
                    elif len(self.train_labels) >= 5:
                        last_5 = self.train_labels[-5:]
                        if all(x == 1 for x in last_5):  # All wins
                            should_retrain = True
                            retrain_reason = "URGENT: 5 consecutive wins - possible overfitting!"
                            retrain_urgency = "urgent"
                    
                    # HIGH: Retrain every 5 trades (more frequent)
                    elif len(self.train_data) % 5 == 0:
                        should_retrain = True
                        retrain_reason = "HIGH: Scheduled retraining (every 5 trades)"
                        retrain_urgency = "high"
                    
                    # MEDIUM: Retrain if recent performance drops
                    elif len(self.train_data) >= 10:
                        recent_trades = self.train_labels[-5:]  # Last 5 trades
                        recent_win_rate = sum(recent_trades) / len(recent_trades)
                        overall_win_rate = sum(self.train_labels) / len(self.train_labels)
                        
                        if recent_win_rate < overall_win_rate * 0.6:  # 40% drop in performance
                            should_retrain = True
                            retrain_reason = f"MEDIUM: Performance drop (recent: {recent_win_rate:.1%}, overall: {overall_win_rate:.1%})"
                            retrain_urgency = "medium"
                    
                    # LOW: Regular retraining with sufficient data
                    elif len(self.train_data) >= 20 and len(self.train_data) % 3 == 0:
                        should_retrain = True
                        retrain_reason = "LOW: Regular retraining with sufficient data"
                        retrain_urgency = "low"
                    
                    if should_retrain:
                        urgency_emoji = {"urgent": "🚨", "high": "⚡", "medium": "🔄", "low": "📊"}
                        print(f"\n{urgency_emoji[retrain_urgency]} [TensorFlow] {retrain_urgency.upper()} Retraining: {retrain_reason}")
                        print(f"[TensorFlow] Training data: {len(self.train_data)} samples")
                        
                        x = np.array(self.train_data)
                        y = np.array(self.train_labels)
                        
                        # Adaptive training based on urgency and data size
                        if retrain_urgency == "urgent":
                            epochs = min(50, max(10, len(self.train_data) // 2))  # More aggressive
                            learning_rate = 0.01  # Higher learning rate
                        elif retrain_urgency == "high":
                            epochs = min(30, max(8, len(self.train_data) // 3))
                            learning_rate = 0.005
                        else:
                            epochs = min(20, max(5, len(self.train_data) // 5))
                            learning_rate = 0.001
                        
                        # Update learning rate
                        self.model.optimizer.learning_rate = learning_rate
                        
                        # Enhanced training with validation split
                        validation_split = 0.2 if len(self.train_data) >= 15 else 0.0
                        
                        # Advanced training with callbacks
                        callbacks = []
                        if validation_split > 0:
                            callbacks.append(self.early_stopping)
                            callbacks.append(self.model_checkpoint)
                        
                        history = self.model.fit(
                            x, y, 
                            epochs=epochs, 
                            validation_split=validation_split,
                            callbacks=callbacks,
                            verbose=0,
                            batch_size=min(32, len(x))
                        )
                        
                        # Get training metrics
                        final_acc = history.history['accuracy'][-1]
                        final_loss = history.history['loss'][-1]
                        
                        metrics_text = f"Accuracy: {final_acc:.3f}, Loss: {final_loss:.3f}, LR: {learning_rate:.4f}"
                        
                        if validation_split > 0:
                            val_acc = history.history.get('val_accuracy', [0])[-1]
                            val_loss = history.history.get('val_loss', [0])[-1]
                            metrics_text += f", Val-Acc: {val_acc:.3f}, Val-Loss: {val_loss:.3f}"
                        
                        print(f"[TensorFlow] Retraining completed. {metrics_text}")
                        
                        # Save the improved model with urgency indicator
                        self.save_model(f"_retrained_{retrain_urgency}")
                        
                        # Show performance summary
                        performance = self.get_model_performance()
                        print(f"[TensorFlow] Model Performance: {performance}")
                        
                        # Show learning progress
                        self.show_learning_progress()
                        
                        # Additional save after batch retraining
                        print(f"[TensorFlow] Model saved after {retrain_urgency} retraining")
                # Update previous trade info for next feature vector
                self.prev_result = 1 if result == "WIN" else 0
                self.prev_profit = profit
                print("\n====================================")
                print(f" Trade #{trade_count}")
                print(f" Predicted   : {direction_pred.upper()} (last 5: {green_count} green, {red_count} red)")
                print(f" Strategy    : {getattr(self, 'current_strategy', 'neutral').upper()} (confidence: {getattr(self, 'strategy_confidence', 0.5):.2f})")
                print(f" Learning    : {getattr(self, 'learning_confidence', 0.5):.2f} confidence, {len(getattr(self, 'learning_patterns', {}).get('candle_patterns', {}))} patterns")
                print(f" RSI         : {last_rsi if last_rsi is not None else 'N/A'} ({rsi_status})")
                print(f" MACD        : {last_macd if last_macd is not None else 'N/A'} | Signal: {last_signal if last_signal is not None else 'N/A'} | Hist: {last_hist if last_hist is not None else 'N/A'} ({macd_status})")
                print(f" Stochastic  : K={last_k if last_k is not None else 'N/A'} D={last_d if last_d is not None else 'N/A'} ({stoch_status})")
                print(f" Amount      : R$ {next_amount:.2f}")
                print(f" Asset       : {asset}")
                print(f" Result      : {result}")
                print(f" Profit/Loss : R$ {profit:.2f}")
                print(f" Total Profit: R$ {total_profit:.2f}")
                print(f" Next Trade Amount: R$ {next_amount:.2f}")
                if self.model is not None:
                    print(f"[TensorFlow] Model used for prediction: {use_model}")
                print("====================================\n")
                logger.info(f"Waiting {interval} seconds before next trade...")
                print(f"⏳ Waiting {interval} seconds before next trade...")
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Auto-trading stopped by user.")
            print("\n✅ Auto-trading stopped by user.")
        finally:
            if self.client and await self.client.check_connect():
                await self.client.close()
                logger.debug("Connection closed after auto-trade loop.")
    """PyQuotex CLI application for trading operations."""

    def __init__(self):
        self.client: Optional[Quotex] = None
        self.model = None
        self.feature_size = 0
        self.train_data = []
        self.train_labels = []
        self.prev_result = 0  # 1 for win, 0 for loss
        self.prev_profit = 0.0
        
        # Incremental learning parameters
        self.memory_buffer = []  # Experience replay buffer
        self.memory_buffer_size = 1000  # Maximum buffer size
        self.incremental_batch_size = 32  # Mini-batch size for incremental learning
        self.learning_memory = []  # Recent learning experiences
        self.learning_memory_size = 100  # Keep recent experiences
        self.catastrophic_forgetting_threshold = 0.15  # Threshold for forgetting detection
        
        # Monitoring system
        self.monitor = Monitor(rolling_window=50, csv_file="trade_metrics.csv")
        self.trade_count = 0
        self.win_streak = 0
        self.loss_streak = 0
        
        self.setup_client()

    def setup_client(self):
        """Initializes the Quotex API client with credentials."""
        try:
            email, password = credentials()
            self.client = Quotex(
                email=email,
                password=password,
                lang="pt"
            )
            logger.info("Quotex client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Quotex client: {e}")
            raise

    async def _connect_with_retry(self, attempts: int = 5) -> Tuple[bool, str]:
        """Internal method to attempt connection with retry logic."""
        logger.info("Attempting to connect to Quotex API...")
        check, reason = await self.client.connect()

        if not check:
            for attempt_num in range(1, attempts + 1):
                logger.warning(f"Connection failed. Attempt {attempt_num} of {attempts}.")

                session_file = Path("session.json")
                if session_file.exists():
                    session_file.unlink()
                    logger.debug("Obsolete session file removed.")

                await asyncio.sleep(2)
                check, reason = await self.client.connect()

                if check:
                    logger.info("Reconnected successfully!")
                    break

            if not check:
                logger.error(f"Failed to connect after {attempts} attempts: {reason}")
                return False, reason

        logger.info(f"Connected successfully: {reason}")
        return check, reason

    def display_banner(self):
        """Displays the application banner, including the private version ad."""
        custom_font = pyfiglet.Figlet(font="ansi_shadow")
        ascii_art = custom_font.renderText("JundAlNabi")

        user_lang = detect_user_language()
        ad_message = LANGUAGE_MESSAGES.get(user_lang, LANGUAGE_MESSAGES["en_US"])["private_version_ad"]

        banner = f"""{ascii_art}
        Author: {__author__} | Version: {__version__}
        Use with moderation, because management is everything!
        Support: alet8319@gmail.com or +92-3478042183

        {ad_message}

        """
        print(banner)

    @ensure_connection()
    async def test_connection(self) -> None:
        """Tests the connection to the Quotex API."""
        logger.info("Running connection test.")
        is_connected = await self.client.check_connect()

        if is_connected:
            logger.info("Connection test successful.")
            print("✅ Connection successful!")
        else:
            logger.error("Connection test failed.")
            print("❌ Connection failed!")

    @ensure_connection()
    async def get_balance(self) -> None:
        """Gets the current account balance (practice by default)."""
        logger.info("Getting account balance.")
        await self.client.change_account("PRACTICE")
        balance = await self.client.get_balance()
        logger.info(f"Current balance: {balance}")
        print(f"💰 Current Balance: R$ {balance:.2f}")

    @ensure_connection()
    async def get_profile(self) -> None:
        """Gets user profile information."""
        logger.info("Getting user profile.")

        profile = await self.client.get_profile()

        description = (
            f"\n👤 User Profile:\n"
            f"Name: {profile.nick_name}\n"
            f"Demo Balance: R$ {profile.demo_balance:.2f}\n"
            f"Live Balance: R$ {profile.live_balance:.2f}\n"
            f"ID: {profile.profile_id}\n"
            f"Avatar: {profile.avatar}\n"
            f"Country: {profile.country_name}\n"
            f"Time Zone: {profile.offset}\n"
        )
        logger.info("Profile retrieved successfully.")
        print(description)

    @ensure_connection()
    async def buy_simple(self, amount: float = 50, asset: str = "EURUSD_otc",
                         direction: str = "call", duration: int = 60) -> None:
        """Executes a simple buy operation."""
        logger.info(f"Executing simple buy: {amount} on {asset} in {direction} direction for {duration}s.")

        await self.client.change_account("PRACTICE")
        asset_name, asset_data = await self.client.get_available_asset(asset, force_open=True)

        if not asset_data or len(asset_data) < 3 or not asset_data[2]:
            logger.error(f"Asset {asset} is closed or invalid.")
            print(f"❌ ERROR: Asset {asset} is closed or invalid.")
            return

        logger.info(f"Asset {asset} is open.")
        status, buy_info = await self.client.buy(
            amount, asset_name, direction, duration, time_mode="TIMER"
        )

        if status:
            logger.info(f"Buy successful: {buy_info}")
            print(f"✅ Buy executed successfully!")
            print(f"Amount: R$ {amount:.2f}")
            print(f"Asset: {asset}")
            print(f"Direction: {direction.upper()}")
            print(f"Duration: {duration}s")
            print(f"Order ID: {buy_info.get('id', 'N/A')}")
        else:
            logger.error(f"Buy failed: {buy_info}")
            print(f"❌ Buy failed: {buy_info}")

        balance = await self.client.get_balance()
        logger.info(f"Current balance: {balance}")
        print(f"💰 Current Balance: R$ {balance:.2f}")

    @ensure_connection()
    async def buy_and_check_win(self, amount: float = 50, asset: str = "EURUSD_otc",
                                direction: str = "put", duration: int = 60) -> None:
        """Executes a buy operation and checks if it was a win or loss."""
        logger.info(
            f"Executing buy and checking result: {amount} on {asset} in {direction} direction for {duration}s.")

        await self.client.change_account("PRACTICE")
        balance_before = await self.client.get_balance()
        logger.info(f"Balance before trade: {balance_before}")
        print(f"💰 Balance Before: R$ {balance_before:.2f}")

        asset_name, asset_data = await self.client.get_available_asset(asset, force_open=True)

        if not asset_data or len(asset_data) < 3 or not asset_data[2]:
            logger.error(f"Asset {asset} is closed or invalid.")
            print(f"❌ ERROR: Asset {asset} is closed or invalid.")
            return

        logger.info(f"Asset {asset} is open.")
        status, buy_info = await self.client.buy(amount, asset_name, direction, duration,
                                                 time_mode="TIMER")

        if not status:
            logger.error(f"Buy operation failed: {buy_info}")
            print(f"❌ Buy operation failed! Details: {buy_info}")
            return

        print(f"📊 Trade executed (ID: {buy_info.get('id', 'N/A')}), waiting for result...")
        logger.info(f"Waiting for trade result ID: {buy_info.get('id', 'N/A')}...")

        if await self.client.check_win(buy_info["id"]):
            profit = self.client.get_profit()
            logger.info(f"WIN! Profit: {profit}")
            print(f"🎉 WIN! Profit: R$ {profit:.2f}")
        else:
            loss = self.client.get_profit()
            logger.info(f"LOSS! Loss: {loss}")
            print(f"💔 LOSS! Loss: R$ {loss:.2f}")

        balance_after = await self.client.get_balance()
        logger.info(f"Balance after trade: {balance_after}")
        print(f"💰 Current Balance: R$ {balance_after:.2f}")

    @ensure_connection()
    async def get_candles(self, asset: str = "CHFJPY_otc", period: int = 60,
                          offset: int = 3600) -> None:
        """Gets historical candle data (candlesticks)."""
        logger.info(f"Getting candles for {asset} with period of {period}s.")

        end_from_time = time.time()
        candles = await self.client.get_candles(asset, end_from_time, offset, period)

        if not candles:
            logger.warning("No candles found for the specified asset.")
            print("⚠️ No candles found for the specified asset.")
            return

        if not candles[0].get("open"):
            candles = process_candles(candles, period)

        candles_color = []
        if len(candles) > 0:
            candles_color = [get_color(candle) for candle in candles if 'open' in candle and 'close' in candle]
        else:
            logger.warning("Not enough candle data to determine colors.")

        logger.info(f"Retrieved {len(candles)} candles.")

        print(f"\n📈 Candles (Candlesticks) for {asset} (Period: {period}s):")
        print(f"Total candles: {len(candles)}")
        if candles_color:
            print(f"Colors of last 10 candles: {' '.join(candles_color[-10:])}")
        else:
            print("   Candle colors not available.")

        print("\n   Last 5 candles:")
        for i, candle in enumerate(candles[-5:]):
            color = candles_color[-(5 - i)] if candles_color and (5 - i) <= len(candles_color) else "N/A"
            emoji = "🟢" if color == "green" else ("🔴" if color == "red" else "⚪")
            print(
                f"{emoji} Open: {candle.get('open', 'N/A'):.4f} → Close: {candle.get('close', 'N/A'):.4f} (Time: {time.strftime('%H:%M:%S', time.localtime(candle.get('time', 0)))})")

    @ensure_connection()
    async def get_assets_status(self) -> None:
        """Gets the status of all available assets (open/closed)."""
        logger.info("Getting assets status.")

        print("\n📊 Assets Status:")
        open_count = 0
        closed_count = 0

        all_assets = self.client.get_all_asset_name()
        if not all_assets:
            logger.warning("Could not retrieve assets list.")
            print("⚠️ Could not retrieve assets list.")
            return

        for asset_info in all_assets:
            asset_symbol = asset_info[0]
            asset_display_name = asset_info[1]

            _, asset_open_data = await self.client.check_asset_open(asset_symbol)

            is_open = False
            if asset_open_data and len(asset_open_data) > 2:
                is_open = asset_open_data[2]

            status_text = "OPEN" if is_open else "CLOSED"
            emoji = "🟢" if is_open else "🔴"

            print(f"{emoji} {asset_display_name} ({asset_symbol}): {status_text}")

            if is_open:
                open_count += 1
            else:
                closed_count += 1

            logger.debug(f"Asset {asset_symbol}: {status_text}")

        print(f"\n📈 Summary: {open_count} open assets, {closed_count} closed assets.")

    @ensure_connection()
    async def get_payment_info(self) -> None:
        """Gets payment information (payout) for all assets."""
        logger.info("Getting payment information.")

        all_data = self.client.get_payment()
        if not all_data:
            logger.warning("No payment information found.")
            print("⚠️ No payment information found.")
            return

        print("\n💰 Payment Information (Payout):")
        print("-" * 50)

        for asset_name, asset_data in list(all_data.items())[:10]:
            profit_1m = asset_data.get("profit", {}).get("1M", "N/A")
            profit_5m = asset_data.get("profit", {}).get("5M", "N/A")
            is_open = asset_data.get("open", False)

            status_text = "OPEN" if is_open else "CLOSED"
            emoji = "🟢" if is_open else "🔴"

            print(f"{emoji} {asset_name} - {status_text}")
            print(f"1M Profit: {profit_1m}% | 5M Profit: {profit_5m}%")
            print("-" * 50)

    @ensure_connection()
    async def balance_refill(self, amount: float = 5000) -> None:
        """Refills the practice account balance."""
        logger.info(f"Refilling practice account balance with R$ {amount:.2f}.")

        await self.client.change_account("PRACTICE")
        result = await self.client.edit_practice_balance(amount)

        if result:
            logger.info(f"Balance refill successful: {result}")
            print(f"✅ Practice account balance refilled to R$ {amount:.2f} successfully!")
        else:
            logger.error("Balance refill failed.")
            print("❌ Practice account balance refill failed.")

        new_balance = await self.client.get_balance()
        print(f"💰 New Balance: R$ {new_balance:.2f}")

    @ensure_connection()
    async def get_realtime_price(self, asset: str = "EURJPY_otc") -> None:
        """Monitors the real-time price of an asset."""
        logger.info(f"Getting real-time price for {asset}.")

        asset_name, asset_data = await self.client.get_available_asset(asset, force_open=True)

        if not asset_data or len(asset_data) < 3 or not asset_data[2]:
            logger.error(f"Asset {asset} is closed or invalid for real-time monitoring.")
            print(f"❌ ERROR: Asset {asset} is closed or invalid for monitoring.")
            return

        logger.info(f"Asset {asset} is open. Starting real-time price monitoring.")
        await self.client.start_realtime_price(asset, 60)

        print(f"\n📊 Monitoring real-time price for {asset}")
        print("Press Ctrl+C to stop monitoring...")
        print("-" * 60)

        try:
            while True:
                candle_price_data = await self.client.get_realtime_price(asset_name)
                if candle_price_data:
                    latest_data = candle_price_data[-1]
                    timestamp = latest_data['time']
                    price = latest_data['price']
                    formatted_time = time.strftime('%H:%M:%S', time.localtime(timestamp))

                    print(f"📈 {asset} | {formatted_time} | Price: {price:.5f}", end="\r")
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Real-time price monitoring interrupted by user.")
            print("\n✅ Real-time monitoring stopped.")
        finally:
            await self.client.stop_realtime_price(asset_name)
            logger.info(f"Real-time price subscription for {asset_name} stopped.")

    @ensure_connection()
    async def get_signal_data(self) -> None:
        """Gets and monitors trading signal data."""
        logger.info("Getting trading signal data.")

        self.client.start_signals_data()
        print("\n📡 Monitoring trading signals...")
        print("Press Ctrl+C to stop monitoring...")
        print("-" * 60)

        try:
            while True:
                signals = self.client.get_signal_data()
                if signals:
                    print(f"🔔 New Signal Received:")
                    print(json.dumps(signals, indent=2,
                                     ensure_ascii=False))
                    print("-" * 60)
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Signal monitoring interrupted by user.")
            print("\n✅ Signal monitoring stopped.")
        finally:
            pass

    async def ml_status(self) -> None:
        """Show machine learning model status and performance."""
        logger.info("Getting machine learning model status.")
        
        print("\n🤖 Machine Learning Model Status:")
        print("=" * 50)
        
        if self.model is None:
            print("❌ No model loaded")
            return
        
        print("✅ Model loaded successfully")
        print(f"📊 Model Architecture: {self.model.count_params()} parameters")
        
        # Show performance statistics
        performance = self.get_model_performance()
        print(f"📈 Performance: {performance}")
        
        # Show recent training data if available
        if self.train_data:
            recent_trades = self.train_labels[-10:] if len(self.train_labels) >= 10 else self.train_labels
            recent_win_rate = sum(recent_trades) / len(recent_trades) if recent_trades else 0
            print(f"🎯 Recent Performance (last {len(recent_trades)} trades): {recent_win_rate:.1%}")
            
            # Show learning insights
            insights = self.get_learning_insights()
            print(f"🧠 Learning Insights: {insights}")
            
            # Show immediate learning updates
            if hasattr(self, 'immediate_updates'):
                print(f"⚡ Immediate Updates: {self.immediate_updates} real-time adaptations")
            
            # Show incremental learning stats
            inc_stats = self.get_incremental_learning_stats()
            print(f"🔄 Incremental Learning: {inc_stats['incremental_updates']} mini-batch updates")
            print(f"💾 Memory Buffer: {inc_stats['memory_buffer_size']}/{self.memory_buffer_size} ({inc_stats['memory_utilization']:.1f}%)")
            print(f"🧠 Learning Memory: {inc_stats['learning_memory_size']}/{self.learning_memory_size}")
            
            if 'recent_accuracy' in inc_stats:
                print(f"📊 Recent Accuracy: {inc_stats['recent_accuracy']:.3f} ± {inc_stats['accuracy_std']:.3f}")
        
        # Show model files
        import os
        import glob
        
        model_files = glob.glob(os.path.join('models', '*.keras'))
        if model_files:
            print(f"\n💾 Saved Models ({len(model_files)} files):")
            for model_file in sorted(model_files, key=os.path.getmtime, reverse=True)[:5]:
                file_size = os.path.getsize(model_file) / 1024  # KB
                mod_time = os.path.getmtime(model_file)
                mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                print(f"  📁 {os.path.basename(model_file)} ({file_size:.1f} KB) - {mod_time_str}")
        else:
            print("\n💾 No saved models found")

    async def ml_save(self, custom_name: str = None) -> None:
        """Manually save the current machine learning model."""
        logger.info("Manually saving machine learning model.")
        
        if self.model is None:
            print("❌ No model to save. Please run auto-trade first to create a model.")
            return
        
        version_suffix = f"_{custom_name}" if custom_name else "_manual"
        self.save_model(version_suffix)
        
        performance = self.get_model_performance()
        print(f"✅ Model saved successfully!")
        print(f"📊 Current Performance: {performance}")

    async def ml_analyze(self) -> None:
        """Show detailed learning analysis and patterns."""
        logger.info("Performing detailed machine learning analysis.")
        
        print("\n🔍 Detailed Learning Analysis:")
        print("=" * 60)
        
        if not self.train_data or len(self.train_data) < 3:
            print("❌ Insufficient data for analysis (need at least 3 trades)")
            return
        
        # Basic statistics
        total_trades = len(self.train_labels)
        wins = sum(self.train_labels)
        losses = total_trades - wins
        win_rate = wins / total_trades
        
        print(f"📊 Basic Statistics:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Wins: {wins} ({win_rate:.1%})")
        print(f"   Losses: {losses} ({(1-win_rate):.1%})")
        
        # Streak analysis
        print(f"\n🔥 Streak Analysis:")
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        current_type = None
        streak_history = []
        
        for i, result in enumerate(self.train_labels):
            if result == 1:  # Win
                if current_type == "win":
                    current_streak += 1
                else:
                    if current_type == "loss":
                        max_loss_streak = max(max_loss_streak, current_streak)
                        streak_history.append(f"Loss streak: {current_streak}")
                    current_streak = 1
                    current_type = "win"
            else:  # Loss
                if current_type == "loss":
                    current_streak += 1
                else:
                    if current_type == "win":
                        max_win_streak = max(max_win_streak, current_streak)
                        streak_history.append(f"Win streak: {current_streak}")
                    current_streak = 1
                    current_type = "loss"
        
        # Update final streaks
        if current_type == "win":
            max_win_streak = max(max_win_streak, current_streak)
            streak_history.append(f"Current win streak: {current_streak}")
        else:
            max_loss_streak = max(max_loss_streak, current_streak)
            streak_history.append(f"Current loss streak: {current_streak}")
        
        print(f"   Max Win Streak: {max_win_streak}")
        print(f"   Max Loss Streak: {max_loss_streak}")
        print(f"   Current Streak: {current_streak} {'wins' if current_type == 'win' else 'losses'}")
        
        # Performance over time
        print(f"\n📈 Performance Over Time:")
        if total_trades >= 10:
            # Divide into quarters
            quarter_size = total_trades // 4
            quarters = []
            for i in range(4):
                start = i * quarter_size
                end = start + quarter_size if i < 3 else total_trades
                quarter_trades = self.train_labels[start:end]
                quarter_rate = sum(quarter_trades) / len(quarter_trades) if quarter_trades else 0
                quarters.append(quarter_rate)
                print(f"   Quarter {i+1}: {quarter_rate:.1%} win rate ({len(quarter_trades)} trades)")
        
        # Recent vs Overall performance
        print(f"\n🎯 Recent vs Overall Performance:")
        if total_trades >= 10:
            recent_10 = self.train_labels[-10:]
            recent_rate = sum(recent_10) / len(recent_10)
            print(f"   Last 10 trades: {recent_rate:.1%}")
            print(f"   Overall: {win_rate:.1%}")
            
            if recent_rate > win_rate + 0.1:
                print(f"   📈 Recent performance is BETTER than overall (+{(recent_rate - win_rate):.1%})")
            elif recent_rate < win_rate - 0.1:
                print(f"   📉 Recent performance is WORSE than overall ({(recent_rate - win_rate):.1%})")
            else:
                print(f"   ➡️ Recent performance is CONSISTENT with overall")
        
        # Learning insights
        print(f"\n🧠 Learning Insights:")
        insights = self.get_learning_insights()
        print(f"   {insights}")
        
        # Learning stats
        print(f"\n⚡ Learning Statistics:")
        if hasattr(self, 'immediate_updates'):
            print(f"   Immediate Updates: {self.immediate_updates}")
            print(f"   Updates per Trade: {self.immediate_updates / total_trades:.1f}")
        
        # Incremental learning stats
        inc_stats = self.get_incremental_learning_stats()
        print(f"   Incremental Updates: {inc_stats['incremental_updates']}")
        print(f"   Memory Buffer: {inc_stats['memory_buffer_size']}/{self.memory_buffer_size} ({inc_stats['memory_utilization']:.1f}%)")
        print(f"   Learning Memory: {inc_stats['learning_memory_size']}/{self.learning_memory_size}")
        
        if 'recent_accuracy' in inc_stats:
            print(f"   Recent Accuracy: {inc_stats['recent_accuracy']:.3f} ± {inc_stats['accuracy_std']:.3f}")
        
        # Catastrophic forgetting check
        is_forgetting, forget_message = self.detect_catastrophic_forgetting()
        if is_forgetting:
            print(f"   🚨 Catastrophic Forgetting: {forget_message}")
        else:
            print(f"   ✅ Memory Stability: {forget_message}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if win_rate < 0.4:
            print("   ⚠️ Low win rate - consider adjusting strategy or parameters")
        elif win_rate > 0.7:
            print("   🔥 High win rate - excellent performance!")
        
        if max_loss_streak >= 5:
            print("   🚨 High loss streaks detected - consider risk management")
        
        if hasattr(self, 'immediate_updates') and self.immediate_updates < total_trades * 0.8:
            print("   ⚡ Consider enabling more immediate learning updates")
        
        print(f"\n📁 Model Files:")
        import os
        import glob
        model_files = glob.glob(os.path.join('models', '*.keras'))
        if model_files:
            for model_file in sorted(model_files, key=os.path.getmtime, reverse=True)[:3]:
                file_size = os.path.getsize(model_file) / 1024
                mod_time = os.path.getmtime(model_file)
                mod_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mod_time))
                print(f"   📄 {os.path.basename(model_file)} ({file_size:.1f} KB) - {mod_time_str}")
        else:
            print("   No model files found")

    async def ml_incremental_management(self, replay=False, buffer_size=None, batch_size=None):
        """Manage incremental learning settings and perform operations."""
        logger.info("Managing incremental learning settings.")
        
        print("\n🔄 Incremental Learning Management:")
        print("=" * 50)
        
        # Update settings if provided
        if buffer_size is not None:
            old_size = self.memory_buffer_size
            self.memory_buffer_size = buffer_size
            print(f"📊 Memory buffer size: {old_size} → {buffer_size}")
            
            # Adjust buffer if new size is smaller
            if len(self.memory_buffer) > buffer_size:
                self.memory_buffer = self.memory_buffer[-buffer_size:]
                print(f"   Trimmed buffer to {len(self.memory_buffer)} experiences")
        
        if batch_size is not None:
            old_size = self.incremental_batch_size
            self.incremental_batch_size = batch_size
            print(f"📦 Incremental batch size: {old_size} → {batch_size}")
        
        # Show current settings
        print(f"\n⚙️ Current Settings:")
        print(f"   Memory Buffer Size: {self.memory_buffer_size}")
        print(f"   Incremental Batch Size: {self.incremental_batch_size}")
        print(f"   Learning Memory Size: {self.learning_memory_size}")
        print(f"   Forgetting Threshold: {self.catastrophic_forgetting_threshold}")
        
        # Show current stats
        inc_stats = self.get_incremental_learning_stats()
        print(f"\n📈 Current Statistics:")
        print(f"   Memory Buffer: {inc_stats['memory_buffer_size']}/{self.memory_buffer_size} ({inc_stats['memory_utilization']:.1f}%)")
        print(f"   Learning Memory: {inc_stats['learning_memory_size']}/{self.learning_memory_size}")
        print(f"   Incremental Updates: {inc_stats['incremental_updates']}")
        
        if 'recent_accuracy' in inc_stats:
            print(f"   Recent Accuracy: {inc_stats['recent_accuracy']:.3f} ± {inc_stats['accuracy_std']:.3f}")
        
        # Perform experience replay if requested
        if replay:
            if self.model is None:
                print("\n❌ No model loaded. Please run auto-trade first to create a model.")
                return
            
            if len(self.memory_buffer) < self.incremental_batch_size:
                print(f"\n❌ Insufficient memory buffer ({len(self.memory_buffer)}/{self.incremental_batch_size}). Need more trading data.")
                return
            
            print(f"\n🔄 Performing Experience Replay Training...")
            print(f"   Buffer Size: {len(self.memory_buffer)} experiences")
            print(f"   Batch Size: {self.incremental_batch_size}")
            
            success = self.experience_replay_training(replay_ratio=0.4)
            
            if success:
                print("✅ Experience replay completed successfully!")
                
                # Show updated stats
                updated_stats = self.get_incremental_learning_stats()
                if 'recent_accuracy' in updated_stats:
                    print(f"📊 Updated Accuracy: {updated_stats['recent_accuracy']:.3f}")
            else:
                print("❌ Experience replay failed!")
        
        # Check for catastrophic forgetting
        print(f"\n🧠 Memory Stability Check:")
        is_forgetting, message = self.detect_catastrophic_forgetting()
        if is_forgetting:
            print(f"   🚨 {message}")
            print("   💡 Consider running experience replay: python app.py ml-incremental --replay")
        else:
            print(f"   ✅ {message}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if inc_stats['memory_utilization'] < 50:
            print("   📈 Low memory utilization - consider reducing buffer size")
        elif inc_stats['memory_utilization'] > 90:
            print("   ⚠️ High memory utilization - consider increasing buffer size")
        
        if inc_stats['incremental_updates'] < 10:
            print("   🔄 Few incremental updates - model may need more trading data")
        
        if 'recent_accuracy' in inc_stats and inc_stats['recent_accuracy'] < 0.5:
            print("   📉 Low recent accuracy - consider experience replay or full retraining")

    async def ml_show_patterns(self) -> None:
        """Show learned patterns and their success rates."""
        logger.info("Displaying learned patterns.")
        
        print("\n🧠 Learned Patterns Analysis:")
        print("=" * 60)
        
        if not hasattr(self, 'learning_patterns') or not self.learning_patterns:
            print("❌ No patterns learned yet. Run auto-trade to start learning.")
            return
        
        total_patterns = 0
        confident_patterns = 0
        
        for pattern_type, patterns in self.learning_patterns.items():
            if not patterns:
                continue
                
            print(f"\n📊 {pattern_type.replace('_', ' ').title()}:")
            print("-" * 40)
            
            # Sort patterns by confidence and success rate
            sorted_patterns = sorted(
                patterns.items(), 
                key=lambda x: (x[1]['confidence'], x[1]['success_rate']), 
                reverse=True
            )
            
            for pattern_key, stats in sorted_patterns[:10]:  # Show top 10
                total_patterns += 1
                if stats['confidence'] > 0.7:
                    confident_patterns += 1
                
                success_rate = stats['success_rate']
                confidence = stats['confidence']
                wins = stats['wins']
                total = stats['total']
                
                # Color coding for success rate
                if success_rate > 0.7:
                    rate_emoji = "🟢"
                elif success_rate > 0.5:
                    rate_emoji = "🟡"
                else:
                    rate_emoji = "🔴"
                
                # Confidence indicator
                if confidence > 0.8:
                    conf_emoji = "🔥"
                elif confidence > 0.6:
                    conf_emoji = "⚡"
                else:
                    conf_emoji = "📊"
                
                print(f"  {rate_emoji} {conf_emoji} {pattern_key[:50]}{'...' if len(pattern_key) > 50 else ''}")
                print(f"     Success: {success_rate:.1%} ({wins}/{total}) | Confidence: {confidence:.1%}")
        
        # Summary
        print(f"\n📈 Pattern Summary:")
        print(f"   Total Patterns: {total_patterns}")
        print(f"   Confident Patterns: {confident_patterns}")
        print(f"   Learning Confidence: {getattr(self, 'learning_confidence', 0.5):.1%}")
        print(f"   Current Strategy: {getattr(self, 'current_strategy', 'neutral').upper()}")
        print(f"   Strategy Confidence: {getattr(self, 'strategy_confidence', 0.5):.1%}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if confident_patterns < 5:
            print("   📚 Need more trading data to build confident patterns")
        elif confident_patterns > 20:
            print("   🎯 Good pattern diversity - system is learning well")
        
        if getattr(self, 'learning_confidence', 0.5) < 0.6:
            print("   🔄 Learning confidence is low - consider more diverse trading")
        else:
            print("   ✅ Learning confidence is good - patterns are reliable")

    async def ml_save_all(self) -> None:
        """Save all learning data (TensorFlow model + patterns)."""
        logger.info("Saving all learning data.")
        
        print("\n💾 Saving All Learning Data:")
        print("=" * 40)
        
        # Save TensorFlow model if available
        if self.model is not None:
            print("🤖 Saving TensorFlow model...")
            self.save_model("_manual_save")
            print("✅ TensorFlow model saved")
        else:
            print("⚠️ No TensorFlow model to save")
        
        # Save rule-based patterns
        if hasattr(self, 'learning_patterns'):
            print("🧠 Saving learning patterns...")
            self.save_learning_patterns()
            print("✅ Learning patterns saved")
        else:
            print("⚠️ No learning patterns to save")
        
        # Save training data
        if hasattr(self, 'train_data') and self.train_data:
            print("📊 Saving training data...")
            self.save_training_data()
            print("✅ Training data saved")
        else:
            print("⚠️ No training data to save")
        
        print("\n🎯 Summary:")
        print(f"   TensorFlow Model: {'✅ Saved' if self.model is not None else '❌ Not available'}")
        print(f"   Learning Patterns: {'✅ Saved' if hasattr(self, 'learning_patterns') else '❌ Not available'}")
        print(f"   Training Data: {'✅ Saved' if hasattr(self, 'train_data') and self.train_data else '❌ Not available'}")
        
        print("\n📁 Files created:")
        import os
        if os.path.exists('models'):
            model_files = [f for f in os.listdir('models') if f.endswith('.keras')]
            print(f"   models/: {len(model_files)} model files")
        
        if os.path.exists('patterns'):
            pattern_files = [f for f in os.listdir('patterns') if f.endswith('.json')]
            print(f"   patterns/: {len(pattern_files)} pattern files")
        
        print("\n✅ All learning data saved successfully!")

    async def monitor_stats(self) -> None:
        """Show monitoring statistics and trade metrics."""
        logger.info("Displaying monitoring statistics.")
        
        print("\n📊 Trading Monitor Statistics:")
        print("=" * 50)
        
        # Get monitor stats
        stats = self.monitor.get_stats()
        
        print(f"📈 Rolling Accuracy: {stats['rolling_accuracy']:.2%} (last {stats['recent_trades']} trades)")
        print(f"💰 Total Profit: R$ {stats['total_profit']:.2f}")
        print(f"🔢 Total Trades: {stats['total_trades']}")
        print(f"🔥 Current Win Streak: {self.win_streak}")
        print(f"💔 Current Loss Streak: {self.loss_streak}")
        
        # Show CSV file info
        import os
        if os.path.exists("trade_metrics.csv"):
            file_size = os.path.getsize("trade_metrics.csv")
            print(f"📄 Metrics File: trade_metrics.csv ({file_size} bytes)")
            
            # Show recent entries
            try:
                with open("trade_metrics.csv", 'r') as f:
                    lines = f.readlines()
                    if len(lines) > 1:  # More than just header
                        print(f"📋 Total Records: {len(lines) - 1}")
                        
                        # Show last 5 entries
                        print(f"\n📝 Recent Trades (last 5):")
                        print("-" * 80)
                        print("Timestamp           | Accuracy | Total Profit | Last Profit | Strategy")
                        print("-" * 80)
                        
                        for line in lines[-5:]:
                            if line.strip() and not line.startswith("timestamp"):
                                parts = line.strip().split(',')
                                if len(parts) >= 9:
                                    timestamp = parts[0][:19]  # Truncate timestamp
                                    accuracy = f"{float(parts[1]):.1%}"
                                    total_profit = f"R$ {float(parts[2]):.2f}"
                                    last_profit = f"R$ {float(parts[3]):.2f}"
                                    strategy = parts[7]
                                    print(f"{timestamp} | {accuracy:>8} | {total_profit:>11} | {last_profit:>10} | {strategy}")
            except Exception as e:
                print(f"⚠️ Could not read CSV file: {e}")
        else:
            print("📄 Metrics File: trade_metrics.csv (not created yet)")
        
        # Show performance analysis
        if stats['recent_trades'] >= 10:
            print(f"\n🎯 Performance Analysis:")
            if stats['rolling_accuracy'] > 0.7:
                print("   🟢 Excellent performance! Keep it up!")
            elif stats['rolling_accuracy'] > 0.5:
                print("   🟡 Good performance, room for improvement")
            else:
                print("   🔴 Performance needs improvement")
            
            if stats['total_profit'] > 0:
                print(f"   💰 Profitable trading: +R$ {stats['total_profit']:.2f}")
            else:
                print(f"   📉 Losses so far: R$ {stats['total_profit']:.2f}")
        
        print(f"\n💡 Recommendations:")
        if stats['recent_trades'] < 10:
            print("   📚 Need more trades for reliable statistics")
        elif stats['rolling_accuracy'] < 0.4:
            print("   🔄 Consider adjusting strategy or learning parameters")
        elif self.loss_streak > 5:
            print("   ⚠️ High loss streak - consider risk management")
        elif self.win_streak > 10:
            print("   🎉 Great win streak! Consider increasing position size carefully")

    async def nn_insights(self) -> None:
        """Show neural network insights and performance analysis."""
        logger.info("Displaying neural network insights.")
        
        print("\n🧠 Neural Network Insights:")
        print("=" * 60)
        
        if self.model is None:
            print("❌ No neural network model available")
            return
        
        # Model architecture info
        print(f"🏗️ Model Architecture:")
        print(f"   Parameters: {self.model.count_params():,}")
        print(f"   Layers: {len(self.model.layers)}")
        print(f"   Feature Size: {self.feature_size}")
        
        # Show layer details
        print(f"\n📊 Layer Details:")
        for i, layer in enumerate(self.model.layers):
            layer_type = type(layer).__name__
            if hasattr(layer, 'units'):
                print(f"   Layer {i+1}: {layer_type} - {layer.units} units")
            else:
                print(f"   Layer {i+1}: {layer_type}")
        
        # Performance analysis
        print(f"\n📈 Performance Analysis:")
        performance = self.analyze_neural_network_performance()
        
        if isinstance(performance, dict):
            print(f"   Recent Accuracy: {performance.get('recent_accuracy', 0):.2%}")
            print(f"   Average Confidence: {performance.get('avg_confidence', 0):.2%}")
            print(f"   High Confidence Accuracy: {performance.get('high_confidence_accuracy', 0):.2%}")
            print(f"   Total Predictions: {performance.get('total_predictions', 0)}")
            print(f"   High Confidence Predictions: {performance.get('high_confidence_predictions', 0)}")
        else:
            print(f"   {performance}")
        
        # Training data info
        print(f"\n📚 Training Data:")
        print(f"   Total Samples: {len(self.train_data)}")
        print(f"   Feature Size: {self.feature_size}")
        if self.train_data:
            wins = sum(self.train_labels)
            losses = len(self.train_labels) - wins
            print(f"   Wins: {wins} ({wins/len(self.train_labels):.1%})")
            print(f"   Losses: {losses} ({losses/len(self.train_labels):.1%})")
        
        # Model compilation info
        print(f"\n⚙️ Model Configuration:")
        print(f"   Optimizer: {self.model.optimizer.__class__.__name__}")
        print(f"   Loss Function: {self.model.loss}")
        print(f"   Metrics: {[metric.name for metric in self.model.metrics]}")
        
        # Learning rate info
        if hasattr(self.model.optimizer, 'learning_rate'):
            lr = self.model.optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                print(f"   Current Learning Rate: {lr.numpy():.6f}")
            else:
                print(f"   Current Learning Rate: {lr:.6f}")
        
        # Callbacks info
        print(f"\n🔄 Training Callbacks:")
        print(f"   Early Stopping: Enabled (patience=10)")
        print(f"   Model Checkpoint: Enabled (save best only)")
        print(f"   Learning Rate Schedule: Exponential Decay")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if isinstance(performance, dict):
            if performance.get('recent_accuracy', 0) > 0.7:
                print("   🟢 Excellent neural network performance!")
            elif performance.get('recent_accuracy', 0) > 0.5:
                print("   🟡 Good performance, continue training")
            else:
                print("   🔴 Performance needs improvement - consider more training data")
            
            if performance.get('avg_confidence', 0) > 0.8:
                print("   🎯 High confidence predictions - model is well-calibrated")
            elif performance.get('avg_confidence', 0) < 0.5:
                print("   ⚠️ Low confidence - model may need more training")
        
        if len(self.train_data) < 50:
            print("   📚 Need more training data for better performance")
        elif len(self.train_data) > 200:
            print("   🎉 Sufficient training data - model should be well-trained")
        
        print(f"\n🔬 Advanced Features:")
        print(f"   ✅ Batch Normalization: Prevents internal covariate shift")
        print(f"   ✅ Dropout Regularization: Prevents overfitting")
        print(f"   ✅ Learning Rate Decay: Adaptive learning rate")
        print(f"   ✅ Early Stopping: Prevents overtraining")
        print(f"   ✅ Model Checkpointing: Saves best model")


def create_parser() -> argparse.ArgumentParser:
    """Creates and configures the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="PyQuotex CLI - Trading automation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python app.py test-connection
  python app.py get-balance
  python app.py buy-simple --amount 100 --asset EURUSD_otc --direction call
  python app.py get-candles --asset GBPUSD --period 300
  python app.py realtime-price --asset EURJPY_otc
  python app.py signals
  python app.py auto-trade --amount 50 --asset EURUSD_otc --direction call --duration 60 --interval 60
  python app.py ml-status
  python app.py ml-analyze
  python app.py ml-patterns
  python app.py ml-save-all
  python app.py monitor-stats
  python app.py nn-insights
  python app.py ml-incremental --replay
  python app.py ml-incremental --buffer-size 2000 --batch-size 64
  python app.py ml-save --name "my_model_v1"
        """
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"PyQuotex {__version__}"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable detailed logging mode (DEBUG)."
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress most output except errors."
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("test-connection", help="Test connection to Quotex API.")

    subparsers.add_parser("get-balance", help="Get current account balance (practice by default).")

    subparsers.add_parser("get-profile", help="Get user profile information.")

    buy_parser = subparsers.add_parser("buy-simple", help="Execute a simple buy operation.")
    buy_parser.add_argument("--amount", type=float, default=50, help="Amount to invest.")
    buy_parser.add_argument("--asset", default="EURUSD_otc", help="Asset to trade.")
    buy_parser.add_argument("--direction", choices=["call", "put"], default="call",
                            help="Trade direction (call for up, put for down).")
    buy_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds.")

    buy_check_parser = subparsers.add_parser("buy-and-check", help="Execute a buy and check win/loss.")
    buy_check_parser.add_argument("--amount", type=float, default=50, help="Amount to invest.")
    buy_check_parser.add_argument("--asset", default="EURUSD_otc", help="Asset to trade.")
    buy_check_parser.add_argument("--direction", choices=["call", "put"], default="put",
                                  help="Trade direction.")
    buy_check_parser.add_argument("--duration", type=int, default=60, help="Duration in seconds.")

    candles_parser = subparsers.add_parser("get-candles", help="Get historical candle data (candlesticks).")
    candles_parser.add_argument("--asset", default="CHFJPY_otc", help="Asset to get candles for.")
    candles_parser.add_argument("--period", type=int, default=60,
                                help="Candle period in seconds (e.g., 60 for 1 minute).")
    candles_parser.add_argument("--offset", type=int, default=3600, help="Offset in seconds to fetch candles.")

    subparsers.add_parser("assets-status", help="Get status (open/closed) of all available assets.")

    subparsers.add_parser("payment-info", help="Get payment information (payout) for all assets.")

    refill_parser = subparsers.add_parser("balance-refill", help="Refill practice account balance.")
    refill_parser.add_argument("--amount", type=float, default=5000, help="Amount to refill practice account.")

    price_parser = subparsers.add_parser("realtime-price", help="Monitor real-time price of an asset.")
    price_parser.add_argument("--asset", default="EURJPY_otc", help="Asset to monitor.")


    auto_trade_parser = subparsers.add_parser("auto-trade", help="Automatically execute trades every interval.")
    auto_trade_parser.add_argument("--amount", type=float, default=50, help="Amount to trade.")
    auto_trade_parser.add_argument("--asset", type=str, default="EURUSD_otc", help="Asset to trade.")
    auto_trade_parser.add_argument("--direction", type=str, default="call", help="Trade direction (call/put).")
    auto_trade_parser.add_argument("--duration", type=int, default=60, help="Trade duration in seconds.")
    auto_trade_parser.add_argument("--interval", type=int, default=60, help="Interval between trades in seconds.")

    subparsers.add_parser("signals", help="Monitor trading signal data.")

    # Machine Learning commands
    ml_subparser = subparsers.add_parser("ml-status", help="Show machine learning model status and performance.")
    
    ml_save_parser = subparsers.add_parser("ml-save", help="Manually save the current machine learning model.")
    ml_save_parser.add_argument("--name", type=str, help="Custom name for the model save.")
    
    ml_analyze_parser = subparsers.add_parser("ml-analyze", help="Show detailed learning analysis and patterns.")
    
    ml_incremental_parser = subparsers.add_parser("ml-incremental", help="Manage incremental learning settings and perform experience replay.")
    ml_incremental_parser.add_argument("--replay", action="store_true", help="Perform experience replay training.")
    ml_incremental_parser.add_argument("--buffer-size", type=int, help="Set memory buffer size.")
    ml_incremental_parser.add_argument("--batch-size", type=int, help="Set incremental batch size.")
    
    ml_patterns_parser = subparsers.add_parser("ml-patterns", help="Show learned patterns and their success rates.")
    
    ml_save_all_parser = subparsers.add_parser("ml-save-all", help="Save all learning data (TensorFlow model + patterns).")
    
    monitor_parser = subparsers.add_parser("monitor-stats", help="Show monitoring statistics and trade metrics.")
    
    nn_parser = subparsers.add_parser("nn-insights", help="Show neural network insights and performance analysis.")

    return parser


async def main():
    """Main entry point of the CLI application."""
    parser = create_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)

    cli = PyQuotexCLI()

    if not args.quiet:
        cli.display_banner()
        await asyncio.sleep(1)

    try:
        if args.command == "test-connection":
            await cli.test_connection()
        elif args.command == "get-balance":
            await cli.get_balance()
        elif args.command == "get-profile":
            await cli.get_profile()
        elif args.command == "buy-simple":
            await cli.buy_simple(args.amount, args.asset, args.direction, args.duration)
        elif args.command == "buy-and-check":
            await cli.buy_and_check_win(args.amount, args.asset, args.direction, args.duration)
        elif args.command == "get-candles":
            await cli.get_candles(args.asset, args.period, args.offset)
        elif args.command == "assets-status":
            await cli.get_assets_status()
        elif args.command == "payment-info":
            await cli.get_payment_info()
        elif args.command == "balance-refill":
            await cli.balance_refill(args.amount)
        elif args.command == "realtime-price":
            await cli.get_realtime_price(args.asset)
        elif args.command == "signals":
            await cli.get_signal_data()
        elif args.command == "auto-trade":
            await cli.auto_trade(args.amount, args.asset, args.direction, args.duration, args.interval)
        elif args.command == "ml-status":
            await cli.ml_status()
        elif args.command == "ml-save":
            await cli.ml_save(args.name if hasattr(args, 'name') else None)
        elif args.command == "ml-analyze":
            await cli.ml_analyze()
        elif args.command == "ml-incremental":
            await cli.ml_incremental_management(
                replay=args.replay if hasattr(args, 'replay') else False,
                buffer_size=args.buffer_size if hasattr(args, 'buffer_size') else None,
                batch_size=args.batch_size if hasattr(args, 'batch_size') else None
            )
        elif args.command == "ml-patterns":
            await cli.ml_show_patterns()
        elif args.command == "ml-save-all":
            await cli.ml_save_all()
        elif args.command == "monitor-stats":
            await cli.monitor_stats()
        elif args.command == "nn-insights":
            await cli.nn_insights()
        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("CLI operation interrupted by user.")
        print("\n✅ Operation interrupted by user.")
    except ConnectionError as e:
        logger.error(f"Connection error during command execution: {e}")
        print(f"❌ Connection error: {e}")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"❌ Error: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error occurred during command execution: {e}", exc_info=True)
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n✅ Program terminated by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}", exc_info=True)
        print(f"❌ FATAL ERROR: {e}")
        sys.exit(1)
