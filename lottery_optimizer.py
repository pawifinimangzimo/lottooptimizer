import yaml
import pandas as pd
import numpy as np
import random
import sympy
from pathlib import Path
import json
from datetime import datetime
import argparse
import collections
from collections import defaultdict
from scipy.stats import chisquare
import traceback

class AdaptiveLotteryOptimizer:
    def __init__(self, config_path="config.yaml"):
        self.config = None
        self.historical = None
        self.upcoming = None
        self.latest_draw = None
        self.number_pool = None
        self.decay_factor = 0.97
        self.prime_numbers = None
        self.frequencies = None
        self.recent_counts = None
        self.cold_numbers = set()
        self.overrepresented_pairs = set()
        self.weights = None
        self.last_generated_sets = None
        self.high_performance_numbers = set()
        
        self.load_config(config_path)
        self.initialize_number_properties()
        self.prepare_filesystem()
        self.load_and_clean_data()
        self.validate_data()
        self.analyze_numbers()
        self.validator = AdaptiveLotteryValidator(self)
        self.args = None  # Will be set in main()
        

        if self.config['output']['verbose']:
            print("\nSYSTEM INITIALIZED WITH:")
            print(f"- Number pool: 1-{self.config['strategy']['number_pool']}")
            print(f"- {len(self.historical)} historical draws loaded")
            if self.upcoming is not None:
                print(f"- {len(self.upcoming)} upcoming draws loaded")
            if self.latest_draw is not None:
                print(f"- Latest draw loaded: {self.latest_draw['date'].strftime('%m/%d/%y')} - {self.latest_draw['numbers']}")
            print(f"- {len(self.prime_numbers)} prime numbers in pool")
            print(f"- Current cold numbers: {sorted(int(n) for n in self.cold_numbers)}")
        # Add defaults if missing
            if 'analysis' not in self.config:
                self.config['analysis'] = {
                    'default_match_threshold': 4,
                    'default_show_top': 5,
                    'min_display_matches': 1
                }


    def initialize_number_properties(self):
        self.number_pool = list(range(1, self.config['strategy']['number_pool'] + 1))
        self.prime_numbers = [n for n in self.number_pool if sympy.isprime(n)]
        
        if self.config['output']['verbose']:
            print(f"Identified {len(self.prime_numbers)} primes: {self.prime_numbers}")

    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            defaults = {
                'data': {
                    'historical_path': 'data/historical.csv',
                    'upcoming_path': 'data/upcoming.csv',
                    'latest_path': 'data/latest_draw.csv',
                    'stats_dir': 'stats/',
                    'results_dir': 'results/',
                    'merge_upcoming': True,
                    'archive_upcoming': True
                },
                'validation': {
                    'mode': 'none',
                    'test_draws': 300,
                    'alert_threshold': 4,
                    'save_report': True
                },
                'strategy': {
                    'number_pool': 55,
                    'numbers_to_select': 6,
                    'frequency_weight': 0.4,
                    'recent_weight': 0.2,
                    'random_weight': 0.4,
                    'low_number_max': 10,
                    'low_number_chance': 0.7,
                    'high_prime_min': 35,
                    'high_prime_chance': 0.25,
                    'cold_threshold': 50,
                    'resurgence_threshold': 3
                },
                'output': {
                    'sets_to_generate': 4,
                    'save_analysis': True,
                    'verbose': True
                }
            }
            
            for section, values in defaults.items():
                if section not in self.config:
                    self.config[section] = values
                else:
                    for key, value in values.items():
                        if key not in self.config[section]:
                            self.config[section][key] = value
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            print("Config file should be in YAML format with proper indentation")
            raise

    def prepare_filesystem(self):
        try:
            Path(self.config['data']['stats_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['data']['results_dir']).mkdir(parents=True, exist_ok=True)
            if self.config['output']['verbose']:
                print(f"Created directories: {self.config['data']['stats_dir']}, {self.config['data']['results_dir']}")
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise

    def load_and_clean_data(self):
        try:
            num_select = self.config['strategy']['numbers_to_select']
            num_cols = [f'n{i+1}' for i in range(num_select)]
            
            hist_path = self.config['data']['historical_path']
            if self.config['output']['verbose']:
                print(f"\nLOADING DATA FROM: {hist_path}")
            
            self.historical = pd.read_csv(
                hist_path, 
                header=None, 
                names=['date', 'numbers'],
                dtype={'date': str, 'numbers': str}
            )
            
            self.historical[num_cols] = self.historical['numbers'].str.split('-', expand=True).astype(int)
            self.historical['date'] = pd.to_datetime(self.historical['date'], format='%m/%d/%y')
            
            if self.config['data']['upcoming_path'].strip():
                try:
                    self.upcoming = pd.read_csv(
                        self.config['data']['upcoming_path'],
                        header=None,
                        names=['date', 'numbers']
                    )
                    self.upcoming[num_cols] = self.upcoming['numbers'].str.split('-', expand=True).astype(int)
                    self.upcoming['date'] = pd.to_datetime(self.upcoming['date'], format='%m/%d/%y')
                    
                    if self.config['data']['merge_upcoming']:
                        self.historical = pd.concat([self.historical, self.upcoming])
                        
                except FileNotFoundError:
                    if self.config['output']['verbose']:
                        print("Note: Upcoming draws file not found")
            
            if self.config['data'].get('latest_path', '').strip():
                try:
                    latest = pd.read_csv(
                        self.config['data']['latest_path'],
                        header=None,
                        names=['date', 'numbers']
                    )
                    if not latest.empty:
                        latest[num_cols] = latest['numbers'].str.split('-', expand=True).astype(int)
                        latest['date'] = pd.to_datetime(latest['date'], format='%m/%d/%y')
                        self.latest_draw = latest.iloc[-1]
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    if self.config['output']['verbose']:
                        print("Note: Latest draw file not found or empty")
            
            if self.config['output']['verbose']:
                print(f"Successfully loaded {len(self.historical)} draws")
                
        except Exception as e:
            print(f"\nDATA LOADING ERROR: {str(e)}")
            print("Required format: MM/DD/YY,N1-N2-... (one draw per line)")
            print(f"Expected {num_select} numbers per draw")
            raise

    def validate_data(self):
        num_select = self.config['strategy']['numbers_to_select']
        num_cols = [f'n{i+1}' for i in range(num_select)]
        max_num = self.config['strategy']['number_pool']
        
        for col in num_cols:
            invalid = self.historical[
                (self.historical[col] < 1) | 
                (self.historical[col] > max_num)
            ]
            if not invalid.empty:
                raise ValueError(f"Invalid numbers found in column {col} (range 1-{max_num})")

    def analyze_numbers(self):
        num_select = self.config['strategy']['numbers_to_select']
        num_cols = [f'n{i+1}' for i in range(num_select)]
        
        numbers = self.historical[num_cols].values.flatten()
        self.frequencies = pd.Series(numbers).value_counts().sort_index()
        
        recent_draws = self.historical.iloc[-int(len(self.historical)*0.2):]
        recent_numbers = recent_draws[num_cols].values.flatten()
        self.recent_counts = pd.Series(recent_numbers).value_counts().reindex(
            self.number_pool, fill_value=0
        )
        
        last_n_draws = self.historical.iloc[-20:][num_cols].values.flatten()
        self.cold_numbers = set(self.number_pool) - set(last_n_draws)
        
        self._find_overrepresented_pairs()
        self.calculate_weights()

        if self.config['output']['verbose']:
            print("\nNUMBER ANALYSIS RESULTS:")
            print("Top 10 frequent numbers:")
            print(self.frequencies.nlargest(10))
            print("\nTop 10 recent numbers:")
            print(self.recent_counts.nlargest(10))
            print(f"\nCold numbers (not drawn in last 20 games): {sorted(int(n) for n in self.cold_numbers)}")
            if self.overrepresented_pairs:
                print("\nMost common number pairs:")
                for pair in sorted(self.overrepresented_pairs, key=lambda x: -self.weights[x[0]]*self.weights[x[1]])[:5]:
                    print(f"{pair[0]}-{pair[1]}")

    def _find_overrepresented_pairs(self):
        num_select = self.config['strategy']['numbers_to_select']
        num_cols = [f'n{i+1}' for i in range(num_select)]
        
        pair_counts = defaultdict(int)
        for _, row in self.historical.iterrows():
            nums = sorted(row[num_cols])
            for i in range(len(nums)):
                for j in range(i+1, len(nums)):
                    pair_counts[(nums[i], nums[j])] += 1
        
        total_draws = len(self.historical)
        pool_size = self.config['strategy']['number_pool']
        expected = total_draws * (num_select*(num_select-1)) / (pool_size*(pool_size-1))
        
        self.overrepresented_pairs = {
            pair for pair, count in pair_counts.items() 
            if count > expected * 1.5
        }

    def calculate_weights(self):
        base_weights = pd.Series(1.0, index=self.number_pool)
        
        if not self.frequencies.empty:
            freq_weights = (self.frequencies / self.frequencies.sum()).fillna(0)
            base_weights += freq_weights * self.config['strategy']['frequency_weight'] * 10
        
        recent_weights = (self.recent_counts / self.recent_counts.sum()).fillna(0)
        base_weights += recent_weights * self.config['strategy']['recent_weight'] * 5
        
        for num in self.high_performance_numbers:
            base_weights[num] *= 1.5
            
        for n1, n2 in self.overrepresented_pairs:
            base_weights[n1] *= 0.9
            base_weights[n2] *= 0.9
            
        for num in self.cold_numbers:
            base_weights[num] *= np.random.uniform(1.1, 1.3)
            
        random_weights = pd.Series(
            np.random.dirichlet(np.ones(len(self.number_pool))) * 0.7,
            index=self.number_pool
        )
        base_weights += random_weights * self.config['strategy']['random_weight'] * 15
        
        self.weights = base_weights / base_weights.sum()

        if self.config['output']['verbose']:
            print("\nTOP 10 WEIGHTED NUMBERS:")
            print(self.weights.sort_values(ascending=False).head(10))

    def generate_sets(self):
        strategies = [
            ('weighted_random', self._generate_weighted_random),
            ('high_low_mix', self._generate_high_low_mix),
            ('prime_balanced', self._generate_prime_balanced),
            ('performance_boosted', self._generate_performance_boosted)
        ]
        
        sets_per_strategy = max(1, self.config['output']['sets_to_generate'] // len(strategies))
        sets = []
        
        for name, strategy in strategies:
            for _ in range(sets_per_strategy):
                try:
                    numbers = strategy()
                    if len(numbers) == self.config['strategy']['numbers_to_select']:
                        sets.append((numbers, name))
                except Exception:
                    continue
        
        self.last_generated_sets = sets

        if self.config['output']['verbose']:
            label = "INITIAL NUMBER SETS:" if not self.high_performance_numbers else "ADAPTED NUMBER SETS:"
            print(f"\n{label}")
            for i, (nums, strategy) in enumerate(sets, 1):
                print(f"Set {i}: {'-'.join(str(int(n)) for n in nums)} ({strategy})")
        
        return sets

    def _generate_weighted_random(self):
        return sorted(np.random.choice(
            self.number_pool,
            size=self.config['strategy']['numbers_to_select'],
            replace=False,
            p=self.weights
        ))

    def _generate_high_low_mix(self):
        low_max = self.config['strategy']['low_number_max']
        low_nums = [n for n in self.number_pool if n <= low_max]
        high_nums = [n for n in self.number_pool if n > low_max]
        
        split_point = self.config['strategy']['numbers_to_select'] // 2
        selected = (
            list(np.random.choice(low_nums, split_point, replace=False, 
                p=self.weights[low_nums]/self.weights[low_nums].sum())) +
            list(np.random.choice(high_nums, self.config['strategy']['numbers_to_select'] - split_point, 
                replace=False, p=self.weights[high_nums]/self.weights[high_nums].sum()))
        )
        return sorted(selected)

    def _generate_prime_balanced(self):
        primes = self.prime_numbers
        non_primes = [n for n in self.number_pool if n not in primes]
        
        num_primes = np.random.choice([
            max(1, len(primes) // 3),
            len(primes) // 2,
            len(primes) // 2 + 1
        ])
        
        selected = (
            list(np.random.choice(primes, num_primes, replace=False,
                p=self.weights[primes]/self.weights[primes].sum())) +
            list(np.random.choice(non_primes, 
                self.config['strategy']['numbers_to_select'] - num_primes,
                replace=False, 
                p=self.weights[non_primes]/self.weights[non_primes].sum()))
        )
        return sorted(selected)

    def _generate_performance_boosted(self):
        if not self.high_performance_numbers:
            return self._generate_weighted_random()
            
        boosted_weights = self.weights.copy()
        for num in self.high_performance_numbers:
            boosted_weights[num] *= 2.0
            
        boosted_weights /= boosted_weights.sum()
        return sorted(np.random.choice(
            self.number_pool,
            size=self.config['strategy']['numbers_to_select'],
            replace=False,
            p=boosted_weights
        ))

    def generate_improved_sets(self, previous_results):
        changes = []
        prev_weights = self.weights.copy() if self.weights is not None else None
        
        if 'high_performance_sets' in previous_results:
            prev_high_performers = set(self.high_performance_numbers)
            new_performers = set()
            
            for nums in previous_results['high_performance_sets']:
                new_performers.update(nums)
            
            self.high_performance_numbers.update(new_performers)
            new_additions = set(self.high_performance_numbers) - prev_high_performers
            
            if new_additions:
                changes.append(f"New high-performers: {sorted([int(n) for n in new_additions])}")
        
        self.calculate_weights()
        
        if prev_weights is not None:
            top_changes = []
            prev_top = prev_weights.nlargest(5)
            current_top = self.weights.nlargest(5)
            
            for num in set(prev_top.index).union(set(current_top.index)):
                prev_rank = prev_top.index.get_loc(num) if num in prev_top.index else None
                curr_rank = current_top.index.get_loc(num) if num in current_top.index else None
                
                if prev_rank != curr_rank:
                    direction = "↑" if (curr_rank is not None and (prev_rank is None or curr_rank < prev_rank)) else "↓"
                    change = abs((self.weights[num] - prev_weights[num]) / prev_weights[num] * 100)
                    top_changes.append(f"{int(num)}{direction}{change:.1f}%")
            
            if top_changes:
                changes.append(f"Weight changes: {', '.join(top_changes)}")
        
        cold_used = [num for num in self.cold_numbers 
                    if num in (num for set_ in self.last_generated_sets for num in set_[0])]
        if cold_used:
            changes.append(f"Cold numbers included: {sorted([int(n) for n in cold_used])}")
        
        improved_sets = self.generate_sets()
        
        adaptation_report = {
            'sets': improved_sets,
            'changes': changes if changes else ["No significant changes - maintaining current strategy"]
        }
        
        if self.config['output']['verbose']:
            print("\n" + "="*60)
            print("ADAPTATION REPORT".center(60))
            print("="*60)
            for change in adaptation_report['changes']:
                print(f"- {change}")
            print("\nADAPTED NUMBER SETS:")
            for i, (nums, strategy) in enumerate(improved_sets, 1):
                print(f"Set {i}: {'-'.join(str(int(n)) for n in nums)} ({strategy})")
            print("="*60)
        
        return improved_sets

    def run_validation(self, mode=None):
        try:
            return self.validator.run(mode or self.config['validation']['mode'])
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {}

    def save_results(self, sets):
        try:
            output_file = Path(self.config['data']['results_dir']) / 'suggestions.csv'
            
            valid_sets = []
            for nums, strategy in sets:
                if (len(nums) == self.config['strategy']['numbers_to_select'] and 
                    len(set(nums)) == self.config['strategy']['numbers_to_select'] and 
                    all(1 <= n <= self.config['strategy']['number_pool'] for n in nums)):
                    valid_sets.append((nums, strategy))
                else:
                    print(f"Discarding invalid set: {nums} (strategy: {strategy})")
            
            if not valid_sets:
                raise ValueError("No valid sets to save")
            
            with open(output_file, 'w') as f:
                f.write("numbers,strategy\n")
                for nums, strategy in valid_sets:
                    f.write(f"{'-'.join(str(int(n)) for n in nums)},{strategy}\n")
                    
            if self.config['output']['verbose']:
                print(f"\nSAVED RESULTS TO: {output_file}")
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

    def run(self, mode):
        """Main validation method with enhanced statistics"""
        results = {}
        
        try:
            # Get historical data range
            test_draws = min(
                self.optimizer.config['validation']['test_draws'],
                len(self.optimizer.historical)
            )
            historical = self.optimizer.historical.iloc[-test_draws:]
            
            # Add recency analysis to all modes
            results['recency_stats'] = self._calculate_recency_stats(historical)
            
            if mode in ('historical', 'both'):
                results['historical'] = self.test_historical()
                results['historical'].update({
                    'number_types': self._analyze_number_types(historical),
                    'recency_distribution': self._get_recency_distribution(historical)
                })
                
            if mode in ('new_draw', 'both') and self.optimizer.upcoming is not None:
                results['new_draw'] = self.check_new_draws()
            
            if mode in ('latest', 'both') and self.optimizer.latest_draw is not None:
                results['latest'] = self.check_latest_draw()
                results['latest']['recency'] = self._get_numbers_recency(
                    results['latest']['draw_numbers'],
                    historical
                )
            
            if self.optimizer.config['validation']['save_report']:
                self.save_report(results)
            
            self.print_enhanced_results(results, historical)
            return results
        
        except Exception as e:
            print(f"Validation error: {str(e)}")
            traceback.print_exc()
            return {}

    def _calculate_recency_stats(self, historical):
        """Calculate recency metrics for all numbers"""
        num_cols = [f'n{i+1}' for i in range(
            self.optimizer.config['strategy']['numbers_to_select']
        )]
        stats = {}
        
        for num in self.optimizer.number_pool:
            recency, days_ago, _ = self._get_recency_info(num, historical, num_cols)
            stats[num] = {
                'recency': recency,
                'days_ago': days_ago,
                'is_cold': num in self.optimizer.cold_numbers
            }
        return stats

    def _analyze_number_types(self, historical):
        """Classify numbers as hot/warm/cold"""
        num_cols = [f'n{i+1}' for i in range(
            self.optimizer.config['strategy']['numbers_to_select']
        )]
        
        analysis = {
            'cold_numbers': list(self.optimizer.cold_numbers),
            'hot_numbers': [],
            'warm_numbers': [],
            'never_drawn': []
        }
        
        for num in self.optimizer.number_pool:
            recency, _, _ = self._get_recency_info(num, historical, num_cols)
            if recency is None:
                analysis['never_drawn'].append(num)
            elif recency <= 3:
                analysis['hot_numbers'].append(num)
            elif recency <= 10:
                analysis['warm_numbers'].append(num)
                
        return analysis

    def _get_recency_distribution(self, historical):
        """Calculate distribution of recency periods"""
        bins = {
            'hot': {'count': 0, 'percentage': 0},
            'warm': {'count': 0, 'percentage': 0},
            'cold': {'count': 0, 'percentage': 0},
            'never': {'count': 0, 'percentage': 0}
        }
        
        for num in self.optimizer.number_pool:
            recency = self.optimizer.config['analysis']['recency_stats'].get(num, {}).get('recency')
            if recency is None:
                bins['never']['count'] += 1
            elif recency <= 3:
                bins['hot']['count'] += 1
            elif recency <= 10:
                bins['warm']['count'] += 1
            else:
                bins['cold']['count'] += 1
        
        total = len(self.optimizer.number_pool)
        for key in bins:
            bins[key]['percentage'] = (bins[key]['count'] / total) * 100
            
        return bins

    def print_enhanced_results(self, results, historical):
        """Print validation results with recency and hot/cold stats"""
        print("\n" + "="*60)
        print("ENHANCED VALIDATION REPORT".center(60))
        print("="*60)
        
        if 'historical' in results:
            hist = results['historical']
            print(f"\nTested against {hist['draws_tested']} historical draws")
            
            # Number type analysis
            print("\nNUMBER TYPE STATISTICS:")
            types = hist['number_types']
            total_numbers = len(self.optimizer.number_pool)
            
            print(f"● Cold numbers: {len(types['cold_numbers'])}/{total_numbers} "
                  f"({len(types['cold_numbers'])/total_numbers*100:.1f}%)")
            print(f"🔥 Hot numbers: {len(types['hot_numbers'])}/{total_numbers} "
                  f"({len(types['hot_numbers'])/total_numbers*100:.1f}%)")
            print(f"♨️ Warm numbers: {len(types['warm_numbers'])}/{total_numbers} "
                  f"({len(types['warm_numbers'])/total_numbers*100:.1f}%)")
            print(f"✖ Never drawn: {len(types['never_drawn'])}/{total_numbers} "
                  f"({len(types['never_drawn'])/total_numbers*100:.1f}%)")
            
            # Recency distribution
            print("\nRECENCY DISTRIBUTION:")
            recency = hist['recency_distribution']
            print(f"🔥 Hot (≤3 draws): {recency['hot']['count']} numbers "
                  f"({recency['hot']['percentage']:.1f}%)")
            print(f"♨️ Warm (4-10 draws): {recency['warm']['count']} numbers "
                  f"({recency['warm']['percentage']:.1f}%)")
            print(f"❄️ Cold (>10 draws): {recency['cold']['count']} numbers "
                  f"({recency['cold']['percentage']:.1f}%)")
            print(f"✖ Never drawn: {recency['never']['count']} numbers "
                  f"({recency['never']['percentage']:.1f}%)")
            
            # Match performance by type
            print("\nPERFORMANCE BY NUMBER TYPE:")
            cold_hits = sum(1 for n in types['cold_numbers'] 
                          if n in hist['high_performance_sets'])
            hot_hits = sum(1 for n in types['hot_numbers'] 
                         if n in hist['high_performance_sets'])
            
            print(f"● Cold numbers hit rate: "
                  f"{cold_hits/max(1,len(types['cold_numbers']))*100:.1f}%")
            print(f"🔥 Hot numbers hit rate: "
                  f"{hot_hits/max(1,len(types['hot_numbers']))*100:.1f}%")

        if 'latest' in results:
            print("\nLATEST DRAW NUMBER STATS:")
            for num in results['latest']['draw_numbers']:
                stats = results['recency_stats'].get(num, {})
                status = "🔥 HOT" if stats.get('recency', 99) <= 3 else \
                         "♨️ WARM" if stats.get('recency', 99) <= 10 else \
                         "❄️ COLD" if num in self.optimizer.cold_numbers else "NEUTRAL"
                print(f"#{num}: {status} | "
                      f"Last drawn: {stats.get('recency', 'Never')} draws ago")

        if self.optimizer.last_generated_sets:
            print("\nRECOMMENDED SETS ANALYSIS:")
            for i, (nums, strategy) in enumerate(self.optimizer.last_generated_sets, 1):
                cold = [n for n in nums if n in self.optimizer.cold_numbers]
                hot = [n for n in nums if results['recency_stats'].get(n, {}).get('recency', 99) <= 3]
                print(f"Set {i}: {', '.join(str(n) for n in nums)}")
                print(f"   Strategy: {strategy} | "
                      f"● Cold: {len(cold)} | "
                      f"🔥 Hot: {len(hot)} | "
                      f"♨️ Warm: {len(nums)-len(cold)-len(hot)}")

        print("\n" + "="*60)
            
        except Exception as e:
            print(f"Error printing results: {str(e)}")


def run(self, mode):
    results = {}
    
    try:
        if mode in ('historical', 'both'):
            if self.optimizer.config['output']['verbose']:
                print("\nRUNNING ENHANCED VALIDATION...")
            
            historical_results = self.test_historical()
            results['historical'] = historical_results
            
            # Add recency analysis to historical results
            historical_results['number_types'] = self._analyze_number_types()
            
            improved_sets = self.optimizer.generate_improved_sets(historical_results)
            self.optimizer.last_generated_sets = improved_sets
            
            if mode == 'both':
                improved_results = self.test_historical(sets=improved_sets)
                results['improved'] = improved_results
             
        if mode in ('new_draw', 'both') and self.optimizer.upcoming is not None:
            results['new_draw'] = self.check_new_draws()
        
        if mode in ('latest', 'both') and self.optimizer.latest_draw is not None:
            results['latest'] = self.check_latest_draw()
        
        if self.optimizer.config['validation']['save_report']:
            self.save_report(results)
        
        # Print enhanced results
        self.print_enhanced_results(results)
        
        return results
    
    except Exception as e:
        print(f"Validation error: {str(e)}")
        return {}

def _analyze_number_types(self):
    """Analyze cold/hot/warm numbers"""
    num_cols = [f'n{i+1}' for i in range(self.optimizer.config['strategy']['numbers_to_select'])]
    last_draw_idx = len(self.optimizer.historical) - 1
    
    analysis = {
        'cold_numbers': list(self.optimizer.cold_numbers),
        'hot_numbers': [],
        'warm_numbers': []
    }
    
    for num in self.optimizer.number_pool:
        recency, _, _ = self._get_recency_info(num, self.optimizer.historical, num_cols)
        if recency is None:
            continue
        if recency <= 3:
            analysis['hot_numbers'].append(num)
        elif recency <= 10:
            analysis['warm_numbers'].append(num)
    
    return analysis

def print_enhanced_results(self, results):
    """Print results with recency stats"""
    print("\n" + "="*60)
    print("ENHANCED VALIDATION RESULTS".center(60))
    print("="*60)
    
    if 'historical' in results:
        hist = results['historical']
        print(f"\nTested against {hist['draws_tested']} historical draws")
        
        # Print number type stats
        print("\nNUMBER TYPE ANALYSIS:")
        print(f"● Cold numbers: {len(hist['number_types']['cold_numbers'])}")  # Fixed parenthesis
        print(f"🔥 Hot numbers: {len(hist['number_types']['hot_numbers'])}")   # Fixed parenthesis
        print(f"♨️ Warm numbers: {len(hist['number_types']['warm_numbers'])}") # Fixed parenthesis
        
        # Print match distribution with types
        print("\nMATCH DISTRIBUTION BY NUMBER TYPE:")
        for i in range(self.optimizer.config['strategy']['numbers_to_select'] + 1):
            print(f"{i} matches: {hist['match_counts'][i]} ({hist['match_percentages'][f'{i}_matches']})")
    
    if 'improved' in results:
        print("\nIMPROVEMENT AFTER ADAPTATION:")
        # ... existing improvement comparison ...
    
    if self.optimizer.last_generated_sets:
        print("\nRECOMMENDED SETS WITH RECENCY:")
        for i, (nums, strategy) in enumerate(self.optimizer.last_generated_sets, 1):
            cold = [n for n in nums if n in self.optimizer.cold_numbers]
            hot = [n for n in nums if n in results['historical']['number_types']['hot_numbers']]
            print(f"Set {i}: {', '.join(str(n) for n in nums)}")
            print(f"   Strategy: {strategy} | Cold: {len(cold)} | Hot: {len(hot)}")

def print_adaptive_results(self, results):
    """Enhanced print method with recency and temperature stats"""
    print("\n" + "="*60)
    print("ENHANCED VALIDATION REPORT".center(60))
    print("="*60)
    
    if 'historical' in results:
        self._print_number_performance(results['historical'])
    
    if 'historical' in results:
        print("\nMATCH DISTRIBUTION WITH NUMBER TYPES:")
        hist = results['historical']
        for i in range(self.optimizer.config['strategy']['numbers_to_select'] + 1):
            cold_pct = self._get_cold_match_percentage(i, hist)
            hot_pct = self._get_hot_match_percentage(i, hist)
            print(f"{i} matches: {hist['match_counts'][i]} ({hist['match_percentages'][f'{i}_matches']})")
            print(f"   Cold numbers: {cold_pct:.1f}% | Hot numbers: {hot_pct:.1f}%")
    
    if self.optimizer.last_generated_sets:
        print("\nRECOMMENDED SETS ANALYSIS:")
        for i, (nums, strategy) in enumerate(self.optimizer.last_generated_sets, 1):
            cold_count = sum(1 for n in nums if n in self.optimizer.cold_numbers)
            recency_stats = [self._get_recency_info(n, self.optimizer.historical, 
                           [f'n{i+1}' for i in range(self.optimizer.config['strategy']['numbers_to_select'])])
                           for n in nums]
            hot_count = sum(1 for r in recency_stats if r and r[0] <= 3)
            
            print(f"Set {i}: {', '.join(str(n) for n in nums)}")
            print(f"   Strategy: {strategy} | Cold: {cold_count} | Hot: {hot_count}")
            
            # Fixed line continuation for recency display:
            recency_info = []
            for n, r in zip(nums, recency_stats):
                if r:
                    recency_info.append(f"{n}:{r[0]}d")
                else:
                    recency_info.append(f"{n}:Never")
            print(f"   Recency: {', '.join(recency_info)}")

def _print_number_performance(self, hist_results):
    """Show performance by number type"""
    print("\nNUMBER TYPE PERFORMANCE:")
    test_draws = hist_results['draws_tested']
    
    # Get cold numbers
    cold_nums = self.optimizer.cold_numbers
    cold_hits = sum(hist_results['match_counts'].get(i, 0) 
                   for i in range(4, self.optimizer.config['strategy']['numbers_to_select'] + 1))
    
    # Get hot numbers (appeared in last 3 draws)
    hot_nums = set()
    last_3_draws = self.optimizer.historical.iloc[-3:].values.flatten()
    for num in self.optimizer.number_pool:
        if num in last_3_draws:
            hot_nums.add(num)
    
    print(f"Cold Numbers ({len(cold_nums)}): {cold_hits/test_draws*100:.1f}% high matches")
    print(f"Hot Numbers ({len(hot_nums)}): {sum(1 for n in hot_nums if n in hist_results['high_performance_sets'])/len(hot_nums)*100:.1f}% in winning sets")

def _get_cold_match_percentage(self, match_count, hist_results):
    """Calculate what % of matches involved cold numbers"""
    # Implementation depends on your match tracking
    # This is a simplified version
    total = hist_results['match_counts'].get(match_count, 1)
    cold_involved = sum(1 for s in hist_results['high_performance_sets'] 
                     if any(n in self.optimizer.cold_numbers for n in s))
    return (cold_involved / total) * 100

def _get_hot_match_percentage(self, match_count, hist_results):
    """Calculate what % of matches involved hot numbers"""
    last_3_draws = set(self.optimizer.historical.iloc[-3:].values.flatten())
    total = hist_results['match_counts'].get(match_count, 1)
    hot_involved = sum(1 for s in hist_results['high_performance_sets'] 
                   if any(n in last_3_draws for n in s))
    return (hot_involved / total) * 100

def parse_args():
    parser = argparse.ArgumentParser(description='Adaptive Lottery Number Optimizer')
    parser.add_argument('--validate-saved', metavar='PATH', 
                       help='Validate saved number sets from CSV file')
    parser.add_argument('--mode', choices=['historical', 'new_draw', 'both', 'none'],
                       help='Validation mode to run')

    parser.add_argument('--analyze-latest', action='store_true', help='Show detailed analysis of numbers in latest draw')
    parser.add_argument('--match-threshold', type=int, default=4,  # Temporary default
                       help='Minimum matches to show (default: 4)')
    parser.add_argument('--show-top', type=int, default=5,  # Temporary default
                       help='Number of high-matching draws to display (default: 5)')

    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    return parser.parse_args()

def main():
    print("🎰 ADAPTIVE LOTTERY OPTIMIZER")
    print("=============================")
    
    args = parse_args()
    
    try:
        optimizer = AdaptiveLotteryOptimizer()
        optimizer.args = args 
        if args.verbose:
            optimizer.config['output']['verbose'] = True

        if 'analysis' in optimizer.config:
            if not hasattr(args, 'match_threshold') or args.match_threshold == 4:  # Only override if using default
                args.match_threshold = optimizer.config['analysis'].get('default_match_threshold', 4)
            if not hasattr(args, 'show_top') or args.show_top == 5:
                args.show_top = optimizer.config['analysis'].get('default_show_top', 5)

        if args.analyze_latest:
            optimizer.validator.analyze_latest_draw() 
        # Handle saved sets validation
        if args.validate_saved:
            results = optimizer.validator.validate_saved_sets(args.validate_saved)
            if results:
                print(f"\nVALIDATION RESULTS (Last {results['test_draws']} draws)")
                print(f"Latest Draw: {results['latest_draw']['date']} - {results['latest_draw']['numbers']}")
                
                for i, res in enumerate(results['results'], 1):
                    print(f"\nSet {i}: {'-'.join(map(str, res['numbers']))} ({res['strategy']})")
                    print(f"Current Matches: {res['current_matches']}/6")
                    if res['current_matches'] > 0:
                        print(f"Matched Numbers: {res['matched_numbers']}")
                    
                    print("\nHistorical Performance:")
                    for num in res['numbers']:
                        print(f"  {num}: {res['historical_stats']['appearances'][num]} appearances "
                              f"({res['historical_stats']['percentages'][num]})")
                    
                    print(f"\nPrevious ≥{res['previous_performance']['alert_threshold']} Matches:")
                    if not res['previous_performance']['high_matches']:
                        print("  None found")
                    else:
                        for match in res['previous_performance']['high_matches']:
                            print(f"  {match['date']}: {match['matches']} matches - {match['numbers']}")
            return

        # Original workflow
        initial_sets = optimizer.generate_sets()
        
        if args.mode or optimizer.config['validation']['mode'] != 'none':
            optimizer.run_validation(args.mode)
            
        if optimizer.save_results(optimizer.last_generated_sets or initial_sets):
            print(f"\n✓ Results saved to '{optimizer.config['data']['results_dir']}/suggestions.csv'")
        
    except Exception as e:
        print(f"\n💥 Error: {str(e)}")
        traceback.print_exc()
        print("\nTROUBLESHOOTING:")
        print("1. Verify data files exist in data/ directory")
        print(f"2. Check number ranges (1-{optimizer.config['strategy']['number_pool']})")
        print("3. For saved sets validation, ensure CSV has 'numbers' column")

if __name__ == "__main__":
    main()