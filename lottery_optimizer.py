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

class AdaptiveLotteryValidator:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def _get_recency_info(self, num, historical, num_cols):
        """Calculate recency data for a number"""
        for i in range(len(historical)-1, -1, -1):
            if num in historical.iloc[i][num_cols].values:
                recency = len(historical) - 1 - i
                days_ago = (historical.iloc[-1]['date'] - historical.iloc[i]['date']).days
                return recency, days_ago, i
        return None, None, None

    def _get_recency_marker(self, recency):
        """Get visual indicator for recency"""
        if recency is None:
            return ""
        bins = self.optimizer.config['analysis'].get('recency_bins', {})
        if recency <= bins.get('hot', 3):
            return "🔥"
        elif recency <= bins.get('warm', 10):
            return "♨️"
        elif recency <= bins.get('cold', 30):
            return "❄️"
        return ""

    def _show_combined_stats(self, numbers, historical):
        """Display unified cold + recency statistics"""
        num_cols = [f'n{i+1}' for i in range(self.optimizer.config['strategy']['numbers_to_select'])]
        cold_numbers = self.optimizer.cold_numbers
        
        # Prepare table data
        table_data = []
        for num in sorted(numbers):
            recency, days_ago, _ = self._get_recency_info(num, historical, num_cols)
            appearances = sum(historical[col].eq(num).sum() for col in num_cols)
            
            # Determine status
            if num in cold_numbers:
                status = "COLD"
            elif recency is None:
                status = "NEVER"
            elif recency <= self.optimizer.config['analysis']['recency_bins']['hot']:
                status = "HOT"
            elif recency <= self.optimizer.config['analysis']['recency_bins']['warm']:
                status = "WARM"
            else:
                status = "DORMANT"
                
            table_data.append({
                'num': num,
                'cold': num in cold_numbers,
                'appearances': appearances,
                'recency': recency,
                'days_ago': days_ago,
                'status': status,
                'marker': self._get_recency_marker(recency)
            })

        # Print formatted table
        print("\n{:^60}".format("COMBINED NUMBER ANALYSIS"))
        print("{:<5} {:<8} {:<12} {:<15} {:<10} {:<10}".format(
            "NUM", "APPEAR", "FREQ", "LAST DRAWN", "STATUS", "RECENCY"
        ))
        print("-" * 70)
        
        for data in table_data:
            recency_str = f"{data['recency']} draws" if data['recency'] is not None else "Never"
            if self.optimizer.config['analysis'].get('recency_units') == 'days' and data['days_ago'] is not None:
                recency_str = f"{data['days_ago']} days"
                
            print("{:<2}{:<3} {:<3}/{:<3} {:<6.2f}% {:<15} {:<10} {:<3}".format(
                "●" if data['cold'] else "",
                data['num'],
                data['appearances'],
                len(historical),
                (data['appearances']/len(historical))*100,
                recency_str,
                data['status'],
                data['marker']
            ))

        # Summary statistics
        hot_nums = [d['num'] for d in table_data if d['status'] == "HOT"]
        cold_nums = [d['num'] for d in table_data if d['cold']]
        
        print("\nSUMMARY:")
        print(f"- Cold numbers: {len(cold_nums)}/{len(numbers)}")
        print(f"- Hot numbers: {len(hot_nums)}/{len(numbers)}")
        print(f"- Numbers both cold and hot: {len(set(cold_nums) & set(hot_nums))}")

    def analyze_latest_draw(self):
        config = self.optimizer.config
        try:
            test_draws = min(
                config['validation']['test_draws'],
                len(self.optimizer.historical)
            )
            historical = self.optimizer.historical.iloc[-test_draws:]
            
            if not hasattr(self.optimizer, 'latest_draw') or self.optimizer.latest_draw is None:
                print("No latest draw available for analysis")
                return

            num_select = config['strategy']['numbers_to_select']
            latest_numbers = [int(self.optimizer.latest_draw[f'n{i+1}']) for i in range(num_select)]
            
            print(f"\nLatest Draw: {self.optimizer.latest_draw['date'].strftime('%Y-%m-%d')}")
            print(f"Numbers: {latest_numbers}")
            print(f"Analyzing against: {len(historical)}/{len(self.optimizer.historical)} draws")
            
            # Show combined analysis
            self._show_combined_stats(latest_numbers, historical)
            
            # Show matching historical draws
            self._show_historical_matches(latest_numbers, historical, config)

        except KeyError as e:
            print(f"Configuration error: {str(e)}")

    def _show_number_stats(self, numbers, historical):
        """Show statistics for each number"""
        num_cols = [f'n{i+1}' for i in range(self.optimizer.config['strategy']['numbers_to_select'])]
        total_draws = len(historical)
        
        print("\nIndividual Number Statistics:")
        for num in sorted(numbers):
            appearances = sum(historical[col].eq(num).sum() for col in num_cols)
            percentage = (appearances / total_draws) * 100
            print(f"#{num}: {appearances}/{total_draws} appearances ({percentage:.2f}%)")

    def _show_historical_matches(self, latest_numbers, historical, config):
        """Show matching historical draws"""
        num_cols = [f'n{i+1}' for i in range(self.optimizer.config['strategy']['numbers_to_select'])]
        latest_set = set(latest_numbers)
        threshold = config['validation'].get('alert_threshold', 4)
        
        matches = []
        for _, row in historical.iterrows():
            draw_numbers = {int(row[col]) for col in num_cols}
            match_count = len(latest_set & draw_numbers)
            if match_count >= 1:
                matches.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'numbers': sorted(draw_numbers),
                    'matches': match_count
                })

        print(f"\nFound {len(matches)} matching draws (showing ≥{threshold} matches):")
        for match in sorted(matches, key=lambda x: (-x['matches'], x['date'])):
            if match['matches'] >= threshold:
                print(f"{match['date']}: {match['matches']} matches - {match['numbers']}")

    def check_latest_draw(self):
        """Validate against the latest draw"""
        if self.optimizer.latest_draw is None:
            if self.optimizer.config['output']['verbose']:
                print("\nNo latest draw found - skipping validation.")
            return None

        num_select = self.optimizer.config['strategy']['numbers_to_select']
        target = set(self.optimizer.latest_draw[[f'n{i+1}' for i in range(num_select)]])
        
        results = {
            'draw_date': self.optimizer.latest_draw['date'].strftime('%m/%d/%y'),
            'draw_numbers': sorted([int(n) for n in target]),
            'sets': []
        }

        for generated_set, strategy in (self.optimizer.last_generated_sets or self.optimizer.generate_sets()):
            matches = len(set(generated_set) & target)
            results['sets'].append({
                'numbers': [int(n) for n in generated_set],
                'strategy': strategy,
                'matches': matches,
                'matched_numbers': sorted([int(n) for n in set(generated_set) & target])
            })

        if self.optimizer.config['output']['verbose']:
            print("\nLATEST DRAW VALIDATION:")
            print(f"Draw: {results['draw_date']} - {results['draw_numbers']}")
            for i, set_result in enumerate(results['sets'], 1):
                print(f"Set {i}: {set_result['matches']} matches - {set_result['matched_numbers']} ({set_result['strategy']})")

        return results

    def validate_saved_sets(self, file_path):
        """Validate saved sets against historical data"""
        try:
            test_draws = min(
                self.optimizer.config['validation']['test_draws'],
                len(self.optimizer.historical)
            )
            alert_threshold = self.optimizer.config['validation']['alert_threshold']
            num_select = self.optimizer.config['strategy']['numbers_to_select']
            num_cols = [f'n{i+1}' for i in range(num_select)]

            # Load saved sets
            df = pd.read_csv(file_path)
            sets = []
            for _, row in df.iterrows():
                if 'numbers' in df.columns:
                    numbers = [int(n) for n in str(row['numbers']).split('-')]
                    strategy = str(row.get('strategy', 'unknown'))
                else:
                    numbers = [int(n) for n in str(row.iloc[0]).split('-')]
                    strategy = 'unknown'
                sets.append((numbers, strategy))

            if not sets:
                raise ValueError("No valid sets found in file")

            # Prepare test data
            test_data = self.optimizer.historical.iloc[-test_draws:]
            test_numbers = test_data[num_cols].values.flatten()
            test_freq = pd.Series(test_numbers).value_counts()

            results = []
            for numbers, strategy in sets:
                current_matches = sorted([n for n in numbers if n in set(self.optimizer.latest_draw[num_cols])])
                
                hist_counts = {num: int(test_freq.get(num, 0)) for num in numbers}
                hist_percent = {num: f"{(count/test_draws)*100:.1f}%" for num, count in hist_counts.items()}
                
                high_matches = []
                for _, prev_draw in test_data.iterrows():
                    prev_nums = {int(n) for n in prev_draw[num_cols]}
                    matches = len(set(numbers) & prev_nums)
                    if matches >= alert_threshold:
                        high_matches.append({
                            'date': prev_draw['date'].strftime('%Y-%m-%d'),
                            'numbers': sorted(prev_nums),
                            'matches': int(matches)
                        })

                results.append({
                    'numbers': numbers,
                    'strategy': strategy,
                    'current_matches': len(current_matches),
                    'matched_numbers': current_matches,
                    'historical_stats': {
                        'appearances': hist_counts,
                        'percentages': hist_percent,
                        'test_draws': test_draws
                    },
                    'previous_performance': {
                        'high_matches': high_matches,
                        'alert_threshold': alert_threshold
                    }
                })

            return {
                'latest_draw': {
                    'date': self.optimizer.latest_draw['date'].strftime('%Y-%m-%d'),
                    'numbers': sorted([int(n) for n in self.optimizer.latest_draw[num_cols]])
                },
                'test_draws': test_draws,
                'results': results
            }

        except Exception as e:
            print(f"\nERROR VALIDATING SAVED SETS: {str(e)}")
            print("Expected CSV format:")
            print("1. 'numbers' column (e.g., '1-2-3-4-5-6')")
            print("2. Optional 'strategy' column")
            return None

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
            print(f"● Cold numbers: {len(hist['number_types']['cold_numbers'])}")
            print(f"🔥 Hot numbers: {len(hist['number_types']['hot_numbers'])}")
            print(f"♨️ Warm numbers: {len(hist['number_types']['warm_numbers'])}")
            
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


        def test_historical(self, sets=None):
            """Test against historical draws"""
            num_select = self.optimizer.config['strategy']['numbers_to_select']
            test_draws = min(
                self.optimizer.config['validation']['test_draws'],
                len(self.optimizer.historical)-1
            )
            test_data = self.optimizer.historical.iloc[-test_draws-1:-1]
            
            stats = {
                'draws_tested': len(test_data),
                'match_counts': {i:0 for i in range(num_select + 1)},
                'best_per_draw': [],
                'high_performance_sets': []
            }
            
            sets_to_test = sets if sets else self.optimizer.last_generated_sets or self.optimizer.generate_sets()
            
            for _, draw in test_data.iterrows():
                target = set(draw[[f'n{i+1}' for i in range(num_select)]])
                best_match = 0
                
                for generated_set, _ in sets_to_test:
                    matches = len(set(generated_set) & target)
                    stats['match_counts'][matches] += 1
                    best_match = max(best_match, matches)
                    
                    if matches >= self.optimizer.config['validation']['alert_threshold']:
                        stats['high_performance_sets'].append(generated_set)
                
                stats['best_per_draw'].append(best_match)
            
            total_comparisons = len(sets_to_test) * len(test_data)
            stats['match_percentages'] = {
                f'{i}_matches': f"{(count/total_comparisons)*100:.2f}%"
                for i, count in stats['match_counts'].items()
            }
            
            if self.optimizer.config['output']['verbose']:
                print("\nVALIDATION RESULTS:")
                print(f"Tested against {len(test_data)} historical draws")
                print("Match distribution:")
                for i in range(num_select + 1):
                    print(f"{i} matches: {stats['match_counts'][i]} ({stats['match_percentages'][f'{i}_matches']})")
                print(f"\nBest match per draw: {collections.Counter(stats['best_per_draw'])}")
            
            return stats

        def check_new_draws(self):
            """Check against upcoming draws"""
            num_select = self.optimizer.config['strategy']['numbers_to_select']
            results = {
                'draws_tested': len(self.optimizer.upcoming),
                'matches': [],
                'detailed_comparisons': []
            }
            
            for _, draw in self.optimizer.upcoming.iterrows():
                target = set(draw[[f'n{i+1}' for i in range(num_select)]])
                draw_comparison = {
                    'draw_numbers': sorted([int(n) for n in target]),
                    'sets': []
                }
                
                best_match = 0
                for generated_set, strategy in (self.optimizer.last_generated_sets or self.optimizer.generate_sets()):
                    matches = len(set(generated_set) & target)
                    draw_comparison['sets'].append({
                        'numbers': [int(n) for n in generated_set],
                        'strategy': strategy,
                        'matches': matches,
                        'matched_numbers': sorted([int(n) for n in set(generated_set) & target])
                    })
                    best_match = max(best_match, matches)
                
                results['matches'].append(best_match)
                results['detailed_comparisons'].append(draw_comparison)
            
            results['match_distribution'] = dict(collections.Counter(results['matches']))
            
            if self.optimizer.config['output']['verbose']:
                print("\nUPCOMING DRAW PREDICTIONS:")
                print(f"Best matches against {len(results['matches'])} upcoming draws:")
                print(f"Match counts: {results['match_distribution']}")
                
                if results['detailed_comparisons']:
                    first_draw = results['detailed_comparisons'][0]
                    print("\nDetailed comparison for first upcoming draw:")
                    print(f"Draw numbers: {first_draw['draw_numbers']}")
                    for i, set_comp in enumerate(first_draw['sets'], 1):
                        print(f"Set {i}: {set_comp['matches']} matches - {set_comp['matched_numbers']} ({set_comp['strategy']})")
            
            return results

        def save_report(self, results):
            """Save validation report"""
            try:
                report_file = Path(self.optimizer.config['data']['stats_dir']) / 'validation_report.json'
                
                with open(report_file, 'w') as f:
                    json.dump(self._convert_results(results), f, indent=2)
                    
                if self.optimizer.config['output']['verbose']:
                    print(f"\nSAVED VALIDATION REPORT TO: {report_file}")
                return True
            except Exception as e:
                print(f"Error saving validation report: {str(e)}")
                return False

        def _convert_results(self, results):
            """Convert results to JSON-serializable format"""
            if isinstance(results, dict):
                return {k: self._convert_results(v) for k, v in results.items()}
            elif isinstance(results, list):
                return [self._convert_results(item) for item in results]
            elif isinstance(results, np.integer):
                return int(results)
            elif isinstance(results, np.floating):
                return float(results)
            elif isinstance(results, np.ndarray):
                return results.tolist()
            return results

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