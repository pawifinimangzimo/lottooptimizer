import yaml
import pandas as pd
import numpy as np
import random
import sympy
from pathlib import Path
import json
from datetime import datetime
import argparse
from collections import defaultdict
from scipy.stats import chisquare
import traceback

class AdaptiveLotteryOptimizer:
    def __init__(self, config_path="config.yaml"):
        self.config = None
        self.historical = None
        self.upcoming = None
        self.number_pool = list(range(1, 56))
        self.decay_factor = 0.97
        self.prime_numbers = [n for n in range(1,56) if sympy.isprime(n)]
        self.frequencies = None
        self.recent_counts = None
        self.cold_numbers = set()
        self.overrepresented_pairs = set()
        self.weights = None
        self.last_generated_sets = None
        self.high_performance_numbers = set()
        
        self.load_config(config_path)
        self.prepare_filesystem()
        self.load_and_clean_data()
        self.validate_data()
        self.analyze_numbers()
        self.validator = AdaptiveLotteryValidator(self)

    def load_config(self, config_path):
        """Load configuration from YAML file with enhanced error handling"""
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            defaults = {
                'data': {
                    'historical_path': 'data/historical.csv',
                    'upcoming_path': 'data/upcoming.csv',
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
            
            self.number_pool = list(range(1, self.config['strategy']['number_pool'] + 1))
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            print("Config file should be in YAML format with proper indentation")
            raise

    def prepare_filesystem(self):
        """Ensure required directories exist"""
        try:
            Path(self.config['data']['stats_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['data']['results_dir']).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise

    def load_and_clean_data(self):
        """Load and clean historical lottery data with detailed error reporting"""
        try:
            # Load historical data with error context
            hist_path = self.config['data']['historical_path']
            print(f"Loading historical data from: {hist_path}")
            
            try:
                self.historical = pd.read_csv(
                    hist_path, 
                    header=None, 
                    names=['date', 'numbers'],
                    dtype={'date': str, 'numbers': str}
                )
            except pd.errors.EmptyDataError:
                raise ValueError("Historical data file is empty")
            except pd.errors.ParserError:
                raise ValueError("Historical data file has formatting issues")

            # Validate we have data before processing
            if len(self.historical) == 0:
                raise ValueError("No data found in historical file")

            # Split numbers into columns with error tracking
            num_cols = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
            try:
                number_split = self.historical['numbers'].str.split('-', expand=True)
                if number_split.shape[1] != 6:
                    raise ValueError(f"Expected 6 numbers per draw, found {number_split.shape[1]}")
                
                self.historical[num_cols] = number_split.astype(int)
            except ValueError as e:
                # Find the problematic rows
                bad_rows = []
                for idx, row in self.historical.iterrows():
                    nums = row['numbers'].split('-')
                    if len(nums) != 6:
                        bad_rows.append((idx, row['date'], row['numbers']))
                    try:
                        [int(n) for n in nums]
                    except ValueError:
                        bad_rows.append((idx, row['date'], row['numbers']))
                
                if bad_rows:
                    print("\nProblematic rows in historical data:")
                    for idx, date, nums in bad_rows[:5]:  # Show first 5 bad rows
                        print(f"Row {idx}: Date={date}, Numbers={nums}")
                    raise ValueError(f"Invalid number format in {len(bad_rows)} rows. First few shown above.")
                raise

            # Convert date field with validation
            try:
                self.historical['date'] = pd.to_datetime(
                    self.historical['date'], 
                    format='%m/%d/%y',
                    errors='raise'
                )
            except ValueError as e:
                bad_dates = self.historical[self.historical['date'].isna()]
                print("\nInvalid date formats found:")
                print(bad_dates[['date']].head())
                raise ValueError(f"Date format should be MM/DD/YY. {len(bad_dates)} invalid dates found.")

            # Load upcoming data if configured
            if self.config['data']['upcoming_path']:
                try:
                    self.upcoming = pd.read_csv(
                        self.config['data']['upcoming_path'],
                        header=None,
                        names=['date', 'numbers'],
                        dtype={'date': str, 'numbers': str}
                    )
                    
                    # Validate upcoming data format
                    if not self.upcoming.empty:
                        upcoming_split = self.upcoming['numbers'].str.split('-', expand=True)
                        if upcoming_split.shape[1] != 6:
                            raise ValueError("Upcoming data must have exactly 6 numbers per draw")
                        
                        self.upcoming[num_cols] = upcoming_split.astype(int)
                        self.upcoming['date'] = pd.to_datetime(self.upcoming['date'], format='%m/%d/%y')
                        
                        if self.config['data']['merge_upcoming']:
                            self.historical = pd.concat([self.historical, self.upcoming])
                            
                except FileNotFoundError:
                    if self.config['output']['verbose']:
                        print("Note: Upcoming draws file not found")
                except Exception as e:
                    print(f"Warning: Error processing upcoming draws - {str(e)}")
                    self.upcoming = None
                    
        except Exception as e:
            print("\nDATA LOADING ERROR DETAILS:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nPlease ensure your data files:")
            print("- Are in the correct location (data/historical.csv)")
            print("- Have exactly 6 numbers per draw (N1-N2-N3-N4-N5-N6)")
            print("- Use MM/DD/YY date format")
            print("- Contain no header row")
            raise

    def validate_data(self):
        """Validate loaded data with detailed checks"""
        required_columns = ['n1', 'n2', 'n3', 'n4', 'n5', 'n6']
        
        # Create list of dataframes to validate
        data_to_validate = [(self.historical, 'historical')]
        if self.upcoming is not None and not self.upcoming.empty:
            data_to_validate.append((self.upcoming, 'upcoming'))
        
        for df, name in data_to_validate:
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns in {name} data: {missing}")
                
            # Check number ranges
            invalid_numbers = []
            for col in required_columns:
                invalid = df[
                    (df[col] < 1) | 
                    (df[col] > self.config['strategy']['number_pool'])
                ]
                if not invalid.empty:
                    invalid_numbers.append((col, invalid[[col, 'date']].values))
            
            if invalid_numbers:
                print("\nINVALID NUMBERS FOUND:")
                for col, bad_values in invalid_numbers[:3]:  # Show first 3 bad entries per column
                    print(f"Column {col}:")
                    for val, date in bad_values[:3]:
                        print(f"  Draw {date}: Value {val} (valid range 1-{self.config['strategy']['number_pool']})")
                raise ValueError(f"Invalid numbers detected in {name} data (some shown above)")

    def analyze_numbers(self):
        """Analyze number frequencies and patterns with type safety"""
        try:
            numbers = self.historical[['n1','n2','n3','n4','n5','n6']].values.flatten()
            self.frequencies = pd.Series(numbers).value_counts().sort_index()
            
            # Convert frequencies to native Python types
            self.frequencies = self.frequencies.astype(int)
            
            recent_draws = self.historical.iloc[-int(len(self.historical)*0.2):]
            recent_numbers = recent_draws[['n1','n2','n3','n4','n5','n6']].values.flatten()
            self.recent_counts = pd.Series(recent_numbers).value_counts().reindex(
                range(1, self.config['strategy']['number_pool']+1), 
                fill_value=0
            ).astype(int)
            
            last_20_draws = self.historical.iloc[-20:][['n1','n2','n3','n4','n5','n6']].values.flatten()
            self.cold_numbers = set(range(1, self.config['strategy']['number_pool']+1)) - set(last_20_draws)
            
            self._find_overrepresented_pairs()
            self.calculate_weights()
            
        except Exception as e:
            print(f"Error during number analysis: {str(e)}")
            raise

    def _find_overrepresented_pairs(self):
        """Identify number pairs that appear together more than expected"""
        try:
            pair_counts = defaultdict(int)
            for _, row in self.historical.iterrows():
                nums = sorted(row[['n1','n2','n3','n4','n5','n6']])
                for i in range(len(nums)):
                    for j in range(i+1, len(nums)):
                        pair_counts[(nums[i], nums[j])] += 1
            
            expected = len(self.historical) * (6*5)/(self.config['strategy']['number_pool']*(self.config['strategy']['number_pool']-1))
            threshold = expected * 1.5
            self.overrepresented_pairs = {pair for pair, count in pair_counts.items() if count > threshold}
            
        except Exception as e:
            print(f"Error finding number pairs: {str(e)}")
            self.overrepresented_pairs = set()

    def calculate_weights(self):
        """Calculate dynamic selection weights with type safety"""
        try:
            base_weights = pd.Series(1.0, index=range(1, self.config['strategy']['number_pool']+1))
            
            # Ensure all weights are native Python floats
            freq_weights = (self.frequencies / self.frequencies.sum()).astype(float)
            base_weights += freq_weights * self.config['strategy']['frequency_weight'] * 10
            
            recent_weights = (self.recent_counts / self.recent_counts.sum()).astype(float)
            base_weights += recent_weights * self.config['strategy']['recent_weight'] * 5
            
            for num in self.high_performance_numbers:
                base_weights[num] *= 1.5
                
            for n1, n2 in self.overrepresented_pairs:
                base_weights[n1] *= 0.9
                base_weights[n2] *= 0.9
                
            for num in self.cold_numbers:
                base_weights[num] *= float(np.random.uniform(1.1, 1.3))
                
            hot_numbers = set(self.frequencies.nlargest(10).index)
            for num in hot_numbers:
                base_weights[num] *= float(np.random.uniform(0.85, 0.95))
            
            random_weights = pd.Series(
                np.random.dirichlet(np.ones(self.config['strategy']['number_pool'])) * 0.7,
                index=range(1, self.config['strategy']['number_pool']+1)
            ).astype(float)
            base_weights += random_weights * self.config['strategy']['random_weight'] * 15
            
            self.weights = (base_weights / base_weights.sum()).astype(float)
            
        except Exception as e:
            print(f"Error calculating weights: {str(e)}")
            # Fallback to uniform weights
            self.weights = pd.Series(
                1.0/self.config['strategy']['number_pool'],
                index=range(1, self.config['strategy']['number_pool']+1)
            )

    def generate_sets(self):
        """Generate number sets using different strategies"""
        try:
            sets = []
            strategies = [
                ('weighted_random', self._generate_weighted_random),
                ('high_low_mix', self._generate_high_low_mix),
                ('prime_balanced', self._generate_prime_balanced),
                ('performance_boosted', self._generate_performance_boosted)
            ]
            
            sets_per_strategy = max(1, self.config['output']['sets_to_generate'] // len(strategies))
            
            for name, strategy in strategies:
                for _ in range(sets_per_strategy):
                    try:
                        numbers = strategy()
                        if len(numbers) != 6 or len(set(numbers)) != 6:
                            raise ValueError(f"Invalid set generated by {name}: {numbers}")
                        sets.append((numbers, name))
                    except Exception as e:
                        print(f"Warning: Error in {name} strategy - {str(e)}")
                        continue
            
            self.last_generated_sets = sets
            return sets
            
        except Exception as e:
            print(f"Error generating number sets: {str(e)}")
            # Return some safe fallback sets
            return [
                ([1, 2, 3, 4, 5, 6], 'fallback'),
                ([7, 8, 9, 10, 11, 12], 'fallback')
            ]

    def _generate_weighted_random(self):
        """Weighted random selection with validation"""
        numbers = np.random.choice(
            self.number_pool,
            size=6,
            replace=False,
            p=self.weights
        )
        return sorted([int(n) for n in numbers])

    def _generate_high_low_mix(self):
        """Mix of high and low numbers with validation"""
        low_max = self.config['strategy']['low_number_max']
        low_nums = [n for n in self.number_pool if n <= low_max]
        high_nums = [n for n in self.number_pool if n > low_max]
        
        low_weights = self.weights[low_nums] / self.weights[low_nums].sum()
        high_weights = self.weights[high_nums] / self.weights[high_nums].sum()
        
        selected = (
            list(np.random.choice(low_nums, 3, replace=False, p=low_weights)) +
            list(np.random.choice(high_nums, 3, replace=False, p=high_weights))
        )
        return sorted([int(n) for n in selected])

    def _generate_prime_balanced(self):
        """Balanced prime/non-prime mix with validation"""
        primes = self.prime_numbers
        non_primes = [n for n in self.number_pool if n not in primes]
        
        prime_weights = self.weights[primes] / self.weights[primes].sum()
        non_prime_weights = self.weights[non_primes] / self.weights[non_primes].sum()
        
        num_primes = np.random.choice([2,3,4])
        selected = (
            list(np.random.choice(primes, num_primes, replace=False, p=prime_weights)) +
            list(np.random.choice(non_primes, 6-num_primes, replace=False, p=non_prime_weights))
        )
        return sorted([int(n) for n in selected])

    def _generate_performance_boosted(self):
        """Favor high-performance numbers with validation"""
        if not self.high_performance_numbers:
            return self._generate_weighted_random()
            
        boosted_weights = self.weights.copy()
        for num in self.high_performance_numbers:
            boosted_weights[num] *= 2.0
            
        boosted_weights /= boosted_weights.sum()
        numbers = np.random.choice(
            self.number_pool,
            size=6,
            replace=False,
            p=boosted_weights
        )
        return sorted([int(n) for n in numbers])

    def generate_improved_sets(self, previous_results):
        """Generate improved sets based on validation results"""
        try:
            if '4_match_sets' in previous_results:
                for nums in previous_results['4_match_sets']:
                    self.high_performance_numbers.update(nums)
            self.calculate_weights()
            return self.generate_sets()
        except Exception as e:
            print(f"Error generating improved sets: {str(e)}")
            return self.generate_sets()

    def run_validation(self, mode=None):
        """Run validation process with error handling"""
        try:
            return self.validator.run(mode or self.config['validation']['mode'])
        except Exception as e:
            print(f"Validation error: {str(e)}")
            return {}

    def save_results(self, sets):
        """Save generated sets to CSV with validation"""
        try:
            output_file = Path(self.config['data']['results_dir']) / 'suggestions.csv'
            
            # Validate sets before saving
            valid_sets = []
            for nums, strategy in sets:
                if (len(nums) == 6 and 
                    len(set(nums)) == 6 and 
                    all(1 <= n <= self.config['strategy']['number_pool'] for n in nums)):
                    valid_sets.append((nums, strategy))
                else:
                    print(f"Discarding invalid set: {nums} (strategy: {strategy})")
            
            if not valid_sets:
                raise ValueError("No valid sets to save")
            
            with open(output_file, 'w') as f:
                f.write("numbers,strategy\n")
                for nums, strategy in valid_sets:
                    f.write(f"{'-'.join(map(str, nums))},{strategy}\n")
                    
            return True
        except Exception as e:
            print(f"Error saving results: {str(e)}")
            return False

class AdaptiveLotteryValidator:
    def __init__(self, optimizer):
        self.optimizer = optimizer
    
    def run(self, mode):
        results = {}
        
        try:
            if mode in ('historical', 'both'):
                initial_results = self.test_historical()
                results['initial'] = self._convert_results(initial_results)
                
                improved_sets = self.optimizer.generate_improved_sets(initial_results)
                self.optimizer.last_generated_sets = improved_sets
                
                improved_results = self.test_historical(sets=improved_sets)
                results['improved'] = self._convert_results(improved_results)
                
                if self.optimizer.config['output']['verbose']:
                    self.print_adaptive_results(results)
            
            if mode in ('new_draw', 'both') and self.optimizer.upcoming is not None and not self.optimizer.upcoming.empty:
                results['new_draw'] = self._convert_results(self.check_new_draws())
            
            if self.optimizer.config['validation']['save_report']:
                self.save_report(results)
            
            return results
        
        except Exception as e:
            print(f"Validation process error: {str(e)}")
            return {}

    def _convert_results(self, results):
        """Convert numpy types to native Python types for JSON serialization"""
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

    def test_historical(self, sets=None):
        """Test against historical draws with detailed stats"""
        try:
            test_draws = min(self.optimizer.config['validation']['test_draws'], len(self.optimizer.historical)-1)
            test_data = self.optimizer.historical.iloc[-test_draws-1:-1]
            
            if test_data.empty:
                raise ValueError("Not enough historical data for validation")
            
            stats = {
                'draws_tested': len(test_data),
                'sets_tested': 0,
                '6_matches': 0,
                '5_matches': 0,
                '4_matches': 0,
                '3_matches': 0,
                '4_match_sets': [],
                '5_match_sets': [],
                '6_match_sets': [],
                'match_breakdown': defaultdict(int),
                'low_num_hits': 0,
                'prime_hits': 0,
                'best_matches_per_draw': []
            }
            
            sets_to_test = sets if sets else self.optimizer.generate_sets()
            stats['sets_tested'] = len(sets_to_test) * len(test_data)
            
            for _, draw in test_data.iterrows():
                try:
                    target = set(draw[['n1','n2','n3','n4','n5','n6']])
                    best_match = 0
                    
                    for generated_set, _ in sets_to_test:
                        matches = len(set(generated_set) & target)
                        stats['match_breakdown'][matches] += 1
                        best_match = max(best_match, matches)
                        
                        if matches >= 4:
                            stats[f'{matches}_match_sets'].append([int(n) for n in generated_set])
                        
                        if any(n <= self.optimizer.config['strategy']['low_number_max'] for n in target):
                            stats['low_num_hits'] += 1
                        if any(n > self.optimizer.config['strategy']['high_prime_min'] and n in self.optimizer.prime_numbers for n in target):
                            stats['prime_hits'] += 1
                    
                    stats['best_matches_per_draw'].append(int(best_match))
                    if best_match == 6:
                        stats['6_matches'] += 1
                    elif best_match == 5:
                        stats['5_matches'] += 1
                    elif best_match == 4:
                        stats['4_matches'] += 1
                    elif best_match == 3:
                        stats['3_matches'] += 1
                
                except Exception as e:
                    print(f"Error processing draw {draw['date']}: {str(e)}")
                    continue
            
            stats['low_num_rate'] = float(stats['low_num_hits'] / stats['sets_tested']) if stats['sets_tested'] > 0 else 0.0
            stats['prime_rate'] = float(stats['prime_hits'] / stats['sets_tested']) if stats['sets_tested'] > 0 else 0.0
            
            # Convert defaultdict to regular dict
            stats['match_breakdown'] = dict(stats['match_breakdown'])
            
            return stats
            
        except Exception as e:
            print(f"Historical test error: {str(e)}")
            return {
                'error': str(e),
                'draws_tested': 0,
                'sets_tested': 0
            }

    def check_new_draws(self):
        """Check against upcoming draws with validation"""
        try:
            results = {
                'draws_tested': len(self.optimizer.upcoming),
                'matches': []
            }
            
            for _, draw in self.optimizer.upcoming.iterrows():
                try:
                    target = set(draw[['n1','n2','n3','n4','n5','n6']])
                    best_match = max(
                        len(set(generated_set) & target)
                        for generated_set, _ in self.optimizer.last_generated_sets
                    )
                    results['matches'].append(int(best_match))
                except Exception as e:
                    print(f"Error processing upcoming draw {draw['date']}: {str(e)}")
                    results['matches'].append(0)
            
            return results
            
        except Exception as e:
            print(f"New draws check error: {str(e)}")
            return {
                'error': str(e),
                'draws_tested': 0,
                'matches': []
            }

    def save_report(self, results):
        """Save validation report to JSON with error handling"""
        try:
            report_file = Path(self.optimizer.config['data']['stats_dir']) / 'validation_report.json'
            
            # Ensure results are JSON-serializable
            serializable_results = self._convert_results(results)
            
            with open(report_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving validation report: {str(e)}")
            return False

    def print_adaptive_results(self, results):
        """Print validation results with formatting"""
        try:
            print("\n" + "="*60)
            print("ADAPTIVE LOTTERY OPTIMIZATION REPORT".center(60))
            print("="*60)
            
            if 'initial' in results and 'improved' in results:
                init = results['initial']
                impr = results['improved']
                
                print("\nPERFORMANCE IMPROVEMENT:")
                print(f"4+ Match Rate: {init['4_matches']/init['draws_tested']:.1%} → "
                      f"{impr['4_matches']/impr['draws_tested']:.1%}")
                
                hp_nums = sorted(self.optimizer.high_performance_numbers)
                print(f"\nHIGH-PERFORMANCE NUMBERS ({len(hp_nums)}):")
                print(", ".join(map(str, hp_nums)))
            
            if self.optimizer.last_generated_sets:
                print("\nIMPROVED NUMBER SETS:")
                for i, (nums, strategy) in enumerate(self.optimizer.last_generated_sets, 1):
                    print(f"Set {i}: {'-'.join(map(str, nums))}  <-- {strategy}")
            
            if 'new_draw' in results:
                print("\nUPCOMING DRAW PREDICTIONS:")
                matches = results['new_draw']['matches']
                print(f"Best matches in last {len(matches)} upcoming draws: {matches}")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"Error printing results: {str(e)}")

def parse_args():
    """Parse command line arguments with error handling"""
    parser = argparse.ArgumentParser(description='Adaptive Lottery Number Optimizer')
    parser.add_argument('--mode', choices=['historical', 'new_draw', 'both', 'none'],
                       help='Validation mode to run')
    try:
        return parser.parse_args()
    except Exception as e:
        print(f"Argument parsing error: {str(e)}")
        return argparse.Namespace(mode=None)

def main():
    print("🎰 ADAPTIVE LOTTERY OPTIMIZER")
    print("=============================")
    
    args = parse_args()
    
    try:
        optimizer = AdaptiveLotteryOptimizer()
        
        # Initial generation
        initial_sets = optimizer.generate_sets()
        print("\nINITIAL NUMBER SETS:")
        for i, (nums, strategy) in enumerate(initial_sets, 1):
            print(f"Set {i:>2}: {'-'.join(map(str, nums))}  <-- {strategy}")
        
        # Run validation if configured
        if args.mode or optimizer.config['validation']['mode'] != 'none':
            results = optimizer.run_validation(args.mode)
        
        # Save final improved sets
        if optimizer.save_results(optimizer.last_generated_sets):
            print(f"\n✓ Final optimized sets saved to '{optimizer.config['data']['results_dir']}/suggestions.csv'")
        
    except Exception as e:
        print(f"\n💥 Critical Error: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        print("\nTROUBLESHOOTING:")
        print("1. Check your historical.csv file exists in the data/ directory")
        print("2. Verify the file format matches exactly: MM/DD/YY,N1-N2-N3-N4-N5-N6")
        print("3. Ensure there are no header rows or blank lines")
        print("4. All numbers must be between 1 and your configured number_pool")
        print("Example valid line: 04/30/25,27-55-44-19-48-43")

if __name__ == "__main__":
    main()