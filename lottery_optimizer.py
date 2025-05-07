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

        if self.config['output']['verbose']:
            print("\nSYSTEM INITIALIZED WITH:")
            print(f"- Number pool: 1-{self.config['strategy']['number_pool']}")
            print(f"- {len(self.historical)} historical draws loaded")
            if self.upcoming is not None:
                print(f"- {len(self.upcoming)} upcoming draws loaded")
            print(f"- {len(self.prime_numbers)} prime numbers in pool")
            print(f"- Current cold numbers: {sorted(self.cold_numbers)}")

    def initialize_number_properties(self):
        """Initialize all number-related properties from config"""
        self.number_pool = list(range(1, self.config['strategy']['number_pool'] + 1))
        self.prime_numbers = [n for n in self.number_pool if sympy.isprime(n)]
        
        if self.config['output']['verbose']:
            print(f"Identified {len(self.prime_numbers)} primes: {self.prime_numbers}")

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
            
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            print("Config file should be in YAML format with proper indentation")
            raise

    def prepare_filesystem(self):
        """Ensure required directories exist"""
        try:
            Path(self.config['data']['stats_dir']).mkdir(parents=True, exist_ok=True)
            Path(self.config['data']['results_dir']).mkdir(parents=True, exist_ok=True)
            if self.config['output']['verbose']:
                print(f"Created directories: {self.config['data']['stats_dir']}, {self.config['data']['results_dir']}")
        except Exception as e:
            print(f"Error creating directories: {str(e)}")
            raise

    def load_and_clean_data(self):
        """Load and clean lottery data with dynamic column handling"""
        try:
            num_select = self.config['strategy']['numbers_to_select']
            num_cols = [f'n{i+1}' for i in range(num_select)]
            
            # Load historical data
            hist_path = self.config['data']['historical_path']
            if self.config['output']['verbose']:
                print(f"\nLOADING DATA FROM: {hist_path}")
            
            self.historical = pd.read_csv(
                hist_path, 
                header=None, 
                names=['date', 'numbers'],
                dtype={'date': str, 'numbers': str}
            )
            
            # Split numbers into dynamic columns
            self.historical[num_cols] = self.historical['numbers'].str.split('-', expand=True).astype(int)
            self.historical['date'] = pd.to_datetime(self.historical['date'], format='%m/%d/%y', errors='raise')
            
            # Load upcoming draws if configured
            if self.config['data']['upcoming_path']:
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
            
            if self.config['output']['verbose']:
                print(f"Successfully loaded {len(self.historical)} draws")
                
        except Exception as e:
            print(f"\nDATA LOADING ERROR: {str(e)}")
            print("Required format: MM/DD/YY,N1-N2-... (one draw per line)")
            print(f"Expected {num_select} numbers per draw")
            raise

    def validate_data(self):
        """Validate loaded data against current configuration"""
        num_select = self.config['strategy']['numbers_to_select']
        num_cols = [f'n{i+1}' for i in range(num_select)]
        max_num = self.config['strategy']['number_pool']
        
        # Check historical data
        for col in num_cols:
            invalid = self.historical[
                (self.historical[col] < 1) | 
                (self.historical[col] > max_num)
            ]
            if not invalid.empty:
                raise ValueError(f"Invalid numbers found in column {col} (range 1-{max_num})")

    def analyze_numbers(self):
        """Analyze number patterns with dynamic column handling"""
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
            print(f"\nCold numbers (not drawn in last 20 games): {sorted(self.cold_numbers)}")
            if self.overrepresented_pairs:
                print("\nMost common number pairs:")
                for pair in sorted(self.overrepresented_pairs, key=lambda x: -self.weights[x[0]]*self.weights[x[1]])[:5]:
                    print(f"{pair[0]}-{pair[1]}")

    def _find_overrepresented_pairs(self):
        """Identify number pairs appearing together more than expected"""
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
        """Calculate dynamic weights for number selection"""
        base_weights = pd.Series(1.0, index=self.number_pool)
        
        # Frequency weighting
        if not self.frequencies.empty:
            freq_weights = (self.frequencies / self.frequencies.sum()).fillna(0)
            base_weights += freq_weights * self.config['strategy']['frequency_weight'] * 10
        
        # Recent appearance weighting
        recent_weights = (self.recent_counts / self.recent_counts.sum()).fillna(0)
        base_weights += recent_weights * self.config['strategy']['recent_weight'] * 5
        
        # Apply strategy adjustments
        for num in self.high_performance_numbers:
            base_weights[num] *= 1.5
            
        for n1, n2 in self.overrepresented_pairs:
            base_weights[n1] *= 0.9
            base_weights[n2] *= 0.9
            
        for num in self.cold_numbers:
            base_weights[num] *= np.random.uniform(1.1, 1.3)
            
        # Add randomness component
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
        """Generate number sets using all strategies"""
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
            print("\nGENERATED NUMBER SETS:")
            for i, (nums, strategy) in enumerate(sets, 1):
                print(f"Set {i}: {'-'.join(map(str, nums))} ({strategy})")
        
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
        """Generate improved sets based on validation results"""
        try:
            if 'match_stats' in previous_results:
                for nums in previous_results.get('high_performance_sets', []):
                    self.high_performance_numbers.update(nums)
            
            if self.config['output']['verbose']:
                print("\nUPDATING HIGH PERFORMANCE NUMBERS:")
                print(f"Current high performers: {sorted(self.high_performance_numbers)}")
            
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
                    f.write(f"{'-'.join(map(str, nums))},{strategy}\n")
                    
            if self.config['output']['verbose']:
                print(f"\nSAVED RESULTS TO: {output_file}")
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
                if self.optimizer.config['output']['verbose']:
                    print("\nRUNNING HISTORICAL VALIDATION...")
                
                historical_results = self.test_historical()
                results['historical'] = historical_results
                
                improved_sets = self.optimizer.generate_improved_sets(historical_results)
                self.optimizer.last_generated_sets = improved_sets
                
                if mode == 'both':
                    improved_results = self.test_historical(sets=improved_sets)
                    results['improved'] = improved_results
            
            if mode in ('new_draw', 'both') and self.optimizer.upcoming is not None:
                if self.optimizer.config['output']['verbose']:
                    print("\nTESTING AGAINST UPCOMING DRAWS...")
                results['new_draw'] = self.check_new_draws()
            
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
        """Test against historical draws with complete match reporting"""
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
        
        # Calculate percentages
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
        """Check against upcoming draws with validation"""
        num_select = self.optimizer.config['strategy']['numbers_to_select']
        results = {
            'draws_tested': len(self.optimizer.upcoming),
            'matches': []
        }
        
        for _, draw in self.optimizer.upcoming.iterrows():
            target = set(draw[[f'n{i+1}' for i in range(num_select)]])
            best_match = max(
                len(set(generated_set) & target)
                for generated_set, _ in (self.optimizer.last_generated_sets or self.optimizer.generate_sets())
            )
            results['matches'].append(best_match)
        
        if self.optimizer.config['output']['verbose']:
            print("\nUPCOMING DRAW PREDICTIONS:")
            print(f"Best matches against {len(results['matches'])} upcoming draws:")
            print(f"Match counts: {collections.Counter(results['matches'])}")
        
        return results

    def save_report(self, results):
        """Save validation report to JSON with error handling"""
        try:
            report_file = Path(self.optimizer.config['data']['stats_dir']) / 'validation_report.json'
            
            # Ensure results are JSON-serializable
            serializable_results = self._convert_results(results)
            
            with open(report_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            if self.optimizer.config['output']['verbose']:
                print(f"\nSAVED VALIDATION REPORT TO: {report_file}")
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
            
            if 'historical' in results:
                hist = results['historical']
                print("\nHISTORICAL VALIDATION:")
                print(f"Tested against {hist['draws_tested']} draws")
                print("\nMATCH DISTRIBUTION:")
                for i in range(self.optimizer.config['strategy']['numbers_to_select'] + 1):
                    print(f"{i} matches: {hist['match_counts'][i]} ({hist['match_percentages'][f'{i}_matches']})")
                
                hp_nums = sorted(self.optimizer.high_performance_numbers)
                print(f"\nHIGH-PERFORMANCE NUMBERS ({len(hp_nums)}):")
                print(", ".join(map(str, hp_nums)))
            
            if 'improved' in results:
                impr = results['improved']
                print("\nIMPROVEMENT AFTER ADAPTATION:")
                print(f"4+ match rate improvement: "
                      f"{float(hist['match_percentages']['4_matches'][:-1])}% → "
                      f"{float(impr['match_percentages']['4_matches'][:-1])}%")
            
            if 'new_draw' in results:
                print("\nUPCOMING DRAW PREDICTIONS:")
                matches = results['new_draw']['matches']
                print(f"Best matches against {len(matches)} upcoming draws:")
                print(f"Match counts: {collections.Counter(matches)}")
            
            if self.optimizer.last_generated_sets:
                print("\nRECOMMENDED NUMBER SETS:")
                for i, (nums, strategy) in enumerate(self.optimizer.last_generated_sets, 1):
                    print(f"Set {i}: {'-'.join(map(str, nums))} ({strategy})")
            
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"Error printing results: {str(e)}")

def parse_args():
    """Parse command line arguments with error handling"""
    parser = argparse.ArgumentParser(description='Adaptive Lottery Number Optimizer')
    parser.add_argument('--mode', choices=['historical', 'new_draw', 'both', 'none'],
                       help='Validation mode to run')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose output')
    try:
        return parser.parse_args()
    except Exception as e:
        print(f"Argument parsing error: {str(e)}")
        return argparse.Namespace(mode=None, verbose=False)

def main():
    print("🎰 ADAPTIVE LOTTERY OPTIMIZER")
    print("=============================")
    
    args = parse_args()
    
    try:
        optimizer = AdaptiveLotteryOptimizer()
        
        # Override config verbosity if CLI flag is set
        if args.verbose:
            optimizer.config['output']['verbose'] = True
        
        # Initial generation
        initial_sets = optimizer.generate_sets()
        print("\nINITIAL NUMBER SETS:")
        for i, (nums, strategy) in enumerate(initial_sets, 1):
            print(f"Set {i:>2}: {'-'.join(map(str, nums))} ({strategy})")
        
        # Run validation if configured
        if args.mode or optimizer.config['validation']['mode'] != 'none':
            results = optimizer.run_validation(args.mode)
        
        # Save final improved sets
        if optimizer.save_results(optimizer.last_generated_sets or initial_sets):
            print(f"\n✓ Final optimized sets saved to '{optimizer.config['data']['results_dir']}/suggestions.csv'")
        
    except Exception as e:
        print(f"\n💥 Critical Error: {str(e)}")
        print("Stack trace:")
        traceback.print_exc()
        print("\nTROUBLESHOOTING:")
        print("1. Check your historical.csv file exists in the data/ directory")
        print("2. Verify the file format matches exactly: MM/DD/YY,N1-N2-... (one draw per line)")
        print(f"3. Ensure all numbers are between 1-{optimizer.config['strategy']['number_pool'] if 'optimizer' in locals() else 'CONFIGURED_MAX'}")
        print("Example valid line: 04/30/25,27-55-44-19-48-43")

if __name__ == "__main__":
    main()