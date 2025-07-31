#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Friedman Test Analysis for ExtraTrees vs TPOT with Original vs BigFeat Features

This script reads the output CSV files generated from the enhanced ensemble learning experiment
and performs the Friedman test to statistically analyze the performance differences between:

1. ExtraTrees (Original features)
2. ExtraTrees (BigFeat features)
3. TPOT (Original features)
4. TPOT (BigFeat features)

The script analyzes F1 Score and Training Time differences across these four conditions.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
from itertools import combinations
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FriedmanAnalyzer:
    """Class to perform Friedman test analysis on ExtraTrees vs TPOT comparison."""

    def __init__(self, results_dir):
        """
        Initialize the analyzer with the results directory.

        Args:
            results_dir (str): Path to the directory containing CSV result files
        """
        self.results_dir = results_dir
        self.devices_data = {}
        self.output_dir = os.path.join(results_dir, 'friedman_analysis')
        os.makedirs(self.output_dir, exist_ok=True)

        # Define the four conditions we're comparing
        self.conditions = [
            'ExtraTrees_Original',
            'ExtraTrees_BigFeat',
            'TPOT_Original',
            'TPOT_BigFeat'
        ]

    def load_device_results(self):
        """Load all device-specific CSV files and extract ExtraTrees and TPOT data."""
        print("Loading device result files...")

        # Find all device metric files
        device_files = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('_metrics.csv') and filename.startswith('device'):
                device_files.append(filename)

        if not device_files:
            raise FileNotFoundError("No device metric files found in the results directory")

        print(f"Found {len(device_files)} device metric files")

        # Load each device's results
        for filename in sorted(device_files):
            device_name = filename.split('_')[0]  # Extract device name (e.g., 'device1')
            file_path = os.path.join(self.results_dir, filename)

            try:
                df = pd.read_csv(file_path)

                # Determine feature type from filename
                if 'original' in filename.lower():
                    feature_type = 'Original'
                elif 'bigfeat' in filename.lower():
                    feature_type = 'BigFeat'
                else:
                    # Skip files we can't identify
                    print(f"Skipping {filename} - cannot determine feature type")
                    continue

                # Store the data
                if device_name not in self.devices_data:
                    self.devices_data[device_name] = {}

                self.devices_data[device_name][feature_type] = df
                print(f"Loaded {filename}: {len(df)} models")

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

        print(f"Successfully loaded data for {len(self.devices_data)} devices")

    def extract_target_algorithms(self, metric='F1 Score'):
        """
        Extract data for the four target algorithm-feature combinations.

        Args:
            metric (str): Metric to extract ('F1 Score' or 'Training Time')

        Returns:
            pd.DataFrame: Data with devices as rows and conditions as columns
        """
        print(f"Extracting data for target algorithms - {metric}...")

        # Initialize result dataframe
        result_data = pd.DataFrame(columns=self.conditions)

        devices_with_complete_data = []

        for device_name, feature_data in self.devices_data.items():
            device_row = {}
            missing_data = False

            # Check both feature types
            for feature_type in ['Original', 'BigFeat']:
                if feature_type not in feature_data:
                    print(f"Warning: {feature_type} data not available for {device_name}")
                    missing_data = True
                    continue

                df = feature_data[feature_type]

                if metric not in df.columns:
                    print(f"Warning: {metric} not found in {device_name} {feature_type} data")
                    missing_data = True
                    continue

                # Extract ExtraTrees data
                extratrees_data = df[df['Model'] == 'ExtraTrees']
                if len(extratrees_data) > 0:
                    condition_name = f'ExtraTrees_{feature_type}'
                    device_row[condition_name] = extratrees_data[metric].iloc[0]
                else:
                    print(f"Warning: ExtraTrees not found in {device_name} {feature_type}")
                    missing_data = True

                # Extract TPOT data
                tpot_data = df[df['Model'] == 'TPOT (ExtraTrees)']
                if len(tpot_data) > 0:
                    condition_name = f'TPOT_{feature_type}'
                    device_row[condition_name] = tpot_data[metric].iloc[0]
                else:
                    print(f"Warning: TPOT (ExtraTrees) not found in {device_name} {feature_type}")
                    missing_data = True

            # Only include devices with complete data for all 4 conditions
            if not missing_data and len(device_row) == 4:
                result_data = pd.concat([result_data, pd.DataFrame([device_row], index=[device_name])],
                                        ignore_index=False)
                devices_with_complete_data.append(device_name)
            else:
                print(f"Excluding {device_name} due to incomplete data")

        print(f"Devices with complete data: {len(devices_with_complete_data)}")
        print(f"Included devices: {devices_with_complete_data}")
        print(f"Final data shape: {result_data.shape}")

        return result_data

    def perform_friedman_test(self, data, metric_name):
        """
        Perform the Friedman test on the four algorithm-feature combinations.

        Args:
            data (pd.DataFrame): Data matrix with devices as rows and conditions as columns
            metric_name (str): Name of the metric being analyzed

        Returns:
            dict: Test results including statistic, p-value, and rankings
        """
        print(f"\nPerforming Friedman test for {metric_name}...")
        print(f"Comparing: {list(data.columns)}")

        if len(data) < 3:
            print(f"Warning: Only {len(data)} datasets available. Need at least 3 for reliable test.")
            return None

        if data.shape[1] != 4:
            print(f"Error: Expected 4 conditions, but got {data.shape[1]}")
            return None

        print(f"Using {len(data)} datasets for analysis")

        # Check for any missing values
        if data.isnull().any().any():
            print("Warning: Found missing values in data")
            data = data.dropna()
            print(f"After removing missing values: {len(data)} datasets")

        # Prepare data for Friedman test (each column is a group)
        groups = [data[col].values for col in data.columns]

        # Perform Friedman test
        try:
            statistic, p_value = friedmanchisquare(*groups)
        except Exception as e:
            print(f"Error performing Friedman test: {str(e)}")
            return None

        # Calculate average rankings
        rankings = {}

        # For each dataset (row), rank the algorithms
        if 'Time' in metric_name:
            # For time metrics, lower is better (rank ascending)
            ranks = data.rank(axis=1, method='average', ascending=True)
        else:
            # For performance metrics like F1 Score, higher is better (rank descending)
            ranks = data.rank(axis=1, method='average', ascending=False)

        for col in data.columns:
            rankings[col] = ranks[col].mean()

        # Sort by average ranking (lower rank = better performance)
        sorted_rankings = dict(sorted(rankings.items(), key=lambda x: x[1]))

        results = {
            'statistic': statistic,
            'p_value': p_value,
            'rankings': sorted_rankings,
            'data_shape': data.shape,
            'conditions': list(data.columns),
            'complete_data': data
        }

        print(f"Friedman test results:")
        print(f"  Test statistic: {statistic:.4f}")
        print(f"  P-value: {p_value:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        print(f"  Significant at α=0.01: {'Yes' if p_value < 0.01 else 'No'}")

        print(f"\nAverage Rankings (1=best):")
        for i, (condition, rank) in enumerate(sorted_rankings.items(), 1):
            print(f"  {i}. {condition}: {rank:.2f}")

        return results

    def perform_posthoc_analysis(self, friedman_results, metric_name):
        """
        Perform post-hoc analysis using Nemenyi test.

        Args:
            friedman_results (dict): Results from Friedman test
            metric_name (str): Name of the metric being analyzed

        Returns:
            pd.DataFrame: Pairwise comparison results
        """
        if friedman_results is None or friedman_results['p_value'] >= 0.05:
            print("Skipping post-hoc analysis (Friedman test not significant)")
            return None

        print(f"\nPerforming post-hoc analysis (Nemenyi test) for {metric_name}...")

        complete_data = friedman_results['complete_data']

        try:
            # Prepare data in the correct format for post-hoc test
            # scikit_posthocs expects data as a 2D array where:
            # - rows are observations (datasets/devices)
            # - columns are treatments (conditions)
            data_array = complete_data.values  # Keep original orientation

            # Perform Nemenyi test
            posthoc_results = sp.posthoc_nemenyi_friedman(data_array)

            # Set the correct index and column names
            posthoc_results.index = complete_data.columns
            posthoc_results.columns = complete_data.columns

            print("Nemenyi test results (p-values):")
            print(posthoc_results.round(4))

            # Create a summary of significant differences
            significant_pairs = []
            conditions = list(complete_data.columns)

            for i, cond1 in enumerate(conditions):
                for j, cond2 in enumerate(conditions):
                    if i < j:  # Avoid duplicate pairs
                        p_val = posthoc_results.loc[cond1, cond2]
                        if p_val < 0.05:
                            rank1 = friedman_results['rankings'][cond1]
                            rank2 = friedman_results['rankings'][cond2]
                            better_condition = cond1 if rank1 < rank2 else cond2
                            significant_pairs.append({
                                'Condition_1': cond1,
                                'Condition_2': cond2,
                                'P_value': p_val,
                                'Significant': 'Yes',
                                'Better_Condition': better_condition,
                                'Rank_Diff': abs(rank1 - rank2)
                            })

            if significant_pairs:
                print(f"\nSignificant pairwise differences (α=0.05):")
                for pair in significant_pairs:
                    print(f"  {pair['Condition_1']} vs {pair['Condition_2']}: "
                          f"p={pair['P_value']:.4f}, {pair['Better_Condition']} is better")

                # Save significant pairs to CSV
                pairs_df = pd.DataFrame(significant_pairs)
                pairs_file = os.path.join(self.output_dir,
                                          f'significant_pairs_{metric_name.replace(" ", "_").lower()}.csv')
                pairs_df.to_csv(pairs_file, index=False)
                print(f"Significant pairs saved to: {pairs_file}")
            else:
                print("No significant pairwise differences found")

            return posthoc_results

        except Exception as e:
            print(f"Error performing post-hoc analysis: {str(e)}")
            print(f"Data shape: {complete_data.shape}")
            print(f"Data columns: {list(complete_data.columns)}")
            return None

    def create_comprehensive_summary(self, f1_results, time_results):
        """
        Create a comprehensive summary comparing all conditions.

        Args:
            f1_results (dict): Friedman test results for F1 Score
            time_results (dict): Friedman test results for Training Time
        """
        print("\nCreating comprehensive summary...")

        summary_data = []

        # Extract data for summary
        conditions = self.conditions

        for condition in conditions:
            row = {'Condition': condition}

            # Add F1 Score ranking
            if f1_results and condition in f1_results['rankings']:
                row['F1_Rank'] = f1_results['rankings'][condition]
                row['F1_Mean'] = f1_results['complete_data'][condition].mean()
                row['F1_Std'] = f1_results['complete_data'][condition].std()
            else:
                row['F1_Rank'] = np.nan
                row['F1_Mean'] = np.nan
                row['F1_Std'] = np.nan

            # Add Training Time ranking
            if time_results and condition in time_results['rankings']:
                row['Time_Rank'] = time_results['rankings'][condition]
                row['Time_Mean'] = time_results['complete_data'][condition].mean()
                row['Time_Std'] = time_results['complete_data'][condition].std()
            else:
                row['Time_Rank'] = np.nan
                row['Time_Mean'] = np.nan
                row['Time_Std'] = np.nan

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)

        # Sort by F1 rank (best first)
        summary_df = summary_df.sort_values('F1_Rank')

        print("\nComprehensive Summary:")
        print(summary_df.to_string(index=False))

        # Save summary
        summary_file = os.path.join(self.output_dir, 'comprehensive_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Comprehensive summary saved to: {summary_file}")

        return summary_df

    def create_enhanced_posthoc_visualization(self, posthoc_results, friedman_results, metric_name):
        """
        Create an enhanced post-hoc analysis visualization with better annotations and insights.

        Args:
            posthoc_results (pd.DataFrame): Results from Nemenyi test
            friedman_results (dict): Results from Friedman test
            metric_name (str): Name of the metric being analyzed
        """
        if posthoc_results is None:
            return

        print(f"Creating enhanced post-hoc visualization for {metric_name}...")

        # Create a comprehensive post-hoc figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. P-value heatmap with FIXED color mapping
        # Create a custom colormap that properly maps significance
        from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

        # Define color boundaries - p < 0.05 is significant (red), p >= 0.05 is not significant (green)
        boundaries = [0, 0.05, 1.0]
        colors_list = ['red', 'lightgreen']  # Red for significant, light green for not significant
        cmap = LinearSegmentedColormap.from_list('significance', colors_list, N=256)
        norm = BoundaryNorm(boundaries, cmap.N)

        # Create the heatmap with proper color mapping
        sns.heatmap(posthoc_results,
                    annot=True,
                    fmt='.4f',
                    cmap=cmap,
                    norm=norm,
                    cbar_kws={'label': 'P-value', 'boundaries': boundaries, 'ticks': [0.025, 0.525]},
                    ax=ax1)

        # Manually set colorbar labels
        cbar = ax1.collections[0].colorbar
        cbar.set_ticklabels(['Significant\n(p < 0.05)', 'Not Significant\n(p ≥ 0.05)'])

        ax1.set_title(
            f'Nemenyi Post-Hoc P-Values ({metric_name})\nRed: Significant (p < 0.05), Green: Not Significant (p ≥ 0.05)')
        ax1.set_xlabel('Candidate Pipeline')
        ax1.set_ylabel('Candidate Pipeline')

        # Add significance annotations with proper logic
        for i in range(len(posthoc_results.index)):
            for j in range(len(posthoc_results.columns)):
                p_val = posthoc_results.iloc[i, j]

                # Determine significance symbol
                if i == j:  # Diagonal (self-comparison)
                    sig_text = '—'  # Dash for self-comparison
                    text_color = 'black'
                elif p_val < 0.001:
                    sig_text = '***'
                    text_color = 'white'
                elif p_val < 0.01:
                    sig_text = '**'
                    text_color = 'white'
                elif p_val < 0.05:
                    sig_text = '*'
                    text_color = 'white'
                else:
                    sig_text = 'ns'
                    text_color = 'black'

                ax1.text(j + 0.5, i + 0.25, sig_text,
                         ha='center', va='center',
                         fontsize=10, fontweight='bold',
                         color=text_color)

        # 2. Ranking comparison bar plot
        rankings = friedman_results['rankings']
        candidate_solutions = list(rankings.keys())
        ranks = list(rankings.values())

        # Color bars based on ranking (best to worst)
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(candidate_solutions)))
        bars = ax2.bar(range(len(candidate_solutions)), ranks, color=colors)

        ax2.set_xlabel('Candidate Pipeline')
        ax2.set_ylabel('Average Rank (lower is better)')
        ax2.set_title(f'Average Rankings ({metric_name})')
        ax2.set_xticks(range(len(candidate_solutions)))
        ax2.set_xticklabels([c.replace('_', '\n') for c in candidate_solutions], rotation=0, fontsize=10)

        # Add rank values on bars
        for bar, rank in zip(bars, ranks):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{rank:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        # 3. Significant pairs network-style visualization
        ax3.set_xlim(0, 10)
        ax3.set_ylim(0, 10)
        ax3.set_aspect('equal')

        # Position candidate solutions in a circle
        n_solutions = len(candidate_solutions)
        angles = np.linspace(0, 2 * np.pi, n_solutions, endpoint=False)
        positions = {}

        for i, solution in enumerate(candidate_solutions):
            x = 5 + 3 * np.cos(angles[i])
            y = 5 + 3 * np.sin(angles[i])
            positions[solution] = (x, y)

            # Draw solution nodes
            circle = plt.Circle((x, y), 0.5, color=colors[i], alpha=0.7)
            ax3.add_patch(circle)
            ax3.text(x, y, solution.replace('_', '\n'), ha='center', va='center',
                     fontsize=8, fontweight='bold')

        # Draw significant connections
        for i in range(len(candidate_solutions)):
            for j in range(i + 1, len(candidate_solutions)):
                sol1, sol2 = candidate_solutions[i], candidate_solutions[j]
                p_val = posthoc_results.loc[sol1, sol2]

                if p_val < 0.05:  # Only show significant connections
                    x1, y1 = positions[sol1]
                    x2, y2 = positions[sol2]

                    # Line thickness based on significance
                    if p_val < 0.001:
                        linewidth = 4
                        alpha = 0.9
                    elif p_val < 0.01:
                        linewidth = 3
                        alpha = 0.7
                    else:
                        linewidth = 2
                        alpha = 0.5

                    ax3.plot([x1, x2], [y1, y2], 'r-', linewidth=linewidth, alpha=alpha)

                    # Add p-value label
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax3.text(mid_x, mid_y, f'{p_val:.3f}', ha='center', va='center',
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                             fontsize=8)

        ax3.set_title(f'Significant Pairwise Differences ({metric_name})\nRed lines indicate p < 0.05')
        ax3.axis('off')

        # 4. Effect size and practical significance
        complete_data = friedman_results['complete_data']

        # Calculate effect sizes (difference in means relative to pooled standard deviation)
        effect_sizes = []
        pair_labels = []
        p_values = []

        for i in range(len(candidate_solutions)):
            for j in range(i + 1, len(candidate_solutions)):
                sol1, sol2 = candidate_solutions[i], candidate_solutions[j]

                mean1 = complete_data[sol1].mean()
                mean2 = complete_data[sol2].mean()

                # Pooled standard deviation
                pooled_std = np.sqrt((complete_data[sol1].var() + complete_data[sol2].var()) / 2)

                if pooled_std > 0:
                    effect_size = abs(mean1 - mean2) / pooled_std
                else:
                    effect_size = 0

                effect_sizes.append(effect_size)
                p_values.append(posthoc_results.loc[sol1, sol2])
                pair_labels.append(
                    f'{sol1.split("_")[0][:2]}-{sol1.split("_")[1][:3]}\nvs\n{sol2.split("_")[0][:2]}-{sol2.split("_")[1][:3]}')

        bars = ax4.bar(range(len(effect_sizes)), effect_sizes)
        ax4.set_xlabel('Candidate Pipeline Pairs')
        ax4.set_ylabel('Effect Size (Cohen\'s d)')
        ax4.set_title(f'Effect Sizes ({metric_name})\nSmall: 0.2, Medium: 0.5, Large: 0.8')
        ax4.set_xticks(range(len(pair_labels)))
        ax4.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)

        # Color bars by both effect size magnitude AND statistical significance
        for i, (bar, effect_size, p_val) in enumerate(zip(bars, effect_sizes, p_values)):
            if p_val < 0.05:  # Statistically significant
                if effect_size >= 0.8:
                    bar.set_color('darkred')  # Large effect + significant
                elif effect_size >= 0.5:
                    bar.set_color('red')  # Medium effect + significant
                elif effect_size >= 0.2:
                    bar.set_color('orange')  # Small effect + significant
                else:
                    bar.set_color('yellow')  # Negligible effect but significant
            else:  # Not statistically significant
                bar.set_color('lightgray')  # Not significant

            # Add effect size values on bars
            ax4.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                     f'{effect_size:.2f}', ha='center', va='bottom', fontsize=9)

        # Add horizontal lines for effect size thresholds
        ax4.axhline(y=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
        ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.9, label='Large effect')
        ax4.legend(fontsize=8, loc='upper right')

        plt.tight_layout()

        # Save the comprehensive post-hoc visualization
        output_file = os.path.join(self.output_dir,
                                   f'comprehensive_posthoc_{metric_name.replace(" ", "_").lower()}.pdf')
        plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Enhanced post-hoc visualization saved to: {output_file}")

        return output_file

    def create_visualizations(self, f1_results, time_results, f1_posthoc=None, time_posthoc=None):
        """
        Create visualizations for the analysis results as separate PDF files, including post-hoc heatmaps.

        Args:
            f1_results (dict): Friedman test results for F1 Score
            time_results (dict): Friedman test results for Training Time
            f1_posthoc (pd.DataFrame): Nemenyi test results for F1 Score
            time_posthoc (pd.DataFrame): Nemenyi test results for Training Time
        """
        print("\nCreating visualizations...")

        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")

        # Define colors for each condition
        colors = {
            'ExtraTrees_Original': '#2E8B57',  # Sea Green
            'ExtraTrees_BigFeat': '#90EE90',  # Light Green
            'TPOT_Original': '#B22222',  # Fire Brick
            'TPOT_BigFeat': '#FF6347'  # Tomato
        }

        # 1. F1 Score Rankings (Bar Plot) - Fixed version
        if f1_results:
            fig1 = plt.figure(figsize=(8, 6))
            ax1 = fig1.add_subplot(111)
            conditions = list(f1_results['rankings'].keys())
            ranks = list(f1_results['rankings'].values())
            bars = ax1.bar(range(len(conditions)), ranks,
                           color=[colors[c] for c in conditions])
            ax1.set_xlabel('Candidate Pipeline')
            ax1.set_ylabel('Average Rank (lower is better)')
            ax1.set_title('F1 Score Rankings')
            ax1.set_xticks(range(len(conditions)))
            ax1.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, fontsize=8)
            for bar, rank in zip(bars, ranks):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{rank:.2f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            fig1.savefig(os.path.join(self.output_dir, 'f1_score_rankings.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig1)
            print(f"F1 Score Rankings saved to: {os.path.join(self.output_dir, 'f1_score_rankings.pdf')}")

        # 2. Training Time Rankings (Bar Plot) - Fixed version
        if time_results:
            fig2 = plt.figure(figsize=(8, 6))
            ax2 = fig2.add_subplot(111)
            conditions = list(time_results['rankings'].keys())
            ranks = list(time_results['rankings'].values())
            bars = ax2.bar(range(len(conditions)), ranks,
                           color=[colors[c] for c in conditions])
            ax2.set_xlabel('Candidate Pipeline')
            ax2.set_ylabel('Average Rank (lower is better)')
            ax2.set_title('Training Time Rankings')
            ax2.set_xticks(range(len(conditions)))
            ax2.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, fontsize=8)
            for bar, rank in zip(bars, ranks):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{rank:.2f}', ha='center', va='bottom', fontsize=9)
            plt.tight_layout()
            fig2.savefig(os.path.join(self.output_dir, 'training_time_rankings.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig2)
            print(f"Training Time Rankings saved to: {os.path.join(self.output_dir, 'training_time_rankings.pdf')}")

        # 3. Box Plot for F1 Score
        if f1_results:
            fig3 = plt.figure(figsize=(8, 6))
            ax3 = fig3.add_subplot(111)
            data_for_boxplot = []
            labels_for_boxplot = []
            for condition in self.conditions:
                if condition in f1_results['complete_data'].columns:
                    data_for_boxplot.append(f1_results['complete_data'][condition].values)
                    labels_for_boxplot.append(condition.replace('_', '\n'))
            bp = ax3.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
            for patch, condition in zip(bp['boxes'], self.conditions):
                if condition in colors:
                    patch.set_facecolor(colors[condition])
                    patch.set_alpha(0.7)
            ax3.set_ylabel('F1 Score')
            ax3.set_xlabel('Candidate Pipeline')
            ax3.set_title('F1 Score Distribution')
            ax3.tick_params(axis='x', labelsize=8)
            plt.setp(ax3.get_xticklabels(), rotation=0)
            plt.tight_layout()
            fig3.savefig(os.path.join(self.output_dir, 'f1_score_distribution.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig3)
            print(f"F1 Score Distribution saved to: {os.path.join(self.output_dir, 'f1_score_distribution.pdf')}")

        # 4. Box Plot for Training Time
        if time_results:
            fig4 = plt.figure(figsize=(8, 6))
            ax4 = fig4.add_subplot(111)
            data_for_boxplot = []
            labels_for_boxplot = []
            for condition in self.conditions:
                if condition in time_results['complete_data'].columns:
                    data_for_boxplot.append(time_results['complete_data'][condition].values)
                    labels_for_boxplot.append(condition.replace('_', '\n'))
            bp = ax4.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
            for patch, condition in zip(bp['boxes'], self.conditions):
                if condition in colors:
                    patch.set_facecolor(colors[condition])
                    patch.set_alpha(0.7)
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_title('Training Time Distribution')
            ax4.tick_params(axis='x', labelsize=8)
            plt.setp(ax4.get_xticklabels(), rotation=0)
            plt.tight_layout()
            fig4.savefig(os.path.join(self.output_dir, 'training_time_distribution.pdf'), format='pdf',
                         bbox_inches='tight')
            plt.close(fig4)
            print(
                f"Training Time Distribution saved to: {os.path.join(self.output_dir, 'training_time_distribution.pdf')}")

        # 5. Scatter Plot: F1 vs Time
        if f1_results and time_results:
            fig5 = plt.figure(figsize=(8, 6))
            ax5 = fig5.add_subplot(111)
            for condition in self.conditions:
                if (condition in f1_results['complete_data'].columns and
                        condition in time_results['complete_data'].columns):
                    f1_values = f1_results['complete_data'][condition]
                    time_values = time_results['complete_data'][condition]
                    ax5.scatter(time_values, f1_values,
                                color=colors[condition], label=condition.replace('_', ' '),
                                alpha=0.7, s=60)
            ax5.set_xlabel('Training Time (seconds)')
            ax5.set_ylabel('F1 Score')
            ax5.set_title('F1 Score vs Training Time')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)
            plt.tight_layout()
            fig5.savefig(os.path.join(self.output_dir, 'f1_vs_training_time.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig5)
            print(f"F1 Score vs Training Time saved to: {os.path.join(self.output_dir, 'f1_vs_training_time.pdf')}")

        # 6. Performance Comparison Heatmap - Fixed version (using time values for coloring)
        if f1_results and time_results:
            fig6 = plt.figure(figsize=(8, 6))
            ax6 = fig6.add_subplot(111)
            comparison_data = []
            condition_labels = []
            time_values = []  # For coloring
            for condition in self.conditions:
                if (condition in f1_results['rankings'] and
                        condition in time_results['rankings']):
                    f1_rank = f1_results['rankings'][condition]
                    time_rank = time_results['rankings'][condition]
                    time_mean = time_results['complete_data'][condition].mean()

                    comparison_data.append([f1_rank, time_rank])
                    time_values.append([time_mean, time_mean])  # Use actual time for both columns
                    condition_labels.append(condition.replace('_', ' '))

            if comparison_data:
                comparison_matrix = np.array(comparison_data)
                time_matrix = np.array(time_values)

                # Create the heatmap using time values for coloring
                im = ax6.imshow(time_matrix, cmap='RdYlGn_r', aspect='auto')
                ax6.set_xticks([0, 1])
                ax6.set_xticklabels(['F1 Rank', 'Time Rank'])
                ax6.set_yticks(range(len(condition_labels)))
                ax6.set_yticklabels(condition_labels, fontsize=8)
                ax6.set_title('Rankings Heatmap\n(Color: Training Time, Labels: Rank Values)')

                # Add rank values as text annotations
                for i in range(len(condition_labels)):
                    for j in range(2):
                        text = ax6.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontweight='bold')

                cbar = plt.colorbar(im, ax=ax6, shrink=0.6)
                cbar.set_label('Training Time (seconds)')
                plt.tight_layout()
                fig6.savefig(os.path.join(self.output_dir, 'rankings_heatmap.pdf'), format='pdf', bbox_inches='tight')
                plt.close(fig6)
                print(f"Rankings Heatmap saved to: {os.path.join(self.output_dir, 'rankings_heatmap.pdf')}")

        # 7. Post-Hoc Heatmap for F1 Score (FIXED VERSION)
        if f1_posthoc is not None:
            fig7 = plt.figure(figsize=(8, 6))
            ax7 = fig7.add_subplot(111)

            # Create proper color mapping for significance
            from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
            boundaries = [0, 0.05, 1.0]
            colors_list = ['red', 'lightgreen']  # Red for significant, green for not significant
            cmap = LinearSegmentedColormap.from_list('significance', colors_list, N=256)
            norm = BoundaryNorm(boundaries, cmap.N)

            sns.heatmap(f1_posthoc, annot=True, fmt='.4f', cmap=cmap, norm=norm,
                        cbar_kws={'label': 'P-value'}, ax=ax7)

            ax7.set_title(
                'Nemenyi Post-Hoc P-Values (F1 Score)\nRed: Significant (p < 0.05), Green: Not Significant (p ≥ 0.05)')
            ax7.set_xlabel('Candidate Pipeline')
            ax7.set_ylabel('Candidate Pipeline')
            plt.tight_layout()
            fig7.savefig(os.path.join(self.output_dir, 'f1_posthoc_heatmap.pdf'), format='pdf', bbox_inches='tight')
            plt.close(fig7)
            print(f"F1 Score Post-Hoc Heatmap saved to: {os.path.join(self.output_dir, 'f1_posthoc_heatmap.pdf')}")

        # 8. Post-Hoc Heatmap for Training Time (FIXED VERSION)
        if time_posthoc is not None:
            fig8 = plt.figure(figsize=(8, 6))
            ax8 = fig8.add_subplot(111)

            # Create proper color mapping for significance
            from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
            boundaries = [0, 0.05, 1.0]
            colors_list = ['red', 'lightgreen']  # Red for significant, green for not significant
            cmap = LinearSegmentedColormap.from_list('significance', colors_list, N=256)
            norm = BoundaryNorm(boundaries, cmap.N)

            sns.heatmap(time_posthoc, annot=True, fmt='.4f', cmap=cmap, norm=norm,
                        cbar_kws={'label': 'P-value'}, ax=ax8)

            ax8.set_title(
                'Nemenyi Post-Hoc P-Values (Training Time)\nRed: Significant (p < 0.05), Green: Not Significant (p ≥ 0.05)')
            ax8.set_xlabel('Candidate Pipeline')
            ax8.set_ylabel('Candidate Pipeline')
            plt.tight_layout()
            fig8.savefig(os.path.join(self.output_dir, 'training_time_posthoc_heatmap.pdf'), format='pdf',
                         bbox_inches='tight')
            plt.close(fig8)
            print(
                f"Training Time Post-Hoc Heatmap saved to: {os.path.join(self.output_dir, 'training_time_posthoc_heatmap.pdf')}")
        # Comprehensive Plot (retained as PDF) - Updated with fixes
        fig_combined = plt.figure(figsize=(16, 12))

        # F1 Rankings
        ax1 = fig_combined.add_subplot(2, 3, 1)
        if f1_results:
            conditions = list(f1_results['rankings'].keys())
            ranks = list(f1_results['rankings'].values())
            bars = ax1.bar(range(len(conditions)), ranks,
                           color=[colors[c] for c in conditions])
            ax1.set_xlabel('Candidate Pipeline')
            ax1.set_ylabel('Average Rank (lower is better)')
            ax1.set_title('F1 Score Rankings')
            ax1.set_xticks(range(len(conditions)))
            ax1.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, fontsize=8)
            for bar, rank in zip(bars, ranks):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{rank:.2f}', ha='center', va='bottom', fontsize=9)

        # Time Rankings
        ax2 = fig_combined.add_subplot(2, 3, 2)
        if time_results:
            conditions = list(time_results['rankings'].keys())
            ranks = list(time_results['rankings'].values())
            bars = ax2.bar(range(len(conditions)), ranks,
                           color=[colors[c] for c in conditions])
            ax2.set_xlabel('Candidate Pipeline')
            ax2.set_ylabel('Average Rank (lower is better)')
            ax2.set_title('Training Time Rankings')
            ax2.set_xticks(range(len(conditions)))
            ax2.set_xticklabels([c.replace('_', '\n') for c in conditions], rotation=0, fontsize=8)
            for bar, rank in zip(bars, ranks):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                         f'{rank:.2f}', ha='center', va='bottom', fontsize=9)

        # F1 Distribution
        ax3 = fig_combined.add_subplot(2, 3, 3)
        if f1_results:
            data_for_boxplot = []
            labels_for_boxplot = []
            for condition in self.conditions:
                if condition in f1_results['complete_data'].columns:
                    data_for_boxplot.append(f1_results['complete_data'][condition].values)
                    labels_for_boxplot.append(condition.replace('_', '\n'))
            bp = ax3.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
            for patch, condition in zip(bp['boxes'], self.conditions):
                if condition in colors:
                    patch.set_facecolor(colors[condition])
                    patch.set_alpha(0.7)
            ax3.set_ylabel('F1 Score')
            ax3.set_title('F1 Score Distribution')
            ax3.tick_params(axis='x', labelsize=8)
            plt.setp(ax3.get_xticklabels(), rotation=0)

        # Time Distribution
        ax4 = fig_combined.add_subplot(2, 3, 4)
        if time_results:
            data_for_boxplot = []
            labels_for_boxplot = []
            for condition in self.conditions:
                if condition in time_results['complete_data'].columns:
                    data_for_boxplot.append(time_results['complete_data'][condition].values)
                    labels_for_boxplot.append(condition.replace('_', '\n'))
            bp = ax4.boxplot(data_for_boxplot, labels=labels_for_boxplot, patch_artist=True)
            for patch, condition in zip(bp['boxes'], self.conditions):
                if condition in colors:
                    patch.set_facecolor(colors[condition])
                    patch.set_alpha(0.7)
            ax4.set_ylabel('Training Time (seconds)')
            ax4.set_title('Training Time Distribution')
            ax4.tick_params(axis='x', labelsize=8)
            plt.setp(ax4.get_xticklabels(), rotation=0)

        # Scatter Plot
        ax5 = fig_combined.add_subplot(2, 3, 5)
        if f1_results and time_results:
            for condition in self.conditions:
                if (condition in f1_results['complete_data'].columns and
                        condition in time_results['complete_data'].columns):
                    f1_values = f1_results['complete_data'][condition]
                    time_values = time_results['complete_data'][condition]
                    ax5.scatter(time_values, f1_values,
                                color=colors[condition], label=condition.replace('_', ' '),
                                alpha=0.7, s=60)
            ax5.set_xlabel('Training Time (seconds)')
            ax5.set_ylabel('F1 Score')
            ax5.set_title('F1 Score vs Training Time')
            ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax5.grid(True, alpha=0.3)

        # Rankings Heatmap - Fixed version
        ax6 = fig_combined.add_subplot(2, 3, 6)
        if f1_results and time_results:
            comparison_data = []
            condition_labels = []
            time_values = []
            for condition in self.conditions:
                if (condition in f1_results['rankings'] and
                        condition in time_results['rankings']):
                    f1_rank = f1_results['rankings'][condition]
                    time_rank = time_results['rankings'][condition]
                    time_mean = time_results['complete_data'][condition].mean()

                    comparison_data.append([f1_rank, time_rank])
                    time_values.append([time_mean, time_mean])
                    condition_labels.append(condition.replace('_', ' '))

            if comparison_data:
                comparison_matrix = np.array(comparison_data)
                time_matrix = np.array(time_values)

                im = ax6.imshow(time_matrix, cmap='RdYlGn_r', aspect='auto')
                ax6.set_xticks([0, 1])
                ax6.set_xticklabels(['F1 Rank', 'Time Rank'])
                ax6.set_yticks(range(len(condition_labels)))
                ax6.set_yticklabels(condition_labels, fontsize=8)
                ax6.set_title('Rankings Heatmap\n(Color: Time, Labels: Rank)')

                for i in range(len(condition_labels)):
                    for j in range(2):
                        text = ax6.text(j, i, f'{comparison_matrix[i, j]:.2f}',
                                        ha="center", va="center", color="black", fontweight='bold')
                plt.colorbar(im, ax=ax6, shrink=0.6)

        plt.tight_layout()
        fig_combined.savefig(os.path.join(self.output_dir, 'comprehensive_analysis.pdf'), format='pdf',
                             bbox_inches='tight')
        plt.close(fig_combined)
        print(f"Comprehensive visualization saved to: {os.path.join(self.output_dir, 'comprehensive_analysis.pdf')}")

    def perform_specific_comparisons(self, f1_data, time_data):
        """
        Perform specific pairwise comparisons of interest.

        Args:
            f1_data (pd.DataFrame): F1 Score data
            time_data (pd.DataFrame): Training Time data
        """
        print("\n" + "=" * 60)
        print("SPECIFIC PAIRWISE COMPARISONS")
        print("=" * 60)

        # Define comparisons of interest
        comparisons = [
            ('ExtraTrees_Original', 'ExtraTrees_BigFeat', 'Feature Engineering Effect (ExtraTrees)'),
            ('TPOT_Original', 'TPOT_BigFeat', 'Feature Engineering Effect (TPOT)'),
            ('ExtraTrees_Original', 'TPOT_Original', 'AutoML Effect (Original Features)'),
            ('ExtraTrees_BigFeat', 'TPOT_BigFeat', 'AutoML Effect (BigFeat Features)'),
            ('ExtraTrees_Original', 'TPOT_BigFeat', 'Best Traditional vs Best AutoML'),
            ('TPOT_Original', 'ExtraTrees_BigFeat', 'AutoML Original vs Traditional BigFeat')
        ]

        comparison_results = []

        for cond1, cond2, description in comparisons:
            print(f"\n{description}:")
            print(f"Comparing {cond1} vs {cond2}")

            result_row = {
                'Comparison': description,
                'Condition_1': cond1,
                'Condition_2': cond2
            }

            # F1 Score comparison
            if cond1 in f1_data.columns and cond2 in f1_data.columns:
                f1_1 = f1_data[cond1].dropna()
                f1_2 = f1_data[cond2].dropna()

                if len(f1_1) > 0 and len(f1_2) > 0:
                    # Wilcoxon signed-rank test for paired data
                    try:
                        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(f1_1, f1_2)
                        effect_size = (f1_1.mean() - f1_2.mean()) / f1_data[[cond1, cond2]].stack().std()

                        result_row.update({
                            'F1_Mean_1': f1_1.mean(),
                            'F1_Mean_2': f1_2.mean(),
                            'F1_Diff': f1_1.mean() - f1_2.mean(),
                            'F1_Wilcoxon_Stat': wilcoxon_stat,
                            'F1_Wilcoxon_P': wilcoxon_p,
                            'F1_Effect_Size': effect_size,
                            'F1_Winner': cond1 if f1_1.mean() > f1_2.mean() else cond2
                        })

                        print(f"  F1 Score - {cond1}: {f1_1.mean():.4f} ± {f1_1.std():.4f}")
                        print(f"  F1 Score - {cond2}: {f1_2.mean():.4f} ± {f1_2.std():.4f}")
                        print(f"  Difference: {f1_1.mean() - f1_2.mean():+.4f}")
                        print(
                            f"  Wilcoxon p-value: {wilcoxon_p:.6f} ({'Significant' if wilcoxon_p < 0.05 else 'Not significant'})")
                        print(f"  Effect size: {effect_size:.4f}")

                    except Exception as e:
                        print(f"  Error in F1 comparison: {str(e)}")

            # Training Time comparison
            if cond1 in time_data.columns and cond2 in time_data.columns:
                time_1 = time_data[cond1].dropna()
                time_2 = time_data[cond2].dropna()

                if len(time_1) > 0 and len(time_2) > 0:
                    try:
                        wilcoxon_stat_time, wilcoxon_p_time = stats.wilcoxon(time_1, time_2)
                        time_ratio = time_1.mean() / time_2.mean()

                        result_row.update({
                            'Time_Mean_1': time_1.mean(),
                            'Time_Mean_2': time_2.mean(),
                            'Time_Ratio': time_ratio,
                            'Time_Wilcoxon_Stat': wilcoxon_stat_time,
                            'Time_Wilcoxon_P': wilcoxon_p_time,
                            'Time_Winner': cond1 if time_1.mean() < time_2.mean() else cond2  # Lower time is better
                        })

                        print(f"  Training Time - {cond1}: {time_1.mean():.2f} ± {time_1.std():.2f} seconds")
                        print(f"  Training Time - {cond2}: {time_2.mean():.2f} ± {time_2.std():.2f} seconds")
                        print(f"  Time ratio ({cond1}/{cond2}): {time_ratio:.2f}x")
                        print(
                            f"  Wilcoxon p-value: {wilcoxon_p_time:.6f} ({'Significant' if wilcoxon_p_time < 0.05 else 'Not significant'})")

                    except Exception as e:
                        print(f"  Error in Time comparison: {str(e)}")

            comparison_results.append(result_row)

        # Save detailed comparison results
        if comparison_results:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_file = os.path.join(self.output_dir, 'detailed_pairwise_comparisons.csv')
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\nDetailed pairwise comparisons saved to: {comparison_file}")

        return comparison_results

    def run_complete_analysis(self):
        """Run the complete Friedman test analysis."""
        print("Starting Friedman Test Analysis")
        print("Comparing: ExtraTrees vs TPOT with Original vs BigFeat Features")
        print("=" * 70)

        try:
            # Load all result files
            self.load_device_results()

            if not self.devices_data:
                print("No data loaded. Exiting analysis.")
                return

            # Extract data for our four target conditions
            print(f"\n{'=' * 60}")
            print("EXTRACTING TARGET ALGORITHM DATA")
            print(f"{'=' * 60}")

            f1_data = self.extract_target_algorithms('F1 Score')
            time_data = self.extract_target_algorithms('Training Time')

            if f1_data.empty and time_data.empty:
                print("No valid data found for target algorithms. Exiting analysis.")
                return

            # Perform Friedman test for F1 Score
            print(f"\n{'=' * 60}")
            print("FRIEDMAN TEST - F1 SCORE")
            print(f"{'=' * 60}")
            f1_results = self.perform_friedman_test(f1_data, 'F1 Score')

            # Perform post-hoc analysis for F1 Score
            f1_posthoc = self.perform_posthoc_analysis(f1_results, 'F1 Score')

            # Perform Friedman test for Training Time
            print(f"\n{'=' * 60}")
            print("FRIEDMAN TEST - TRAINING TIME")
            print(f"{'=' * 60}")
            time_results = self.perform_friedman_test(time_data, 'Training Time')

            # Perform post-hoc analysis for Training Time
            time_posthoc = self.perform_posthoc_analysis(time_results, 'Training Time')

            # Create comprehensive summary
            print(f"\n{'=' * 60}")
            print("CREATING COMPREHENSIVE SUMMARY")
            print(f"{'=' * 60}")
            summary_df = self.create_comprehensive_summary(f1_results, time_results)

            # Create visualizations
            self.create_visualizations(f1_results, time_results, f1_posthoc, time_posthoc)

            # Perform specific pairwise comparisons
            self.perform_specific_comparisons(f1_data, time_data)

            print(f"\n{'=' * 60}")
            print("ANALYSIS COMPLETE")
            print(f"{'=' * 60}")
            print(f"Results saved in: {self.output_dir}")

        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            raise


def main():
    """Main function to run the analysis."""
    # Set default results directory
    default_results_dir = "../Results/IoT/Enhanced_Ensemble_TPOT_ExtraTrees_Results"

    # Use command-line argument if provided, otherwise use default
    results_dir = sys.argv[1] if len(sys.argv) == 2 else default_results_dir

    # Verify directory exists
    if not os.path.isdir(results_dir):
        print(f"Error: Directory {results_dir} does not exist")
        sys.exit(1)

    # Initialize and run analyzer
    analyzer = FriedmanAnalyzer(results_dir)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()