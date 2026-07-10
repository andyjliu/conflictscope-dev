import os
import re
import argparse
import json
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import statsmodels.stats.proportion as smp
from sklearn.metrics.pairwise import cosine_similarity
import choix
from sentence_transformers import SentenceTransformer

# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['savefig.dpi'] = 300

def clean_value_name(value_name):
    """Clean value name for display in plots."""
    if value_name.startswith('being as '):
        value_name = value_name[9:]
    if '(i.e.' in value_name:
        value_name = value_name.split('(i.e.')[0].strip()   
    if value_name.endswith(' as possible'):
        value_name = value_name[:-12]
    return value_name.capitalize()

def load_data_files(data_dir):
    """Load all CSV files from the specified directory."""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    data = {}
    
    for file in files:
        try:
            file_path = os.path.join(data_dir, file)
            df = pd.read_csv(file_path)
            model_name = file.split('.csv')[0]
            data[model_name] = df
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    return data

def combine_data(model_data):
    """Combine all model data into a single DataFrame."""
    dfs = []
    for model_name, df in model_data.items():
        if 'model' not in df.columns:
            df['model'] = model_name
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_binomial_ci(successes, total):
    """
    Compute confidence interval for binomial proportion using normal approximation.
    
    Args:
        successes: Number of successful trials
        total: Total number of trials
        
    Returns:
        tuple: (ci_lower, ci_upper)
    """
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_continuous_ci(mean, std, n):
    """
    Compute confidence interval for continuous variable using normal approximation.
    
    Args:
        mean: Sample mean
        std: Sample standard deviation
        n: Sample size
        
    Returns:
        tuple: (mean, standard_error) for 95% confidence interval
    """
    if n == 0:
        return 0, 0
        
    z = 1.96  # 95% confidence level
    se = std / np.sqrt(n)  # standard error = std / sqrt(n)
    ci_lower = mean - z * se
    ci_upper = mean + z * se
    
    return ci_lower, ci_upper

def compute_average_diversity(df):
    """Compute average diversity score for a DataFrame of scenarios with confidence intervals."""
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # Group by sorted value pairs
    df['sorted_values'] = df.apply(
        lambda row: tuple(sorted([row['value1'], row['value2']])), 
        axis=1)
    diversity_scores = []
    for sorted_values, group in df.groupby('sorted_values'):
        descriptions = group['description'].tolist()
        if len(descriptions) > 1:  # Need at least 2 descriptions to compute similarity
            embeddings = model.encode(descriptions)
            similarities = cosine_similarity(embeddings)
            # Get average similarity excluding self-similarity
            n = similarities.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_similarity = similarities[mask].mean()
            diversity_scores.append(1 - avg_similarity)  # Convert similarity to diversity
            
    if not diversity_scores:
        return {'mean': 0, 'ci_lower': 0, 'ci_upper': 0}
        
    mean_diversity = np.mean(diversity_scores)
    std_diversity = np.std(diversity_scores)
    n = len(diversity_scores)
    ci_lower, ci_upper = compute_continuous_ci(mean_diversity, std_diversity, n)
    
    return {
        'mean': mean_diversity,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }

def compute_discriminative_power(preference_rates):
    """
    Compute the rate of statistically significant differences between values.
    
    Returns:
        float: Proportion of value pairs with statistically significant preferences (0-1)
    """
    total_pairs = len(preference_rates)
    if total_pairs == 0:
        return 0.0
        
    significant_diffs = 0
    for _, row in preference_rates.iterrows():
        # Check if confidence interval doesn't include 0.5
        if (row['ci_lower'] > 0.5) or (row['ci_upper'] < 0.5):
            significant_diffs += 1
    
    # Return the rate (proportion) of significant differences
    return significant_diffs / total_pairs

def compute_binary_ci(successes, total):
    """
    Compute confidence interval for binary proportion using normal approximation.
    
    Args:
        successes: Number of successful trials
        total: Total number of trials
        
    Returns:
        tuple: (ci_lower, ci_upper) for 95% confidence interval
    """
    if total == 0:
        return 0, 0
        
    p_hat = successes / total
    z = 1.96  # 95% confidence level
    se = np.sqrt(p_hat * (1 - p_hat) / total)
    ci_lower = max(0, p_hat - z * se)
    ci_upper = min(1, p_hat + z * se)
    
    return ci_lower, ci_upper

def compute_filter_pass_rates(df):
    """Compute pass rates for each filter and overall with confidence intervals."""
    if 'check_results' not in df.columns:
        return {}
    
    # Parse check_results if they're stored as strings
    if df['check_results'].dtype == 'object':
        df['check_results'] = df['check_results'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
    
    filter_rates = {}
    total_scenarios = len(df)
    
    # Compute individual filter pass rates
    for row in df.iterrows():
        check_results = row[1]['check_results']
        if isinstance(check_results, dict):
            for filter_name, passed in check_results.items():
                if filter_name not in filter_rates:
                    filter_rates[filter_name] = {'successes': 0, 'total': 0}
                filter_rates[filter_name]['successes'] += int(passed)
                filter_rates[filter_name]['total'] += 1
    
    # Convert counts to rates and compute CIs
    result = {}
    for filter_name, counts in filter_rates.items():
        rate = counts['successes'] / counts['total']
        ci_lower, ci_upper = compute_binomial_ci(
            counts['successes'], counts['total']
        )
        result[filter_name] = {
            'rate': rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }
    
    # Add overall pass rate
    overall_successes = len(df[df['keep_scenario'] == True])
    overall_rate = overall_successes / total_scenarios
    overall_ci_lower, overall_ci_upper = compute_binomial_ci(
        overall_successes, total_scenarios
    )
    result['overall'] = {
        'rate': overall_rate,
        'ci_lower': overall_ci_lower,
        'ci_upper': overall_ci_upper
    }
    
    return result

def compute_observed_agreement(data):
    """
    Compute the observed agreement between models efficiently - the probability that two randomly 
    chosen models will give the same response to a randomly chosen question.
    
    Args:
        data (DataFrame): DataFrame containing model choices with columns:
            - scenario_id: ID of the scenario
            - model: Name of the model
            - choice: 'A' or 'B' indicating which value was chosen
            
    Returns:
        float: Observed agreement rate between models
    """
    # Convert choices to binary (0 for A, 1 for B)
    data['binary_choice'] = (data['choice'] != 'A').astype(int)
    
    # Create a pivot table with scenarios as rows and models as columns
    pivot = data.pivot_table(
        index='scenario_id',
        columns='model',
        values='binary_choice',
        aggfunc='first'
    )
    
    # Remove scenarios with any missing responses
    pivot = pivot.dropna()
    
    if len(pivot) == 0:
        return -1
    
    # Convert to numpy array for faster computation
    response_matrix = pivot.values
    n_scenarios, n_models = response_matrix.shape
    
    # For each scenario, count agreements across all model pairs
    total_agreements = 0
    total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
    
    # Vectorized computation of agreements
    for i in range(n_models):
        for j in range(i+1, n_models):
            # Count where models i and j gave the same answer
            agreements = np.sum(response_matrix[:, i] == response_matrix[:, j])
            total_agreements += agreements
    
    # Return agreement rate
    return total_agreements / total_comparisons if total_comparisons > 0 else -1

def compute_iaa(data):
    """Compute interannotator agreement by scenario generation model with confidence intervals."""
    grouped = data.groupby(['generating_model', 'scenario_id'])
    
    iaa_results = {
        'observed_agreement': {},
        'ci_lower': {},
        'ci_upper': {}
    }
    
    for gen_model, group in data.groupby('generating_model'):
        try:
            # Compute agreement for this model
            agreement = compute_observed_agreement(group)
            
            if agreement >= 0:
                # Count total comparisons
                n_models = len(group['model'].unique())
                n_scenarios = len(group['scenario_id'].unique())
                total_comparisons = n_scenarios * (n_models * (n_models - 1)) // 2
                
                # Count agreements
                total_agreements = agreement * total_comparisons
                
                # Compute confidence intervals
                ci_lower, ci_upper = compute_binomial_ci(
                    total_agreements, total_comparisons
                )
                
                iaa_results['observed_agreement'][gen_model] = agreement
                iaa_results['ci_lower'][gen_model] = ci_lower
                iaa_results['ci_upper'][gen_model] = ci_upper
            else:
                iaa_results['observed_agreement'][gen_model] = None
                iaa_results['ci_lower'][gen_model] = None
                iaa_results['ci_upper'][gen_model] = None
                
        except Exception as e:
            print(f"Error computing observed agreement for {gen_model}: {e}")
            iaa_results['observed_agreement'][gen_model] = None
            iaa_results['ci_lower'][gen_model] = None
            iaa_results['ci_upper'][gen_model] = None
    
    return iaa_results

def compute_preference_rates(data):
    """Compute preference rates for each pair of values and each model."""
    normalized_data = data.copy()
    
    def sort_value_pair(row):
        values = sorted([row['value1'], row['value2']])
        flip_choice = (row['value1'] != values[0]) and (row['choice'] in ['A', 'B'])
        return values[0], values[1], flip_choice
    
    normalized_data[['normalized_value1', 'normalized_value2', 'flip_choice']] = normalized_data.apply(
        sort_value_pair, axis=1, result_type='expand'
    )
    
    normalized_data['normalized_choice'] = normalized_data.apply(
        lambda row: ('B' if row['choice'] == 'A' else 'A') if row['flip_choice'] else row['choice'],
        axis=1
    )
    
    grouped = normalized_data.groupby(['model', 'normalized_value1', 'normalized_value2'])
    
    results = []
    for (model, value1, value2), group in grouped:
        total = len(group)
        count_a = (group['normalized_choice'] == 'A').sum()
        count_b = (group['normalized_choice'] == 'B').sum()
        
        preference_rate = count_a / total if total > 0 else 0
        
        if total > 0:
            ci = smp.proportion_confint(count_a, total, alpha=0.05, method='wilson')
            ci_lower = ci[0]
            ci_upper = ci[1]
        else:
            ci_lower = 0
            ci_upper = 0
        
        results.append({
            'model': model,
            'value1': value1,
            'value2': value2,
            'preference_rate': preference_rate,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'total': total,
            'count_value1': count_a,
            'count_value2': count_b
        })
    
    return pd.DataFrame(results)

def plot_iaa_results(iaa_results, title, output_dir):
    """Plot IAA results."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gen_models = list(iaa_results['observed_agreement'].keys())
    agreements = [iaa_results['observed_agreement'][model] for model in gen_models]
    std_errors = [abs(iaa_results['ci_upper'][model] - iaa_results['observed_agreement'][model]) for model in gen_models]
    
    sorted_data = sorted(zip(gen_models, agreements, std_errors), key=lambda x: x[1])
    gen_models, agreements, std_errors = zip(*sorted_data)
    
    bars = ax.bar(gen_models, agreements, color='lightgreen')
    ax.errorbar(gen_models, agreements, yerr=std_errors, fmt='none', ecolor='black', capsize=5)
    
    for bar, agreement in zip(bars, agreements):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{agreement:.2f}",
            ha='center',
            va='bottom',
            color='black'
        )
    
    ax.set_title('Observed Agreement Between Models')
    ax.set_ylabel('Observed Agreement Rate')
    ax.set_ylim(0, max(agreements) + 0.15)
    ax.grid(True, axis='y')
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    filename = f"iaa_results_{title.replace(' ', '_').replace('/', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    
    return fig

def plot_preference_rates(preference_rates, title, output_dir):
    """Plot preference rates for each pair of values."""
    value_pairs = preference_rates[['value1', 'value2']].drop_duplicates().values
    n_pairs = len(value_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4 * n_pairs))
    
    if n_pairs == 1:
        axes = [axes]
    
    for i, (value1, value2) in enumerate(value_pairs):
        pair_data = preference_rates[(preference_rates['value1'] == value1) & 
                                   (preference_rates['value2'] == value2)]
        pair_data = pair_data.sort_values('preference_rate')
        
        ax = axes[i]
        bars = ax.barh(pair_data['model'], pair_data['preference_rate'], color='skyblue')
        
        ax.errorbar(
            pair_data['preference_rate'],
            pair_data['model'],
            xerr=np.vstack([
                abs(pair_data['preference_rate'] - pair_data['ci_lower']),
                abs(pair_data['ci_upper'] - pair_data['preference_rate'])
            ]),
            fmt='none',
            ecolor='black',
            capsize=5
        )
        
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, axis='x')
        ax.axvline(0.5, color='gray', linestyle='--')
        
        clean_v1 = clean_value_name(value1)
        clean_v2 = clean_value_name(value2)
        
        ax.set_title(f'Preference Rate: {clean_v1} vs {clean_v2}')
        ax.set_xlabel(f'Preference Rate (proportion of responses favoring {clean_v1} over {clean_v2})')
        
        for j, (bar, value) in enumerate(zip(bars, pair_data['preference_rate'])):
            ax.text(
                bar.get_width() + 0.01, 
                j,
                f"{value:.2f}",
                va='center',
                ha='left',
                color='black'
            )
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    filename = f"preference_rates_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    
    return fig

def compute_likert_metrics(data):
    """
    Compute Likert metrics including:
    - Average Likert rating across both actions
    - Average Likert error
    - Average polarization (absolute difference between actions)
    - Average polarization error
    - Likert diff rate (how often likert_a and likert_b are different)
    
    Args:
        data (DataFrame): DataFrame containing model evaluation data with 'likert_a' and 'likert_b' columns
        
    Returns:
        tuple: (likert_preferences, likert_polarization, likert_diff_result) DataFrames and dict with simplified metrics
    """
    if 'likert_a' not in data.columns or 'likert_b' not in data.columns:
        print("No 'likert_a' or 'likert_b' columns found in the data. Skipping Likert scale analysis.")
        return None, None, None
    
    # Make sure likert columns are numeric
    data['likert_a'] = pd.to_numeric(data['likert_a'], errors='coerce')
    data['likert_b'] = pd.to_numeric(data['likert_b'], errors='coerce')
    
    # Drop rows with missing likert values
    data_clean = data.dropna(subset=['likert_a', 'likert_b'])
    
    if len(data_clean) == 0:
        print("No valid Likert scale ratings found. Skipping Likert scale analysis.")
        return None, None, None
    
    # Compute likert_diff metric first
    different_count = (data_clean['likert_a'] != data_clean['likert_b']).sum()
    total_count = len(data_clean)
    likert_diff_rate = different_count / total_count if total_count > 0 else 0
    ci_lower, ci_upper = compute_binomial_ci(different_count, total_count)
    
    likert_diff_result = {
        'rate': likert_diff_rate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'count': total_count,
        'different_count': different_count
    }
    
    # Normalize value pairs
    normalized_data = data_clean.copy()
    
    def sort_value_pair(row):
        values = sorted([row['value1'], row['value2']])
        flip_likert = (row['value1'] != values[0])
        return values[0], values[1], flip_likert
    
    normalized_data[['normalized_value1', 'normalized_value2', 'flip_likert']] = normalized_data.apply(
        sort_value_pair, axis=1, result_type='expand'
    )
    
    # Flip the likert scores if needed
    normalized_data['normalized_likert_a'] = normalized_data.apply(
        lambda row: -row['likert_a'] if row['flip_likert'] else row['likert_a'],
        axis=1
    )
    normalized_data['normalized_likert_b'] = normalized_data.apply(
        lambda row: -row['likert_b'] if row['flip_likert'] else row['likert_b'],
        axis=1
    )
    
    # Compute metrics by model and value pair
    grouped = normalized_data.groupby(['model', 'normalized_value1', 'normalized_value2'])
    
    likert_preferences = []
    likert_polarization = []
    
    for (model, value1, value2), group in grouped:
        # Compute average likert across both actions
        avg_likert = (group['normalized_likert_a'].mean() + group['normalized_likert_b'].mean()) / 2
        std_likert = np.sqrt((group['normalized_likert_a'].std()**2 + group['normalized_likert_b'].std()**2) / 2)
        
        # Calculate 95% confidence interval for the mean
        n = len(group)
        se = std_likert / (n ** 0.5) if n > 0 else 0
        
        # t-distribution with n-1 degrees of freedom, for small samples
        import scipy.stats as stats
        if n > 1:
            t_val = stats.t.ppf(0.975, n-1)
            ci_lower = avg_likert - t_val * se
            ci_upper = avg_likert + t_val * se
        else:
            ci_lower = avg_likert
            ci_upper = avg_likert
        
        # Compute polarization (absolute difference between likert ratings)
        # First compute absolute differences for each comparison
        abs_diffs = (group['normalized_likert_a'] - group['normalized_likert_b']).abs()
        avg_polarization = abs_diffs.mean()
        std_polarization = abs_diffs.std()
        
        # Calculate confidence interval for polarization
        se_polarization = std_polarization / (n ** 0.5) if n > 0 else 0
        if n > 1:
            ci_lower_polarization = avg_polarization - t_val * se_polarization
            ci_upper_polarization = avg_polarization + t_val * se_polarization
        else:
            ci_lower_polarization = avg_polarization
            ci_upper_polarization = avg_polarization
        
        likert_preferences.append({
            'model': model,
            'value1': value1,
            'value2': value2,
            'avg_likert': avg_likert,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'count': n
        })
        
        likert_polarization.append({
            'model': model,
            'value1': value1,
            'value2': value2,
            'avg_polarization': avg_polarization,
            'ci_lower': ci_lower_polarization,
            'ci_upper': ci_upper_polarization,
            'count': n
        })
    
    likert_preferences_df = pd.DataFrame(likert_preferences)
    likert_polarization_df = pd.DataFrame(likert_polarization)
    
    return likert_preferences_df, likert_polarization_df, likert_diff_result

def plot_likert_preferences(likert_preferences, title, output_dir):
    """Plot average Likert scale ratings and polarization for each pair of values."""
    value_pairs = likert_preferences[['value1', 'value2']].drop_duplicates().values
    n_pairs = len(value_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(12, 4 * n_pairs))
    
    if n_pairs == 1:
        axes = [axes]
    
    for i, (value1, value2) in enumerate(value_pairs):
        pair_data = likert_preferences[(likert_preferences['value1'] == value1) & 
                                      (likert_preferences['value2'] == value2)]
        pair_data = pair_data.sort_values('avg_likert')
        
        ax = axes[i]
        
        # Plot average likert ratings
        bars = ax.barh(pair_data['model'], pair_data['avg_likert'], 
                      color='lightgreen', alpha=0.7)
        
        # Add error bars
        ax.errorbar(
            pair_data['avg_likert'],
            pair_data['model'],
            xerr=np.vstack([
                pair_data['avg_likert'] - pair_data['ci_lower'],
                pair_data['ci_upper'] - pair_data['avg_likert']
            ]),
            fmt='none',
            ecolor='black',
            capsize=5,
            alpha=0.5
        )
        
        ax.set_xlim(-1, 1)
        ax.grid(True, axis='x')
        ax.axvline(0, color='gray', linestyle='--')
        
        clean_v1 = clean_value_name(value1)
        clean_v2 = clean_value_name(value2)
        
        ax.set_title(f'Average Likert Ratings: {clean_v1} vs {clean_v2}')
        ax.set_xlabel('Average Likert Rating (-1 to 1)')
        
        # Add value labels
        for j, (bar, value) in enumerate(zip(bars, pair_data['avg_likert'])):
            ax.text(
                bar.get_width() + (0.01 * (1 if value >= 0 else -1)), 
                j,
                f"{value:.2f}",
                va='center',
                ha='left' if value >= 0 else 'right',
                color='black'
            )
    
    plt.tight_layout()
    fig.suptitle(title, fontsize=16, y=1.02)
    
    filename = f"likert_preferences_{title.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    
    return fig

def create_value_hierarchy(preference_rates, significance_threshold=0.05):
    """Create a value hierarchy based on pairwise preferences."""
    hierarchies = {}
    
    for model, model_data in preference_rates.groupby('model'):
        graph = {}
        
        for _, row in model_data.iterrows():
            value1 = clean_value_name(row['value1'])
            value2 = clean_value_name(row['value2'])
            pref_rate = row['preference_rate']
            
            significant = (row['ci_lower'] > 0.5) or (row['ci_upper'] < 0.5)
            
            if value1 not in graph:
                graph[value1] = {'score': 0, 'edges': [], 'non_sig_edges': []}
            if value2 not in graph:
                graph[value2] = {'score': 0, 'edges': [], 'non_sig_edges': []}
            
            if significant:
                if pref_rate > 0.5:
                    graph[value1]['score'] += 1
                    graph[value1]['edges'].append((value2, pref_rate))
                elif pref_rate < 0.5:
                    graph[value2]['score'] += 1
                    graph[value2]['edges'].append((value1, 1 - pref_rate))
            else:
                if pref_rate >= 0.5:
                    graph[value1]['non_sig_edges'].append((value2, pref_rate))
                else:
                    graph[value2]['non_sig_edges'].append((value1, 1 - pref_rate))
        
        hierarchies[model] = graph
    
    return hierarchies

def plot_value_hierarchy(hierarchy, model, output_dir, show_non_significant=True):
    """Plot value hierarchy."""
    G = nx.DiGraph()
    
    avg_preferences = {}
    for value in hierarchy:
        G.add_node(value)
        total_weight = 0
        num_edges = 0
        for target, weight in hierarchy[value]['edges']:
            total_weight += weight
            num_edges += 1
        for target, weight in hierarchy[value].get('non_sig_edges', []):
            total_weight += weight
            num_edges += 1
        avg_preferences[value] = total_weight / max(num_edges, 1)
    
    non_sig_edges = []
    for source, data in hierarchy.items():
        for target, weight in data['edges']:
            G.add_edge(source, target, weight=weight, significant=True)
        if show_non_significant:
            for target, weight in data.get('non_sig_edges', []):
                G.add_edge(source, target, weight=weight, significant=False)
                non_sig_edges.append((source, target))
    
    ranks = {}
    unranked = set(G.nodes())
    current_rank = 1
    
    while unranked:
        eligible = []
        for node in unranked:
            predecessors = [pred for pred in G.predecessors(node) 
                          if pred in unranked and G[pred][node].get('significant', False)]
            if not predecessors:
                eligible.append(node)
        
        if not eligible:
            eligible = list(unranked)
        
        eligible.sort(key=lambda x: avg_preferences[x], reverse=True)
        
        for node in eligible:
            ranks[node] = current_rank
            unranked.remove(node)
        
        current_rank += 1
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    pos = {}
    rank_groups = {}
    
    for value, rank in ranks.items():
        if rank not in rank_groups:
            rank_groups[rank] = []
        rank_groups[rank].append(value)
    
    for rank in rank_groups:
        rank_groups[rank].sort(key=lambda x: avg_preferences[x], reverse=True)
    
    max_rank = max(ranks.values())
    base_height = max_rank * 4
    
    for rank, values in rank_groups.items():
        width = len(values)
        x_spacing = 8
        for i, value in enumerate(values):
            x = (i - (width - 1)/2) * x_spacing
            pref_adjustment = avg_preferences[value] * 2
            y = base_height - (rank - 1) * 4 + pref_adjustment
            pos[value] = (x, y)
    
    node_sizes = [6000 for _ in range(len(hierarchy))]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.8, ax=ax)
    
    nx.draw_networkx_labels(G, pos, font_size=18, font_weight="bold", ax=ax)
    
    sig_edge_list = [(u, v) for u, v, data in G.edges(data=True) if data.get('significant', True)]
    if sig_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=sig_edge_list, 
                             width=2.5, alpha=0.7, edge_color='gray',
                             connectionstyle='arc3,rad=0.2',
                             arrowsize=25, ax=ax)
        
        edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True) 
                      if data.get('significant', True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                   font_size=16, ax=ax)
    
    if show_non_significant and non_sig_edges:
        nx.draw_networkx_edges(G, pos, edgelist=non_sig_edges, 
                             width=1.5, alpha=0.4, edge_color='lightgray',
                             connectionstyle='arc3,rad=0.2',
                             style='dashed', arrowsize=20, ax=ax)
        
        non_sig_edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True) 
                              if not data.get('significant', True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=non_sig_edge_labels, 
                                   font_size=16, font_color='gray', ax=ax)
    
    ax.axis('off')
    plt.title(f"Value Hierarchy for {model}", fontsize=22, pad=20)
    
    plt.tight_layout()
    short_model_name = model.split('/')[1].replace(' ', '_').lower() if '/' in model else model.replace(' ', '_').lower()
    filename = f"value_hierarchy_{short_model_name}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    
    plt.close()
    
    return fig

def plot_value_hierarchies(preference_rates, title, output_dir, show_non_significant=True):
    """Plot value hierarchies for all models."""
    os.makedirs(output_dir, exist_ok=True)
    hierarchies = create_value_hierarchy(preference_rates)
    
    for model, hierarchy in hierarchies.items():
        plot_value_hierarchy(hierarchy, model, output_dir, show_non_significant)
    
    print(f"Value hierarchy plots saved to {output_dir}")

def log_metrics(model_name, value_set, metric_dict, variant, wandb_project, job_name):
    """Log metrics to W&B with separate rows for each model."""
    import wandb  # imported lazily: only needed with --wandb, and the env's wandb
                  # install is currently broken (protobuf mismatch from the vllm upgrade)
    run = wandb.init(reinit=True, project=wandb_project, name=f"{job_name}_{model_name}_{value_set}_{variant}")

    # Create a simple metrics dictionary with clean metric names
    metrics = {
        'args/model': model_name,
        'args/value_set': value_set,
        'args/data_variant': variant,
        **metric_dict
    }
    wandb.log(metrics)
    run.finish()

def fit_bradley_terry_model(data):
    """
    Fit a Bradley-Terry model to raw model choices and compute bootstrapped confidence intervals.
    
    Args:
        data (DataFrame): DataFrame containing raw model choices with columns:
            - value1: First value in comparison
            - value2: Second value in comparison
            - choice: 'A' or 'B' indicating which value was chosen
            
    Returns:
        tuple: (params, rankings_df) where params are the fitted parameters and rankings_df contains
               the rankings with confidence intervals
    """
    # Get unique values and create mapping to indices
    unique_values = sorted(set(data['value1'].unique()) | set(data['value2'].unique()))
    n_values = len(unique_values)
    value_to_idx = {val: i for i, val in enumerate(unique_values)}
    
    # Convert data to the format required by choix
    comparisons = []
    for _, row in data.iterrows():
        if row['choice'] == 'A':
            # Value1 was chosen over Value2
            comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
        else:  # row['choice'] == 'B'
            # Value2 was chosen over Value1
            comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
    
    # Fit the Bradley-Terry model
    params = choix.ilsr_pairwise(n_values, comparisons, alpha=0.01)
    
    # Bootstrap confidence intervals
    n_bootstrap = 1000
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Sample with replacement from the original data
        bootstrap_data = data.sample(n=len(data), replace=True)
        
        # Convert bootstrap data to choix format
        bootstrap_comparisons = []
        for _, row in bootstrap_data.iterrows():
            if row['choice'] == 'A':
                bootstrap_comparisons.append((value_to_idx[row['value1']], value_to_idx[row['value2']]))
            else:
                bootstrap_comparisons.append((value_to_idx[row['value2']], value_to_idx[row['value1']]))
        
        # Fit model on bootstrap sample
        bootstrap_params = choix.ilsr_pairwise(n_values, bootstrap_comparisons, alpha=0.01)
        bootstrap_estimates.append(bootstrap_params)
    
    # Compute confidence intervals
    bootstrap_estimates = np.array(bootstrap_estimates)
    ci_lower = np.percentile(bootstrap_estimates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_estimates, 97.5, axis=0)
    
    # Create rankings DataFrame
    rankings_df = pd.DataFrame({
        'value': unique_values,
        'ability': params,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })
    
    # Sort by ability
    rankings_df = rankings_df.sort_values('ability', ascending=False)
    
    return params, rankings_df

def plot_bradley_terry_rankings(rankings_df, model_name, output_dir):
    """
    Plot Bradley-Terry model rankings with confidence intervals.
    
    Args:
        rankings_df (DataFrame): DataFrame containing rankings and confidence intervals
        model_name (str): Name of the model being evaluated
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Sort values by ability
    rankings_df = rankings_df.sort_values('ability')
    
    # Create point plot with error bars
    y_pos = np.arange(len(rankings_df))
    plt.errorbar(rankings_df['ability'], y_pos, 
                xerr=[rankings_df['ability'] - rankings_df['ci_lower'], 
                      rankings_df['ci_upper'] - rankings_df['ability']],
                fmt='o', capsize=5, capthick=2, elinewidth=2, markersize=8)
    
    # Add value names
    plt.yticks(y_pos, [clean_value_name(v) for v in rankings_df['value']])
    
    # Add labels and title
    plt.xlabel('Bradley-Terry Ability Score')
    plt.title(f'Value Rankings for {model_name}')
    
    # Add grid
    plt.grid(True, axis='x')
    
    # Save plot
    plt.tight_layout()
    short_model_name = model_name.split('/')[1].replace(' ', '_').lower() if '/' in model_name else model_name.replace(' ', '_').lower()
    filename = f"bradley_terry_rankings_{short_model_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze ConflictBench model evaluations.')
    
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing model evaluation CSV files')
    parser.add_argument('--scenario-dir', type=str, required=True,
                        help='Directory containing scenario CSV files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save output files')
    parser.add_argument('--hide-non-significant', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-project', type=str, default='conflictbench_gen',
                        help='W&B project name')
    parser.add_argument('--job-name', type=str, default=None)
    parser.add_argument('--value-set', type=str, required=True,
                        help='Name of the value set being evaluated')
    parser.add_argument('--compute-bradley-terry', action='store_true',
                        help='Compute Bradley-Terry model rankings')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all data
    model_data = load_data_files(args.model_dir)
    scenario_data = load_data_files(args.scenario_dir)

    # Combine all scenario data with filtering information
    combined_scenarios = combine_data(scenario_data)
    if 'keep_scenario' not in combined_scenarios.columns:
        combined_scenarios['keep_scenario'] = True
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario']].drop_duplicates()
    else:
        scenario_filters = combined_scenarios[['scenario_id', 'keep_scenario', 'check_results']].drop_duplicates()
    combined_data = combine_data(model_data)
    
    # Merge filtering information
    combined_data = combined_data.merge(scenario_filters, on='scenario_id', how='left')

    data_variants = {
        'unfiltered': combined_data,
        'filtered': combined_data[combined_data['keep_scenario'] == True]
    }
    if len(data_variants['unfiltered']) == len(data_variants['filtered']):
        print("Filtered and unfiltered datasets are identical. Running only filtered analysis.")
        data_variants = {'filtered': data_variants['filtered']}
    
    # Process each variant
    for variant_name, data in data_variants.items():
        
        # Get list of generation models
        generating_models = data['generating_model'].unique()
        
        # Group data by generation model for processing
        for gen_model in tqdm(generating_models, desc="Processing generation models"):
            
            # Get scenarios for this generation model
            gen_model_scenarios = combined_scenarios[combined_scenarios['generating_model'] == gen_model]
            # Get evaluation data for this generation model
            gen_model_data = data[data['generating_model'] == gen_model]

            # Create a metrics dictionary for this generation model
            model_metrics = {}
            
            # 1. Compute filter pass rates
            filter_rates = compute_filter_pass_rates(gen_model_scenarios)
            for filter_name, rates in filter_rates.items():
                model_metrics[f'filtering/{filter_name}'] = rates['rate']
                error = max(rates['ci_upper'] - rates['rate'], rates['rate'] - rates['ci_lower'])
                model_metrics[f'filtering/{filter_name}_error'] = error
                
            # 2. Compute diversity
            diversity = compute_average_diversity(gen_model_scenarios)
            model_metrics['metrics/diversity'] = diversity['mean']
            error = max(diversity['ci_upper'] - diversity['mean'], diversity['mean'] - diversity['ci_lower'])
            model_metrics['metrics/diversity_error'] = error
            
            # 3. Compute IAA (observed agreement)
            iaa_results = compute_iaa(gen_model_data)
            if gen_model in iaa_results['observed_agreement']:
                model_metrics['metrics/observed_agreement'] = iaa_results['observed_agreement'][gen_model]
                error = max(iaa_results['ci_upper'][gen_model] - iaa_results['observed_agreement'][gen_model],
                          iaa_results['observed_agreement'][gen_model] - iaa_results['ci_lower'][gen_model])
                model_metrics['metrics/observed_agreement_error'] = error

            # 4. Compute preference rates and discriminative power
            preference_rates = compute_preference_rates(gen_model_data)
            
            # Calculate overall discriminative power for this generation model
            disc_power = compute_discriminative_power(preference_rates)
            model_metrics['metrics/discriminative_power'] = disc_power

            likert_preferences, likert_polarization, likert_diff_result = compute_likert_metrics(gen_model_data)
            
            if likert_preferences is not None:
                # Compute average likert rating (absolute value)
                avg_likert_abs = likert_polarization['avg_polarization'].mean()
                std_likert_abs = likert_polarization['avg_polarization'].std()
                n_likert = len(likert_polarization)
                ci_lower, ci_upper = compute_continuous_ci(avg_likert_abs, std_likert_abs, n_likert)
                
                model_metrics['metrics/avg_likert_polarization'] = avg_likert_abs
                error = max(ci_upper - avg_likert_abs, avg_likert_abs - ci_lower)
                model_metrics['metrics/avg_likert_polarization_error'] = error
                
                # Calculate discriminative power for likert ratings
                if 'ci_lower' in likert_preferences.columns and 'ci_upper' in likert_preferences.columns:
                    total_pairs = len(likert_preferences)
                    significant_pairs = sum(
                        ((row['ci_lower'] > 0) and (row['ci_upper'] > 0)) or 
                        ((row['ci_lower'] < 0) and (row['ci_upper'] < 0))
                        for _, row in likert_preferences.iterrows()
                    )
                    likert_disc_power = significant_pairs / total_pairs if total_pairs > 0 else 0
                    model_metrics['metrics/likert_discriminative_power'] = likert_disc_power
            
            # Add likert_diff metric to model_metrics
            if likert_diff_result is not None:
                model_metrics['metrics/likert_diff'] = likert_diff_result['rate']
                error = max(likert_diff_result['ci_upper'] - likert_diff_result['rate'], 
                          likert_diff_result['rate'] - likert_diff_result['ci_lower'])
                model_metrics['metrics/likert_diff_error'] = error

            # Log metrics for this generation model
            if args.wandb:
                if args.job_name is None:
                    args.job_name = args.output_dir.split('/')[-2]
                log_metrics(gen_model, args.value_set, model_metrics, variant_name, args.wandb_project, args.job_name)
            
        # Generate plots for this variant
        iaa_results = compute_iaa(data)
        preference_rates = compute_preference_rates(data)
        
        plot_iaa_results(iaa_results, f"IAA Results - {variant_name.title()}", args.output_dir)
        plot_preference_rates(preference_rates, f"Preference Rates - {variant_name.title()}", args.output_dir)
        plot_value_hierarchies(preference_rates, f"Value Hierarchies - {variant_name.title()}", 
                             f"{args.output_dir}/value_hierarchy_{variant_name}", 
                             not args.hide_non_significant)
        likert_preferences, likert_polarization, likert_diff_result = compute_likert_metrics(data)
        if likert_preferences is not None:
            plot_likert_preferences(likert_preferences, f"Likert Preferences - {variant_name.title()}", args.output_dir)
            
        # Compute Bradley-Terry model for each evaluation model (only if flag is set)
        if args.compute_bradley_terry:
            all_rankings = []
            for eval_model, eval_data in tqdm(data.groupby('model'), desc="Computing Bradley-Terry model"):
                params, rankings_df = fit_bradley_terry_model(eval_data)
                bradley_terry_output_dir = os.path.join(args.output_dir, 'bradley_terry_rankings')
                os.makedirs(bradley_terry_output_dir, exist_ok=True)
                plot_bradley_terry_rankings(rankings_df, eval_model, bradley_terry_output_dir)
                rankings_df['model'] = eval_model
                all_rankings.append(rankings_df)
            
            # Combine all rankings and save to a single CSV
            if all_rankings:
                combined_rankings = pd.concat(all_rankings, ignore_index=True)
                combined_rankings.to_csv(os.path.join(args.output_dir, f"bradley_terry_rankings_{variant_name}.csv"), index=False)
            
        # Save preference rates to CSV
        preference_rates.to_csv(os.path.join(args.output_dir, f"preference_rates_{variant_name}.csv"), index=False)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()