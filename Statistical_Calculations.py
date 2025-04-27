import argparse
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
import statsmodels.formula.api as smf


def load_and_prepare(path: str) -> pd.DataFrame:
    """
    Load CSV data, rename columns, and add binary consensus column.
    """
    df = pd.read_csv(path, sep=';')
    df = df.rename(columns={
        'Filename': 'scenario',
        'Reasoning Depth (Avg)': 'depth_avg',
        'Disagreements': 'disagreements',
        'Consensus Reached': 'consensus',
        'Best Agents (Evaluator)': 'best_agents'
    })
    df['scenario'] = df['scenario'].astype(int)
    df['consensus_binary'] = df['consensus'].map({'Yes': 1, 'No': 0})
    return df


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    """
    Compute Clopper-Pearson (exact) binomial confidence interval.
    """
    return proportion_confint(k, n, alpha=alpha, method='beta')


def h1_consensus_rates(df: pd.DataFrame, label: str) -> None:
    """
    Print overall and scenario-level consensus rates with 95% CIs.
    """
    total = len(df)
    successes = df['consensus_binary'].sum()
    low, high = clopper_pearson(successes, total)
    rate = successes / total
    print(f"{label} overall: {successes}/{total} = {rate:.1%}; 95% CI = {low:.1%}–{high:.1%}")
    for scenario, group in df.groupby('scenario'):
        k = group['consensus_binary'].sum()
        n = len(group)
        low_s, high_s = clopper_pearson(k, n)
        rate_s = k / n
        print(f"  {scenario}%: {k}/{n} = {rate_s:.1%}; 95% CI = {low_s:.1%}–{high_s:.1%}")


def h2_fisher(normal: pd.DataFrame, manipulated: pd.DataFrame) -> None:
    """
    Perform Fisher's exact test comparing consensus in normal vs manipulated data.
    """
    a = normal['consensus_binary'].sum()
    b = len(normal) - a
    c = manipulated['consensus_binary'].sum()
    d = len(manipulated) - c
    table = np.array([[a, b], [c, d]])
    odds, p_value = fisher_exact(table)
    tbl = Table2x2(table)
    ci_low, ci_high = tbl.oddsratio_confint()
    print("H2 Fisher's Exact Test")
    print(f"Contingency table:
{table}")
    print(f"Odds ratio = {odds:.2f} (95% CI = {ci_low:.2f}–{ci_high:.2f}); p = {p_value:.2f}")


def h3_analysis(df: pd.DataFrame) -> None:
    """
    Analyze reasoning depth and disagreements versus consensus,
    including Mann–Whitney U, Spearman correlation, and logistic regression.
    """
    depth_yes = df[df['consensus_binary'] == 1]['depth_avg']
    depth_no = df[df['consensus_binary'] == 0]['depth_avg']
    u_depth, p_depth = mannwhitneyu(depth_yes, depth_no, alternative='two-sided')
    rho_depth, p_rho_depth = spearmanr(df['depth_avg'], df['consensus_binary'])
    print("H3 Reasoning Depth")
    print(f"Mann-Whitney U = {u_depth:.1f}; p = {p_depth:.3f}")
    print(f"Spearman rho = {rho_depth:.2f}; p = {p_rho_depth:.3f}")

    dis_yes = df[df['consensus_binary'] == 1]['disagreements']
    dis_no = df[df['consensus_binary'] == 0]['disagreements']
    u_dis, p_dis = mannwhitneyu(dis_yes, dis_no, alternative='two-sided')
    rho_dis, p_rho_dis = spearmanr(df['disagreements'], df['consensus_binary'])
    print("H3 Disagreements")
    print(f"Mann-Whitney U = {u_dis:.1f}; p = {p_dis:.3f}")
    print(f"Spearman rho = {rho_dis:.2f}; p = {p_rho_dis:.3f}")

    model = smf.logit('consensus_binary ~ depth_avg + disagreements', data=df).fit(disp=False)
    params = np.exp(model.params)
    conf = np.exp(model.conf_int())
    print("H3 Logistic Regression")
    for var in ['depth_avg', 'disagreements']:
        or_val = params[var]
        low_ci, high_ci = conf.loc[var]
        p_val = model.pvalues[var]
        print(f"{var}: OR = {or_val:.4f} (95% CI = {low_ci:.4f}–{high_ci:.4f}); p = {p_val:.3f}")


def h4_trend(df: pd.DataFrame) -> None:
    """
    Test trend of consensus across data completeness levels per dataset
    using Spearman correlation and logistic regression.
    """
    for label, subset in df.groupby('dataset'):
        rho, p_val = spearmanr(subset['scenario'], subset['consensus_binary'])
        model = smf.logit('consensus_binary ~ scenario', data=subset).fit(disp=False)
        or_val = np.exp(model.params['scenario'])
        low_ci, high_ci = np.exp(model.conf_int()).loc['scenario']
        print(f"{label} trend")
        print(f"Spearman rho = {rho:.2f}; p = {p_val:.3f}")
        print(f"Logistic OR per 1% = {or_val:.3f} (95% CI = {low_ci:.3f}–{high_ci:.3f}); p = {model.pvalues['scenario']:.3f}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for input file paths.
    """
    parser = argparse.ArgumentParser(description="Run consensus analyses for H1–H4")
    parser.add_argument("--normal", required=True, help="Path to normal data CSV")
    parser.add_argument("--manipulated", required=True, help="Path to manipulated data CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normal_df = load_and_prepare(args.normal)
    manip_df = load_and_prepare(args.manipulated)
    normal_df['dataset'] = 'Normal'
    manip_df['dataset'] = 'Manipulated'
    combined = pd.concat([normal_df, manip_df], ignore_index=True)

    print("=== H1 Consensus Rates ===")
    h1_consensus_rates(normal_df, "Normal")
    h1_consensus_rates(manip_df, "Manipulated")

    print("\n=== H2 Data Quality Effect ===")
    h2_fisher(normal_df, manip_df)

    print("\n=== H3 Conflict Emergence ===")
    h3_analysis(combined)

    print("\n=== H4 Completeness Trend ===")
    h4_trend(combined)


if __name__ == "__main__":
    main()
