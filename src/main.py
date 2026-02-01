import os

import numpy as np
import polars as pl
from numba import njit
from scipy.stats import kurtosis, skew


@njit
def run_strat_1(
    horizon_years, savings_arr, world_arr, em_arr, m_lengths, n_sims, k_val, t_w, t_e, reb_thresh
):
    np.random.seed(42)
    start_age = 65 - horizon_years
    offset = (start_age - 25) * 12
    n_months = horizon_years * 12 - 1
    final_vals = np.zeros(n_sims)
    final_mdd = np.zeros(n_sims)
    final_costs = np.zeros(n_sims)
    final_turnover = np.zeros(n_sims)
    len_w = len(world_arr)

    one_way_factor = np.sqrt(k_val)

    for s in range(n_sims):
        gross_initial = np.sum(savings_arr[: offset + 1])
        s_initial = gross_initial * np.sqrt(k_val)
        cum_cost = gross_initial - s_initial

        val_w = s_initial * t_w
        val_e = s_initial * t_e

        peak = val_w + val_e
        worst_dd = 0.0
        cum_to_pct = 0.0

        for m in range(n_months):
            num_days = m_lengths[np.random.randint(0, len(m_lengths))]

            for _ in range(num_days):
                idx = np.random.randint(0, len_w)
                val_w *= 1 + world_arr[idx]
                val_e *= 1 + em_arr[idx]

                curr_val = val_w + val_e
                if curr_val > peak:
                    peak = curr_val
                if peak > 0:
                    dd = (curr_val / peak) - 1.0
                    if dd < worst_dd:
                        worst_dd = dd

            gross_monthly = savings_arr[offset + m + 1]
            s_monthly = gross_monthly * np.sqrt(k_val)
            cum_cost += gross_monthly - s_monthly

            total_current = val_w + val_e
            target_w_val = (total_current + s_monthly) * t_w

            add_w = min(max(target_w_val - val_w, 0.0), s_monthly)
            add_e = s_monthly - add_w
            val_w += add_w
            val_e += add_e

            total_val = val_w + val_e
            if total_val > peak:
                peak = total_val

            weight_w = val_w / total_val
            if abs(weight_w - t_w) > reb_thresh:
                pre_rebal_val = val_w + val_e

                if weight_w > t_w:
                    num = val_w - t_w * (val_w + val_e)
                    den = 1 + t_w * (k_val - 1)
                    x = num / den
                    val_w -= x
                    val_e += x * k_val
                else:
                    num = val_e - t_e * (val_w + val_e)
                    den = 1 + t_e * (k_val - 1)
                    x = num / den
                    val_e -= x
                    val_w += x * k_val

                post_rebal_val = val_w + val_e
                cum_cost += pre_rebal_val - post_rebal_val

                traded_amt = x + (x * one_way_factor)
                cum_to_pct += traded_amt / pre_rebal_val

        final_vals[s] = val_w + val_e
        final_mdd[s] = worst_dd
        final_costs[s] = cum_cost
        final_turnover[s] = cum_to_pct
    return final_vals, final_mdd, final_costs, final_turnover


@njit
def run_strat_2(
    horizon_years,
    savings_arr,
    world_arr,
    em_arr,
    gbi_arr,
    m_lengths,
    n_sims,
    k_val,
    t_w,
    t_e,
    t_g,
    reb_thresh,
):
    np.random.seed(42)
    start_age = 65 - horizon_years
    offset = (start_age - 25) * 12
    n_months = horizon_years * 12 - 1
    final_vals = np.zeros(n_sims)
    final_mdd = np.zeros(n_sims)
    final_costs = np.zeros(n_sims)
    final_turnover = np.zeros(n_sims)
    len_pool = len(world_arr)

    one_way_factor = np.sqrt(k_val)

    for s in range(n_sims):
        gross_initial = np.sum(savings_arr[: offset + 1])
        s_initial = gross_initial * np.sqrt(k_val)
        cum_cost = gross_initial - s_initial

        val_w, val_e, val_g = s_initial * t_w, s_initial * t_e, s_initial * t_g

        peak = val_w + val_e + val_g
        worst_dd = 0.0
        cum_to_pct = 0.0

        for m in range(n_months):
            num_days = m_lengths[np.random.randint(0, len(m_lengths))]

            for _ in range(num_days):
                idx = np.random.randint(0, len_pool)
                val_w *= 1 + world_arr[idx]
                val_e *= 1 + em_arr[idx]
                val_g *= 1 + gbi_arr[idx]

                curr_val = val_w + val_e + val_g
                if curr_val > peak:
                    peak = curr_val
                if peak > 0:
                    dd = (curr_val / peak) - 1.0
                    if dd < worst_dd:
                        worst_dd = dd

            gross_monthly = savings_arr[offset + m + 1]
            s_monthly = gross_monthly * np.sqrt(k_val)
            cum_cost += gross_monthly - s_monthly

            total_pre = val_w + val_e + val_g
            tgts = np.array([t_w, t_e, t_g])
            vals = np.array([val_w, val_e, val_g])
            gaps = np.maximum(0.0, tgts * (total_pre + s_monthly) - vals)

            max_gap = np.max(gaps)
            sorted_gaps = np.sort(gaps)
            if s_monthly <= (max_gap - sorted_gaps[1]):
                lvl = max_gap - s_monthly
            else:
                lvl = (np.sum(gaps) - s_monthly) / np.count_nonzero(gaps)

            adjustments = np.maximum(0.0, gaps - lvl)
            val_w += adjustments[0]
            val_e += adjustments[1]
            val_g += adjustments[2]

            total_val = val_w + val_e + val_g
            if total_val > peak:
                peak = total_val

            if (
                (abs(val_w / total_val - t_w) > reb_thresh)
                or (abs(val_e / total_val - t_e) > reb_thresh)
                or (abs(val_g / total_val - t_g) > reb_thresh)
            ):
                old_w, old_e, old_g = val_w, val_e, val_g
                pre_rebal_total = total_val

                src_v, src_t = 0.0, 0.0
                if val_w > t_w * total_val:
                    src_v += val_w
                    src_t += t_w
                if val_e > t_e * total_val:
                    src_v += val_e
                    src_t += t_e
                if val_g > t_g * total_val:
                    src_v += val_g
                    src_t += t_g
                v_new = (total_val - (1 - k_val) * src_v) / (1 - (1 - k_val) * src_t)

                val_w, val_e, val_g = v_new * t_w, v_new * t_e, v_new * t_g

                cum_cost += total_val - (val_w + val_e + val_g)

                gross_sells = 0.0
                if old_w > val_w:
                    gross_sells += old_w - val_w
                if old_e > val_e:
                    gross_sells += old_e - val_e
                if old_g > val_g:
                    gross_sells += old_g - val_g

                total_traded_volume = gross_sells + (gross_sells * one_way_factor)

                cum_to_pct += total_traded_volume / pre_rebal_total

        final_vals[s] = val_w + val_e + val_g
        final_mdd[s] = worst_dd
        final_costs[s] = cum_cost
        final_turnover[s] = cum_to_pct
    return final_vals, final_mdd, final_costs, final_turnover


@njit
def run_strat_3(
    horizon_years,
    savings_arr,
    age_arr,
    world_arr,
    em_arr,
    gbi_arr,
    m_lengths,
    n_sims,
    k_val,
    reb_thresh,
):
    np.random.seed(42)
    start_age_idx = (65 - horizon_years - 25) * 12
    offset = int(start_age_idx)
    n_months = horizon_years * 12 - 1
    final_vals = np.zeros(n_sims)
    final_mdd = np.zeros(n_sims)
    final_costs = np.zeros(n_sims)
    final_turnover = np.zeros(n_sims)
    len_pool = len(world_arr)

    one_way_factor = np.sqrt(k_val)

    for s in range(n_sims):
        curr_age = age_arr[offset]
        t_g = curr_age / 100.0
        t_w, t_e = (1.0 - t_g) * 0.70, (1.0 - t_g) * 0.30

        gross_initial = np.sum(savings_arr[: offset + 1])
        s_initial = gross_initial * np.sqrt(k_val)
        cum_cost = gross_initial - s_initial

        val_w, val_e, val_g = s_initial * t_w, s_initial * t_e, s_initial * t_g

        peak = val_w + val_e + val_g
        worst_dd = 0.0
        cum_to_pct = 0.0

        for m in range(n_months):
            num_days = m_lengths[np.random.randint(0, len(m_lengths))]

            for _ in range(num_days):
                idx = np.random.randint(0, len_pool)
                val_w *= 1 + world_arr[idx]
                val_e *= 1 + em_arr[idx]
                val_g *= 1 + gbi_arr[idx]

                curr_val = val_w + val_e + val_g
                if curr_val > peak:
                    peak = curr_val
                if peak > 0:
                    dd = (curr_val / peak) - 1.0
                    if dd < worst_dd:
                        worst_dd = dd

            curr_age = age_arr[offset + m + 1]
            t_g = curr_age / 100.0
            t_w, t_e = (1.0 - t_g) * 0.70, (1.0 - t_g) * 0.30

            gross_monthly = savings_arr[offset + m + 1]
            s_monthly = gross_monthly * np.sqrt(k_val)
            cum_cost += gross_monthly - s_monthly

            total_pre = val_w + val_e + val_g
            tgts = np.array([t_w, t_e, t_g])
            vals = np.array([val_w, val_e, val_g])
            gaps = np.maximum(0.0, tgts * (total_pre + s_monthly) - vals)

            max_gap = np.max(gaps)
            sorted_gaps = np.sort(gaps)
            if s_monthly <= (max_gap - sorted_gaps[1]):
                lvl = max_gap - s_monthly
            else:
                lvl = (np.sum(gaps) - s_monthly) / np.count_nonzero(gaps)

            adjustments = np.maximum(0.0, gaps - lvl)
            val_w += adjustments[0]
            val_e += adjustments[1]
            val_g += adjustments[2]
            total_val = val_w + val_e + val_g
            if total_val > peak:
                peak = total_val

            if (
                (abs(val_w / total_val - t_w) > reb_thresh)
                or (abs(val_e / total_val - t_e) > reb_thresh)
                or (abs(val_g / total_val - t_g) > reb_thresh)
            ):
                old_w, old_e, old_g = val_w, val_e, val_g
                pre_rebal_total = total_val

                src_v, src_t = 0.0, 0.0
                if val_w > t_w * total_val:
                    src_v += val_w
                    src_t += t_w
                if val_e > t_e * total_val:
                    src_v += val_e
                    src_t += t_e
                if val_g > t_g * total_val:
                    src_v += val_g
                    src_t += t_g
                v_new = (total_val - (1 - k_val) * src_v) / (1 - (1 - k_val) * src_t)
                val_w, val_e, val_g = v_new * t_w, v_new * t_e, v_new * t_g

                cum_cost += total_val - (val_w + val_e + val_g)

                gross_sells = 0.0
                if old_w > val_w:
                    gross_sells += old_w - val_w
                if old_e > val_e:
                    gross_sells += old_e - val_e
                if old_g > val_g:
                    gross_sells += old_g - val_g

                total_traded_volume = gross_sells + (gross_sells * one_way_factor)
                cum_to_pct += total_traded_volume / pre_rebal_total

        final_vals[s] = val_w + val_e + val_g
        final_mdd[s] = worst_dd
        final_costs[s] = cum_cost
        final_turnover[s] = cum_to_pct
    return final_vals, final_mdd, final_costs, final_turnover


@njit
def calc_targets_strat4(fin_val, hc_val, w_to_use):
    target_equity_chf = w_to_use * (fin_val + hc_val)
    E = target_equity_chf / fin_val
    if E <= 1.0:
        return 0.7 * E, 0.3 * E, 1.0 - E, 0.0
    elif E <= 2.0 / 1.3:
        L = E - 1.0
        em = 0.3 * E
        w_asset = 1.0 - em - L
        return w_asset, em, 0.0, L
    elif E <= 2.0:
        L = E - 1.0
        em = 2.0 - E
        return 0.0, em, 0.0, L
    else:
        return 0.0, 0.0, 0.0, 1.0


@njit
def calc_targets_strat5(fin_val, hc_val, w_to_use):
    target_equity_chf = w_to_use * (fin_val + hc_val)
    E = target_equity_chf / fin_val
    E_capped = min(1.0, E)
    return 0.7 * E_capped, 0.3 * E_capped, 1.0 - E_capped, 0.0


@njit
def run_strat_common(
    horizon_years,
    w_current,
    savings_arr,
    hc_arr,
    world_arr,
    em_arr,
    gbi_arr,
    letf_arr,
    m_lengths,
    n_sims,
    k_val,
    reb_thresh,
    strat_type,
):
    np.random.seed(42)
    start_age_idx = (65 - horizon_years - 25) * 12
    offset = int(start_age_idx)
    n_months = horizon_years * 12 - 1
    final_vals = np.zeros(n_sims)
    final_mdd = np.zeros(n_sims)
    final_costs = np.zeros(n_sims)
    final_turnover = np.zeros(n_sims)

    len_pool = len(world_arr)

    one_way_factor = np.sqrt(k_val)

    for s in range(n_sims):
        gross_initial = np.sum(savings_arr[: offset + 1])
        s_initial = gross_initial * np.sqrt(k_val)
        cum_cost = gross_initial - s_initial

        hc_initial = hc_arr[offset]
        if strat_type == 4 or strat_type == 7:
            tw_w, tw_e, tw_g, tw_l = calc_targets_strat4(s_initial, hc_initial, w_current)
        else:
            tw_w, tw_e, tw_g, tw_l = calc_targets_strat5(s_initial, hc_initial, w_current)

        val_w, val_e, val_g, val_l = (
            s_initial * tw_w,
            s_initial * tw_e,
            s_initial * tw_g,
            s_initial * tw_l,
        )

        peak = val_w + val_e + val_g + val_l
        worst_dd = 0.0
        cum_to_pct = 0.0

        for m in range(n_months):
            num_days = m_lengths[np.random.randint(0, len(m_lengths))]

            for _ in range(num_days):
                idx = np.random.randint(0, len_pool)
                val_w *= 1 + world_arr[idx]
                val_e *= 1 + em_arr[idx]
                val_g *= 1 + gbi_arr[idx]
                val_l *= 1 + letf_arr[idx]

                curr_val = val_w + val_e + val_g + val_l
                if curr_val > peak:
                    peak = curr_val
                if peak > 0:
                    dd = (curr_val / peak) - 1.0
                    if dd < worst_dd:
                        worst_dd = dd

            hc_m = hc_arr[offset + m + 1]
            gross_monthly = savings_arr[offset + m + 1]
            s_monthly = gross_monthly * np.sqrt(k_val)
            cum_cost += gross_monthly - s_monthly

            total_pre = val_w + val_e + val_g + val_l
            total_fin_post_cash = total_pre + s_monthly

            if strat_type == 4 or strat_type == 7:
                tw_w, tw_e, tw_g, tw_l = calc_targets_strat4(total_fin_post_cash, hc_m, w_current)
            else:
                tw_w, tw_e, tw_g, tw_l = calc_targets_strat5(total_fin_post_cash, hc_m, w_current)

            if strat_type in [4, 5]:
                tgts = np.array([tw_w, tw_e, tw_g, tw_l])
                vals = np.array([val_w, val_e, val_g, val_l])
                gaps = np.maximum(0.0, tgts * total_fin_post_cash - vals)
                gs = np.sort(gaps)[::-1]
                lvl = gs[0] - s_monthly
                for i in range(1, 4):
                    diff = 0.0
                    for j in range(i):
                        diff += gs[j] - gs[i]
                    if s_monthly > diff:
                        lvl = gs[i] - (s_monthly - diff) / (i + 1)

                val_w += np.maximum(0.0, gaps[0] - lvl)
                val_e += np.maximum(0.0, gaps[1] - lvl)
                val_g += np.maximum(0.0, gaps[2] - lvl)
                val_l += np.maximum(0.0, gaps[3] - lvl)

                total_val = val_w + val_e + val_g + val_l
                if total_val > peak:
                    peak = total_val

                if (
                    abs(val_w / total_val - tw_w) > reb_thresh
                    or abs(val_e / total_val - tw_e) > reb_thresh
                    or abs(val_g / total_val - tw_g) > reb_thresh
                    or abs(val_l / total_val - tw_l) > reb_thresh
                ):
                    old_w, old_e, old_g, old_l = val_w, val_e, val_g, val_l
                    pre_rebal_total = total_val

                    src_v, src_t = 0.0, 0.0
                    if val_w > tw_w * total_val:
                        src_v += val_w
                        src_t += tw_w
                    if val_e > tw_e * total_val:
                        src_v += val_e
                        src_t += tw_e
                    if val_g > tw_g * total_val:
                        src_v += val_g
                        src_t += tw_g
                    if val_l > tw_l * total_val:
                        src_v += val_l
                        src_t += tw_l
                    v_new = (total_val - (1 - k_val) * src_v) / (1 - (1 - k_val) * src_t)
                    val_w, val_e, val_g, val_l = (
                        v_new * tw_w,
                        v_new * tw_e,
                        v_new * tw_g,
                        v_new * tw_l,
                    )

                    cum_cost += total_val - (val_w + val_e + val_g + val_l)

                    gross_sells = 0.0
                    if old_w > val_w:
                        gross_sells += old_w - val_w
                    if old_e > val_e:
                        gross_sells += old_e - val_e
                    if old_g > val_g:
                        gross_sells += old_g - val_g
                    if old_l > val_l:
                        gross_sells += old_l - val_l

                    total_traded_volume = gross_sells + (gross_sells * one_way_factor)
                    cum_to_pct += total_traded_volume / pre_rebal_total

            else:
                val_w += s_monthly * tw_w
                val_e += s_monthly * tw_e
                val_g += s_monthly * tw_g
                val_l += s_monthly * tw_l
                total_val = val_w + val_e + val_g + val_l
                if total_val > peak:
                    peak = total_val

        final_vals[s] = val_w + val_e + val_g + val_l
        final_mdd[s] = worst_dd
        final_costs[s] = cum_cost
        final_turnover[s] = cum_to_pct

    return final_vals, final_mdd, final_costs, final_turnover


def run_simulation_combination(hc_file, return_file):
    print(f"\n{'#' * 70}")
    print(f"PROCESSING: HC={hc_file} | RETURNS={return_file}")
    print(f"{'#' * 70}\n")

    hc_df = pl.read_excel(hc_file)
    returns_df = pl.read_excel(return_file)

    savings_amount = hc_df.select("Savings_Amount").to_series().to_numpy().astype(np.float64)
    hc_array = hc_df.select("Human_Capital").to_series().to_numpy().astype(np.float64)
    age_array = hc_df.select("Age").to_series().to_numpy().astype(np.float64)
    world_pool = returns_df.select("World").to_series().to_numpy().astype(np.float64)
    em_pool = returns_df.select("EM").to_series().to_numpy().astype(np.float64)
    gbi_pool = returns_df.select("GBI").to_series().to_numpy().astype(np.float64)
    letf_pool = returns_df.select(pl.nth(4)).to_series().to_numpy().astype(np.float64)

    month_lengths = (
        returns_df.with_columns(pl.col("Date").dt.truncate("1mo").alias("Month"))
        .group_by("Month")
        .agg(pl.len().alias("days"))
        .sort("Month")
        .select("days")
        .to_series()
        .to_numpy()
        .astype(np.int64)
    )

    n_simulations = 10000
    horizons = [5, 10, 15, 20, 30, 40]
    transaction_costs_list = [0.002, 0.005, 0.010]
    rebalance_threshold = 0.05
    w_risky_pref_list = [0.28120056326605, 1.40600281633025, 2.8120056326605]

    strat1_history = {}

    def print_header(title):
        print(f"\n{'=' * 15} {title} {'=' * 15}")
        print(
            f"{'H':<2} | {'Median':<12} | {'Avg':<12} | {'Std':<12} | {'Skew':<8} | {'Kurt':<8} | {'Min':<10} | {'Max':<10} | {'MDD':<8} | {'TCR':<8} | {'Turn':<8} | {'P(<Inv)':<8} | {'CVaR5%':<10} | {'W5%':<10} | {'B5%':<10}| {'P(>S1)':<8} | {'EU 0.5':<10} | {'EU 1.0':<10} | {'EU 5.0':<10} | {'CE 0.5':<10} | {'CE 1.0':<10} | {'CE 5.0':<12} "
        )
        print("-" * 220)
        # MDD = Maximum Drawdown
        # TCR = Total Cost Ratio
        # Turn = Portfolio Turnover Rate
        # P(<Inv) = Probability of ending up with less capital than invested
        # W5% = Worst 5% of outcomes
        # B5% = Best 5% of outcomes
        # EU 0.5 = Expected Utility at 0.5 Risk Aversion
        # EU 1.0 = Expected Utility at 1.0 Risk Aversion
        # EU 5.0 = Expected Utility at 5.0 Risk Aversion

    def print_results(h, res, mdd_res, cost_res, turnover_res, total_invested, success_prob=None):
        avg_w = np.mean(res)
        std_w = np.std(res, ddof=1)
        sk = skew(res)
        kt = kurtosis(res, fisher=True)
        w5, b5, amin, amax = np.percentile(res, 5), np.percentile(res, 95), np.min(res), np.max(res)

        cvar_5 = np.mean(res[res <= w5])
        med_mdd = np.median(mdd_res)
        tcr_arr = cost_res / res
        med_tcr = np.median(tcr_arr)

        ann_turnover_arr = turnover_res / h
        med_turnover = np.median(ann_turnover_arr)

        prob_under = np.mean(res < total_invested)

        if success_prob is not None:
            sp_str = f"{success_prob * 100:.1f}%"
        else:
            sp_str = "-"

        eu_05 = np.mean((res**0.5) / 0.5)
        eu_10 = np.mean(np.log(res))
        eu_50 = np.mean((res**-4) / -4)

        ce_05 = ((1 - 0.5) * eu_05) ** (1 / (1 - 0.5))
        ce_10 = np.exp(eu_10)
        ce_50 = ((1 - 5.0) * eu_50) ** (1 / (1 - 5.0))

        print(
            f"{h:>2} | {np.median(res):12,.2f} | {avg_w:12,.2f} | {std_w:12,.2f} | {sk:8.2f} | {kt:8.2f} | {amin:10,.2f} | {amax:10,.2f}| {med_mdd:8.2%} | {med_tcr:8.2%} | {med_turnover:8.2%} | {prob_under:8.1%} | {cvar_5:10,.2f} | {w5:10,.2f} | {b5:10,.2f} | {sp_str:8} | {eu_05:10.5f} | {eu_10:10.5f} | {eu_50:10.30f} | {ce_05:10,.2f} | {ce_10:10,.2f} | {ce_50:12,.2f}"
        )

    for s_idx in [1, 2, 3]:
        names = {1: "SP100-R", 2: "SP60-R", 3: "TDF-R"}
        print_header(f"STRATEGY {names[s_idx]}")
        for cost in transaction_costs_list:
            k = (1 - cost) ** 2
            print(f"Cost: {cost * 100:.1f}%")
            for h in horizons:
                start_age_h = 65 - h
                offset_h = int((start_age_h - 25) * 12)
                n_months_h = h * 12 - 1
                total_invested = np.sum(savings_amount[: offset_h + 1 + n_months_h])

                if s_idx == 1:
                    res, mdd, costs, turn = run_strat_1(
                        h,
                        savings_amount,
                        world_pool,
                        em_pool,
                        month_lengths,
                        n_simulations,
                        k,
                        0.7,
                        0.3,
                        rebalance_threshold,
                    )
                    strat1_history[(h, cost)] = res.copy()
                elif s_idx == 2:
                    res, mdd, costs, turn = run_strat_2(
                        h,
                        savings_amount,
                        world_pool,
                        em_pool,
                        gbi_pool,
                        month_lengths,
                        n_simulations,
                        k,
                        0.42,
                        0.18,
                        0.40,
                        rebalance_threshold,
                    )
                else:
                    res, mdd, costs, turn = run_strat_3(
                        h,
                        savings_amount,
                        age_array,
                        world_pool,
                        em_pool,
                        gbi_pool,
                        month_lengths,
                        n_simulations,
                        k,
                        rebalance_threshold,
                    )

                print_results(h, res, mdd, costs, turn, total_invested, None)

    for s_idx in [4, 5, 6, 7]:
        names = {4: "DLP-L-R", 5: "DLP-R", 6: "DLP-BH", 7: "DLP-L-BH"}
        print_header(f"STRATEGY {names[s_idx]}")
        for cost in transaction_costs_list:
            k = (1 - cost) ** 2
            for w in w_risky_pref_list:
                print(f"Cost: {cost * 100:.1f}% | w: {w}")
                for h in horizons:
                    start_age_h = 65 - h
                    offset_h = int((start_age_h - 25) * 12)
                    n_months_h = h * 12 - 1
                    total_invested = np.sum(savings_amount[: offset_h + 1 + n_months_h])

                    res, mdd, costs, turn = run_strat_common(
                        h,
                        w,
                        savings_amount,
                        hc_array,
                        world_pool,
                        em_pool,
                        gbi_pool,
                        letf_pool,
                        month_lengths,
                        n_simulations,
                        k,
                        rebalance_threshold,
                        s_idx,
                    )

                    prob_success = None
                    if s_idx == 4:
                        ref_res = strat1_history.get((h, cost))
                        if ref_res is not None:
                            prob_success = np.sum(res > ref_res) / n_simulations

                    print_results(h, res, mdd, costs, turn, total_invested, prob_success)


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    hc_files = ["HC_Berufslehre.xlsx", "HC_HF.xlsx", "HC_Hochschule.xlsx", "HC_Obl_Schule.xlsx"]
    return_files = ["Return_Data_1.xlsx", "Return_Data_2.xlsx"]
    # Return_Data_1.xlsx uses LETF returns assuming 0.6% TER
    # Return_Data_2.xlsx uses LETF returns assuming 0.8% TER
    # HC_Obl_Schule.xlsx uses income data for education level basic education
    # HC_Berufslehre.xlsx uses income data for education level apprenticeship
    # HC_HF.xlsx uses income data for education level technical college
    # HC_Hochschule.xlsx uses income data for education level university

    for r_file in return_files:
        for h_file in hc_files:
            r_path = os.path.join(BASE_DIR, r_file)
            h_path = os.path.join(BASE_DIR, h_file)

            if os.path.exists(r_path) and os.path.exists(h_path):
                run_simulation_combination(h_path, r_path)
            else:
                print(f"Skipping: {h_file} or {r_file} not found in {BASE_DIR}")

    print("\nALL SIMULATIONS COMPLETED")
