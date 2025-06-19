import pandas as pd
import numpy as np
from scipy.special import gammaln
from sklearn.model_selection import train_test_split
import time


class IntervalsLabel:

    def __init__(self, MGs, df_for_func, test_size=0.2, random_state=42):
        self.MGs_set = set(MGs)
        self.df_for_func = df_for_func.copy()
        self.test_size = test_size
        self.random_state = random_state
        self.train_df = df_for_func.copy()
        self.test_df = pd.DataFrame()
        self.df_intervals = pd.DataFrame()
        self.th_col = None
        self.label_col = None
        if test_size is not None:
            self.split_data()

    def split_data(self):
        if self.label_col:
            stratify = self.df_for_func[self.label_col]
        else:
            stratify = None
        self.train_df, self.test_df = train_test_split(
            self.df_for_func,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )

    @staticmethod
    def get_data_for_intervals(df_for_func, th_col, label_col, type1_health_start=None, type2_health_start=None,
                               GMTh_compare_val=False):
        if GMTh_compare_val:
            df_res = pd.DataFrame(
                columns=['MG', 'delta', 'start_point', 'left_interval_limit', 'right_interval_limit', 'label 0',
                         'label 1', 'informativeness', 'informativeness_kind', 'GMTh'])
        else:
            df_res = pd.DataFrame(
                columns=['MG', 'delta', 'start_point', 'left_interval_limit', 'right_interval_limit', 'label 0',
                         'label 1', 'informativeness', 'informativeness_kind'])

        if type1_health_start:
            df_res = find_intervals(df_for_func, 1, type1_health_start, df_res, th_col, label_col, ['GML'],
                                    GMTh_compare_val)
        if type2_health_start:
            df_res = find_intervals(df_for_func, 2, type2_health_start, df_res, th_col, label_col, ['GML'],
                                    GMTh_compare_val)

        df_res = df_res.sort_values(by=['MG']).reset_index(drop=True)
        return df_res

    @staticmethod
    def create_dict_from_features_value(features_list, val):
        return {feature: val for feature in features_list}

    @staticmethod
    def create_dict_from_features(features_list, df_for_func):
        return {feature: df_for_func[feature].median() for feature in features_list}

    def get_intervals(self, label_col, th_col, type1_features_equal, type1_features_MG, type1_features_opposite,
                      type2_features_diff, type2_features_MG_0, type2_features_MG_medi):
        self.th_col = th_col
        self.label_col = label_col
        if self.test_size is not None and self.test_df.empty:
            self.split_data()

        self.df_intervals = pd.DataFrame()
        df_label = self.train_df[self.train_df[label_col] == 1]

        type1_features_equal = list(set(type1_features_equal) & self.MGs_set)
        type1_features_MG = list(set(type1_features_MG) & self.MGs_set)
        type1_features_opposite = list(set(type1_features_opposite) & self.MGs_set)
        type2_features_diff = list(set(type2_features_diff) & self.MGs_set)
        type2_features_MG_0 = list(set(type2_features_MG_0) & self.MGs_set)
        type2_features_MG_medi = list(set(type2_features_MG_medi) & self.MGs_set)

        type1_start = self.create_dict_from_features_value(type1_features_MG, 0)
        type1_start_equal = self.create_dict_from_features_value(type1_features_equal, 0)
        type1_start_opposite = self.create_dict_from_features_value(type1_features_opposite, 0)

        type2_start = self.create_dict_from_features_value(type2_features_diff + type2_features_MG_0, 0)
        type2_start_medi = self.create_dict_from_features(type2_features_MG_medi, df_label)
        type2_start.update(type2_start_medi)

        for feature_opp in type1_features_opposite:
            df_for_func_opposite = self.train_df[self.train_df[feature_opp] >= 0]
            type_start = {feature_opp: self.train_df[feature_opp].median()}
            df_res_opposite_f = self.get_data_for_intervals(df_for_func_opposite, th_col, label_col,
                                                            type1_health_start=type_start, GMTh_compare_val=False)
            df_res_opposite_f['left_interval_limit'] = -df_res_opposite_f['right_interval_limit']
            self.df_intervals = pd.concat([self.df_intervals, df_res_opposite_f])

        df_for_func_equal = self.train_df[~self.train_df.duplicated(['protocol_id'])]
        df_res_equal = self.get_data_for_intervals(df_for_func_equal, th_col, label_col,
                                                   type1_health_start=type1_start_equal, GMTh_compare_val=False)
        df_res = self.get_data_for_intervals(self.train_df, th_col, label_col,
                                             type1_health_start=type1_start, type2_health_start=type2_start,
                                             GMTh_compare_val=False)

        self.df_intervals = pd.concat([self.df_intervals, df_res_equal, df_res]).sort_values(by=['informativeness'],
                                                                                             ascending=False).reset_index(
            drop=True)

        if not self.test_df.empty:
            self._calculate_test_informativeness()

        return self.df_intervals

    def _calculate_test_informativeness(self):
        # Precompute totals for labels and thresholds
        total_label0 = (self.test_df[self.label_col] == 0).sum()
        total_label1 = (self.test_df[self.label_col] == 1).sum()
        th_total = self.test_df[self.th_col].value_counts().to_dict()
        unique_ths = list(th_total.keys())
        n_ths = len(unique_ths)

        # Precompute threshold indicators for all test samples
        ths = self.test_df[self.th_col].values
        th_indicators = np.zeros((len(ths), n_ths), dtype=bool)
        for i, th in enumerate(unique_ths):
            th_indicators[:, i] = (ths == th)
        th_indicators_int = th_indicators.astype(int)

        # Initialize test_informativeness with NaNs
        test_inf = [np.nan] * len(self.df_intervals)

        # Group intervals by MG for batch processing
        grouped = self.df_intervals.groupby('MG')

        for mg, intervals_group in grouped:
            # Original indices of these intervals in df_intervals
            indices = intervals_group.index.tolist()
            intervals_data = intervals_group[
                ['left_interval_limit', 'right_interval_limit', 'informativeness_kind']].values
            n_intervals = len(indices)

            # Extract MG values from test data
            mg_values = self.test_df[mg].values

            # Create masks for all intervals in this MG using broadcasting
            lefts = intervals_data[:, 0]
            rights = intervals_data[:, 1]
            masks = (mg_values[:, np.newaxis] >= lefts) & (mg_values[:, np.newaxis] <= rights)
            masks_int = masks.astype(int)

            # Compute label counts using matrix multiplication
            label0 = (self.test_df[self.label_col] == 0).values.astype(int)
            label1 = (self.test_df[self.label_col] == 1).values.astype(int)
            label0_counts = np.dot(label0, masks_int)
            label1_counts = np.dot(label1, masks_int)

            # Compute label proportions safely avoiding division by zero
            label0_props = np.divide(label0_counts, total_label0, out=np.zeros_like(label0_counts, dtype=float),
                                     where=total_label0 != 0)
            label1_props = np.divide(label1_counts, total_label1, out=np.zeros_like(label1_counts, dtype=float),
                                     where=total_label1 != 0)

            # Compute threshold counts and proportions
            th_counts = np.dot(masks_int.T, th_indicators_int)  # (n_intervals, n_ths)
            th_props_per_interval = []
            for j in range(n_intervals):
                props = {}
                for i, th in enumerate(unique_ths):
                    total = th_total.get(th, 0)
                    count = th_counts[j, i]
                    props[th] = count / total if total else 0
                th_props_per_interval.append(props)

            # Calculate informativeness for each interval
            test_inf_mg = []
            for j in range(n_intervals):
                inf_kind = intervals_data[j, 2]
                label0_p = label0_props[j]
                label1_p = label1_props[j]
                th_props = th_props_per_interval[j]

                if inf_kind == 'GML':
                    inf = np.sqrt((1 - label0_p) * label1_p)
                elif inf_kind == 'GMTh':
                    product = 1.0
                    for th, prop in th_props.items():
                        product *= prop if th >= 3 else (1 - prop)
                    inf = product ** (1 / len(th_props)) if th_props else 0
                elif inf_kind == 'SL':
                    k0 = label0_counts[j]
                    k1 = label1_counts[j]
                    log_res = (gammaln(total_label0 + 1) - gammaln(k0 + 1) - gammaln(total_label0 - k0 + 1) +
                               gammaln(total_label1 + 1) - gammaln(k1 + 1) - gammaln(total_label1 - k1 + 1))
                    total_n = total_label0 + total_label1
                    total_p = k0 + k1
                    log_comb_total = gammaln(total_n + 1) - gammaln(total_p + 1) - gammaln(total_n - total_p + 1)
                    inf = - (log_res - log_comb_total)
                else:
                    inf = 0.0
                test_inf_mg.append(inf)

            # Update test_inf with the computed values
            for i, idx in enumerate(indices):
                test_inf[idx] = test_inf_mg[i]

        # Assign the computed values to the DataFrame
        self.df_intervals['test_informativeness'] = test_inf

    @staticmethod
    def calculate_informativeness(df, MG, left, right, inf_kind, th_col, label_col):
        subset = df[(df[MG] >= left) & (df[MG] <= right)]
        if subset.empty:
            return 0.0

        total_label0 = (df[label_col] == 0).sum()
        total_label1 = (df[label_col] == 1).sum()
        total_ths = df[th_col].value_counts().to_dict()

        label0_count = (subset[label_col] == 0).sum()
        label1_count = (subset[label_col] == 1).sum()

        label0_prop = label0_count / total_label0 if total_label0 else 0
        label1_prop = label1_count / total_label1 if total_label1 else 0

        th_props = {}
        for th in total_ths:
            th_count = (subset[th_col] == th).sum()
            th_props[th] = th_count / total_ths[th] if total_ths[th] else 0

        if inf_kind == 'GML':
            return np.sqrt((1 - label0_prop) * label1_prop)
        elif inf_kind == 'GMTh':
            product = 1.0
            for th, prop in th_props.items():
                product *= prop if th >= 3 else (1 - prop)
            return product ** (1 / len(th_props)) if th_props else 0
        else:
            log_res = 0.0
            mg_counts = {
                'labels': {0: label0_count, 1: label1_count},
                'ths': {th: th_count for th, th_count in th_props.items()}
            }
            if inf_kind == 'SL':
                dct = {'0': total_label0, '1': total_label1}
                for key in ['0', '1']:
                    n = dct[key]
                    k = mg_counts['labels'].get(int(key), 0)
                    log_res += gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                total_n = total_label0 + total_label1
                total_p = label0_count + label1_count
                log_comb_total = gammaln(total_n + 1) - gammaln(total_p + 1) - gammaln(total_n - total_p + 1)
                log_res -= log_comb_total
                return -log_res
            else:
                return 0.0


def find_intervals(df, interval_type, type_health_start, df_res, th_col, label_col, inf_kinds,
                   GMTh_compare_value=False):
    y_data = df[label_col].values
    ths_data = df[th_col].values
    total_label0 = (y_data == 0).sum()
    total_label1 = (y_data == 1).sum()
    for MG in type_health_start.keys():
        start_time = time.time()
        mg_data = df[MG].values

        mg_min = np.min(mg_data)
        mg_max = np.max(mg_data)
        range_mg = mg_max - mg_min
        if range_mg <= 0:
            continue

        delta_percentages = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        delta_values = [p * range_mg for p in delta_percentages]

        sorted_idx = np.argsort(mg_data)
        x_sorted = mg_data[sorted_idx]
        y_sorted = y_data[sorted_idx]
        ths_sorted = ths_data[sorted_idx]

        cum_label0 = np.cumsum(y_sorted == 0)
        cum_label1 = np.cumsum(y_sorted == 1)
        unique_ths, ths_indices = np.unique(ths_sorted, return_inverse=True)
        cum_ths = {th: np.cumsum(ths_sorted == th) for th in unique_ths}

        total_ths = {th: (ths_data == th).sum() for th in unique_ths}

        best_rows = []
        for delta in delta_values:
            if interval_type == 1:
                interval = IntervalType1(delta, x_sorted, ths_sorted, y_sorted, cum_label0, cum_label1, cum_ths,
                                         total_label0, total_label1, total_ths, MG, type_health_start[MG], inf_kinds[0])
            else:
                interval = IntervalType2(delta, x_sorted, ths_sorted, y_sorted, cum_label0, cum_label1, cum_ths,
                                         total_label0, total_label1, total_ths, MG, type_health_start[MG], inf_kinds[0])

            res = interval.find()
            if not res[0]:
                continue

            row = {
                'MG': MG, 'delta': delta, 'start_point': type_health_start[MG],
                'left_interval_limit': res[0][0], 'right_interval_limit': res[0][1],
                'label 0': res[2][0], 'label 1': res[2][1],
                'informativeness': res[1], 'informativeness_kind': res[3]
            }
            if GMTh_compare_value:
                row['GMTh'] = res[4]
                if row['GMTh'] < GMTh_compare_value:
                    continue
            best_rows.append(row)

        if best_rows:
            best_df = pd.DataFrame(best_rows).sort_values('informativeness', ascending=False).head(1)
            df_res = pd.concat([df_res, best_df], ignore_index=True)
    return df_res


class IntervalType1:
    def __init__(self, delta, x_sorted, ths_sorted, y_sorted, cum_label0, cum_label1, cum_ths,
                 total_label0, total_label1, total_ths, MG, start_val, inf_kind):
        self.delta = delta
        self.x_sorted = x_sorted
        self.ths_sorted = ths_sorted
        self.y_sorted = y_sorted
        self.cum_label0 = cum_label0
        self.cum_label1 = cum_label1
        self.cum_ths = cum_ths
        self.total_label0 = total_label0
        self.total_label1 = total_label1
        self.total_ths = total_ths
        self.MG = MG
        self.start_val = start_val
        self.inf_kind = inf_kind

        self.left_limit = self.start_val
        self.right_limit = self.start_val
        self.prev_eff = -1
        self.prev_interval = []
        self.prev_prop = {0: 0, 1: 0}
        self.th_prev_prop = {k: 0 for k in self.total_ths}

    def find(self):
        while True:
            current_eff = self._compute_effectiveness()
            if current_eff <= self.prev_eff and self.prev_interval:
                return self.prev_interval, self.prev_eff, self.prev_prop, self.inf_kind, self._compute_gmth()
            self.prev_eff = current_eff
            self.prev_interval = [self.left_limit, self.right_limit]
            self.prev_prop = {0: self.label0_prop, 1: self.label1_prop}
            self.th_prev_prop = {th: self.th_props.get(th, 0) for th in self.total_ths}
            self.right_limit += self.delta

    def _compute_effectiveness(self):
        left_idx = np.searchsorted(self.x_sorted, self.left_limit, side='left')
        right_idx = np.searchsorted(self.x_sorted, self.right_limit, side='right')

        label0_count = self.cum_label0[right_idx - 1] - (self.cum_label0[left_idx - 1] if left_idx > 0 else 0)
        label1_count = self.cum_label1[right_idx - 1] - (self.cum_label1[left_idx - 1] if left_idx > 0 else 0)
        self.label0_prop = label0_count / self.total_label0 if self.total_label0 else 0
        self.label1_prop = label1_count / self.total_label1 if self.total_label1 else 0

        self.th_props = {}
        for th in self.total_ths:
            th_count = self.cum_ths[th][right_idx - 1] - (self.cum_ths[th][left_idx - 1] if left_idx > 0 else 0)
            self.th_props[th] = th_count / self.total_ths[th] if self.total_ths[th] else 0

        if self.inf_kind == 'GML':
            return ((1 - self.label0_prop) * self.label1_prop) ** 0.5
        elif self.inf_kind == 'GMTh':
            product = 1.0
            for th, prop in self.th_props.items():
                product *= prop if th >= 3 else (1 - prop)
            return product ** (1 / len(self.th_props))
        else:
            return self._calc_stat_inf()

    def _calc_stat_inf(self):
        log_res = 0.0
        mg_counts = {'labels': {0: self.label0_prop * self.total_label0, 1: self.label1_prop * self.total_label1},
                     'ths': self.th_props}
        dct = mg_counts['labels'] if self.inf_kind == 'SL' else self.total_ths
        mg_dct = {k: int(v * dct[k]) for k, v in (mg_counts['labels' if self.inf_kind == 'SL' else 'ths'].items())}

        for key in mg_dct:
            n = dct[key]
            k = mg_dct[key]
            log_res += gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

        total_n = sum(dct.values())
        total_p = sum(mg_dct.values())
        log_comb_total = gammaln(total_n + 1) - gammaln(total_p + 1) - gammaln(total_n - total_p + 1)
        log_res -= log_comb_total
        return -log_res

    def _compute_gmth(self):
        product = 1.0
        for th, prop in self.th_props.items():
            product *= prop if th >= 3 else (1 - prop)
        return product ** (1 / len(self.th_props)) if self.th_props else 0


class IntervalType2(IntervalType1):
    def __init__(self, delta, x_sorted, ths_sorted, y_sorted, cum_label0, cum_label1, cum_ths,
                 total_label0, total_label1, total_ths, MG, start_val, inf_kind):
        super().__init__(delta, x_sorted, ths_sorted, y_sorted, cum_label0, cum_label1, cum_ths,
                         total_label0, total_label1, total_ths, MG, start_val, inf_kind)
        self.center = start_val
        self.left_limit = self.center
        self.right_limit = self.center

    def find(self):
        while True:
            current_eff = self._compute_effectiveness()
            if current_eff <= self.prev_eff and self.prev_interval:
                return self.prev_interval, self.prev_eff, self.prev_prop, self.inf_kind, self._compute_gmth()
            self.prev_eff = current_eff
            self.prev_interval = [self.left_limit, self.right_limit]
            self.prev_prop = {0: self.label0_prop, 1: self.label1_prop}
            self.th_prev_prop = {th: self.th_props.get(th, 0) for th in self.total_ths}
            self.left_limit -= self.delta
            self.right_limit += self.delta
