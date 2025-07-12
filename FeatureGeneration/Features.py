import hashlib
import numpy as np
import pandas as pd
import itertools
import os
import gc
from SymmetricVectors import SymmetricVectors


class Features:

    def __init__(self, scheme):
        self.scheme = scheme
        self.grid_shape = scheme.get_grid_shape()
        self.symmetric_vectors_obj = SymmetricVectors(scheme)
        self.symmetric_vectors = self.symmetric_vectors_obj.get_symmetric_vectors()
        self.matrix_names = self.symmetric_vectors_obj.get_matrix_names()

    def get_features(self, chunk_step=500, output_dir=None):
        feature_functions = [
            ('Mean', np.mean),
            ('Std', np.std),
            ('Max', np.max),
            ('Min', np.min),
            ('Osc', np.ptp),
            ("L1", self.l1_norm),
            ("L2", self.l2_norm)
        ]

        patients_data = self.scheme.get_tensors()
        if patients_data.ndim != 4:
            raise ValueError("Expected 4D tensor: [patients, groups, rows, cols]")

        # Статистики матриц
        matrix_stats, matrix_metadata_list = self.compute_matrix_statistics(patients_data)
        matrix_df = pd.DataFrame(matrix_stats)

        # Формируем метаданные для DataFrame
        matrix_metadata = [
            (feature_name, 9, matrix_name)  # Теперь включаем название матрицы
            for feature_name, matrix_name in matrix_metadata_list
        ]

        output_dir = output_dir or "../Features/"
        os.makedirs(output_dir, exist_ok=True)

        feature_metadata = matrix_metadata.copy()

        # Симметричные вектора по чанкам
        sym_vecs = self.symmetric_vectors
        sym_keys = list(sym_vecs.keys())
        total_keys = len(sym_keys)
        chunk_size = total_keys // chunk_step
        remainder = total_keys % chunk_step
        total_chunks = chunk_size + (1 if remainder else 0)

        matrix_path = os.path.join(output_dir, f"features_chunk_{total_chunks}.parquet")
        matrix_df.to_parquet(matrix_path, index=False)

        for chunk_idx in range(total_chunks):
            start = chunk_idx * chunk_step
            end = start + chunk_step if chunk_idx < chunk_size else start + remainder
            current_keys = sym_keys[start:end]

            chunk_df, chunk_metadata = self._process_chunk(
                patients_data,
                {k: sym_vecs[k] for k in current_keys},
                feature_functions
            )

            chunk_path = os.path.join(output_dir, f"features_chunk_{chunk_idx}.parquet")
            chunk_df.to_parquet(chunk_path, index=False)
            feature_metadata.extend(chunk_metadata)

            del chunk_df
            gc.collect()

        # metadata DataFrame
        group_type_descriptions = {
            1: "Per-channel Features f(x)",
            2: "Difference Features f(x - y)",
            3: "Per-channel Difference f(x) - f(y)",
            4: "Cross-function Difference f(x) - g(y)",
            5: "Same-vector Function Difference f(x) - g(x)",
            6: "Matrix Statistic Comparisons",
            7: "Transformed Features",
            8: "Absolute Difference Features f(|x - y|)",
            9: "Matrix Statistic Features"
        }

        group_df = pd.DataFrame(
            [(feat, group, sym_vec, group_type_descriptions.get(group, "Unknown"))
             for feat, group, sym_vec in feature_metadata],
            columns=['Feature', 'GroupType', 'SymmetryVector', 'GroupTypeDescription']
        )
        group_df.to_csv(os.path.join(output_dir, 'feature_groups.csv'), index=False)
        return None

    def compute_matrix_statistics(self, patients_data):
        """Вычисляет статистики для каждой матрицы (канала) и возвращает метаданные"""
        stats_functions = [
            ('Mean', np.mean),
            ('Std', np.std),
            ('Max', np.max),
            ('Min', np.min),
            ('Osc', np.ptp),
            ("L1", self.l1_norm),
            ("L2", self.l2_norm)
        ]

        matrix_stats = {}
        matrix_metadata = []  # Список для хранения метаданных
        n_patients, n_channels, _, _ = patients_data.shape

        for i, name in enumerate(self.matrix_names):
            channel_data = patients_data[:, i, :, :]
            for stat_name, func in stats_functions:
                try:
                    stat_value = func(channel_data, axis=(1, 2))
                except TypeError:
                    stat_value = np.array([func(mat) for mat in channel_data])

                feature_name = f"{name}_{stat_name}"
                matrix_stats[feature_name] = stat_value
                # Добавляем метаданные: имя признака и название матрицы
                matrix_metadata.append((feature_name, name))

        return matrix_stats, matrix_metadata

    @staticmethod
    def l1_norm(x, axis=None):
        return np.abs(x).sum(axis=axis)

    @staticmethod
    def l2_norm(x, axis=None):
        return np.sqrt((x ** 2).sum(axis=axis))

    @staticmethod
    def _hash_array(arr):
        """Хеширует массив значений для проверки уникальности"""
        # Приводим к float32 для единообразия
        arr = arr.astype(np.float32)
        # Преобразуем в байты и хешируем
        return hashlib.sha256(arr.tobytes()).hexdigest()

    @staticmethod
    def _apply_func(arr, func):
        try:
            return func(arr, axis=1)
        except TypeError:
            return np.array([func(row) for row in arr])

    @staticmethod
    def _prepare_indexes(vec):
        if len(vec[0]) == 3:
            g = np.array([p[0] for p in vec], dtype=np.uint16)
        else:
            g = None
        i = np.array([p[1] for p in vec], dtype=np.uint16)
        j = np.array([p[2] for p in vec], dtype=np.uint16)
        return g, i, j

    def _process_chunk(self, dataset, vectors, functions):
        n_patients, n_groups, _, _ = dataset.shape
        all_features = {}
        feature_metadata = []
        existing_value_hashes = set()  # Для отслеживания уникальных значений признаков

        matrix_stats = {
            'mean': dataset.mean(axis=(2, 3)),
            'std': dataset.std(axis=(2, 3)),
            'max': dataset.max(axis=(2, 3)),
            'min': dataset.min(axis=(2, 3)),
            'osc': np.ptp(dataset, axis=(2, 3)),
            'l1': self.l1_norm(dataset, axis=(2, 3)),
            'l2': self.l2_norm(dataset, axis=(2, 3))
        }

        for vec_name, pairs in vectors.items():
            vec1, vec2 = zip(*pairs)
            g1, i1, j1 = self._prepare_indexes(vec1)
            g2, i2, j2 = self._prepare_indexes(vec2)

            patients_idx = np.arange(n_patients)[:, None]
            v1 = dataset[patients_idx, g1, i1, j1]
            v2 = dataset[patients_idx, g2, i2, j2]

            diff = v1 - v2
            abs_diff = np.abs(diff)

            # Выделение V1 и V2 из dot names
            v1_dots = []
            for (m_idx, r, c) in vec1:
                matrix_name = self.matrix_names[m_idx]
                dot_grid = self.symmetric_vectors_obj.get_dot_names(matrix_name)
                v1_dots.append(dot_grid[r][c])

            v2_dots = []
            for (m_idx, r, c) in vec2:
                matrix_name = self.matrix_names[m_idx]
                dot_grid = self.symmetric_vectors_obj.get_dot_names(matrix_name)
                v2_dots.append(dot_grid[r][c])

            v1_unique = sorted(list(set(v1_dots)))
            v2_unique = sorted(list(set(v2_dots)))
            v1_str = ",".join(v1_unique)
            v2_str = ",".join(v2_unique)
            new_vec_name = f"{vec_name} | V1={{{v1_str}}}; V2={{{v2_str}}}"

            self._generate_features(
                original_vec_name=vec_name,
                processed_vec_name=new_vec_name,
                v1=v1, v2=v2, diff=diff, abs_diff=abs_diff,
                matrix_stats=matrix_stats,
                functions=functions,
                all_features=all_features,
                metadata=feature_metadata,
                g1=g1, g2=g2,
                existing_value_hashes=existing_value_hashes
            )

        return pd.DataFrame(all_features), feature_metadata

    def _generate_features(self, original_vec_name, processed_vec_name,
                           v1, v2, diff, abs_diff,
                           matrix_stats, functions,
                           all_features, metadata,
                           g1, g2, existing_value_hashes):
        channels = np.unique(g1) if g1 is not None else [0]

        def _add_feature(feature_name, feature_values):
            # Проверяем уникальность значений признака
            arr_hash = self._hash_array(feature_values)

            if arr_hash in existing_value_hashes:
                return False  # Признак-дубликат не добавляем

            # Проверяем уникальность имени
            base_name = feature_name
            suffix = 1
            while feature_name in all_features:
                suffix += 1
                feature_name = f"{base_name}_dup{suffix}"

            # Добавляем признак
            all_features[feature_name] = feature_values
            existing_value_hashes.add(arr_hash)
            return True

        # Type 1: Per-channel features
        for ch_idx, ch in enumerate(channels):
            ch_mask = (g1 == ch)
            n_points = ch_mask.sum()  # Количество точек в этом канале

            if n_points == 0:
                continue

            matrix_name = self.matrix_names[ch]

            # Для одного элемента - используем просто значение
            if n_points == 1:
                v1_val = v1[:, ch_mask].flatten()
                v2_val = v2[:, ch_mask].flatten()

                feat_name_v1 = f"{processed_vec_name}_V1_{matrix_name}"
                feat_name_v2 = f"{processed_vec_name}_V2_{matrix_name}"

                if _add_feature(feat_name_v1, v1_val):
                    metadata.append((feat_name_v1, 1, original_vec_name))
                if _add_feature(feat_name_v2, v2_val):
                    metadata.append((feat_name_v2, 1, original_vec_name))
            # Для нескольких элементов - применяем статистические функции
            else:
                for fname, func in functions:
                    v1_val = self._apply_func(v1[:, ch_mask], func)
                    v2_val = self._apply_func(v2[:, ch_mask], func)

                    feat_name_v1 = f"{processed_vec_name}_{fname}_V1_{matrix_name}"
                    feat_name_v2 = f"{processed_vec_name}_{fname}_V2_{matrix_name}"

                    if _add_feature(feat_name_v1, v1_val):
                        metadata.append((feat_name_v1, 1, original_vec_name))
                    if _add_feature(feat_name_v2, v2_val):
                        metadata.append((feat_name_v2, 1, original_vec_name))

        # Type 2: f(x - y) → vector_diff
        for fname, func in functions:
            key = f"{processed_vec_name}_{fname}_vector_diff"
            if _add_feature(key, self._apply_func(diff, func)):
                metadata.append((key, 2, original_vec_name))

        # Type 3: f(x)_V1 - f(x)_V2 → funcs_diff
        for fname, func in functions:
            for ch in channels:
                key = f"{processed_vec_name}_{fname}_funcs_diff_ch{ch}"
                # Вычисляем значение без обращения к словарю
                ch_mask = (g1 == ch)
                if np.any(ch_mask):
                    v1_val = self._apply_func(v1[:, ch_mask], func)
                    v2_val = self._apply_func(v2[:, ch_mask], func)
                    value = v1_val - v2_val
                else:
                    value = np.zeros(v1.shape[0])

                if _add_feature(key, value):
                    metadata.append((key, 3, original_vec_name))

        # Type 4: f(x)_V1 - g(y)_V2 → funcs_diff
        for (f1, func1), (f2, func2) in itertools.product(functions, repeat=2):
            if f1 == f2:
                continue
            key = f"{processed_vec_name}_{f1}_V1-{f2}_V2_funcs_diff"
            value = self._apply_func(v1, func1) - self._apply_func(v2, func2)
            if _add_feature(key, value):
                metadata.append((key, 4, original_vec_name))

        # Type 5: f(x) - g(x) on same vector → funcs_diff
        for (f1, func1), (f2, func2) in itertools.product(functions, repeat=2):
            if f1 == f2:
                continue
            key_v1 = f"{processed_vec_name}_{f1}-{f2}_funcs_diff_V1"
            key_v2 = f"{processed_vec_name}_{f1}-{f2}_funcs_diff_V2"
            if _add_feature(key_v1, self._apply_func(v1, func1) - self._apply_func(v1, func2)):
                metadata.append((key_v1, 5, original_vec_name))
            if _add_feature(key_v2, self._apply_func(v2, func1) - self._apply_func(v2, func2)):
                metadata.append((key_v2, 5, original_vec_name))

        # Type 6: Matrix stat comparisons
        for stat_name in matrix_stats:
            stat_values = matrix_stats[stat_name]
            for fname, func in functions:
                for ch in channels:
                    # Вычисляем значения напрямую
                    ch_mask = (g1 == ch)
                    if np.any(ch_mask):
                        v1_val = self._apply_func(v1[:, ch_mask], func)
                        v2_val = self._apply_func(v2[:, ch_mask], func)
                    else:
                        v1_val = np.zeros(v1.shape[0])
                        v2_val = np.zeros(v2.shape[0])

                    key_v1 = f"{processed_vec_name}_Matrix_{stat_name}-{fname}_V1_ch{ch}"
                    key_v2 = f"{processed_vec_name}_Matrix_{stat_name}-{fname}_V2_ch{ch}"
                    if _add_feature(key_v1, stat_values[:, ch] - v1_val):
                        metadata.append((key_v1, 6, original_vec_name))
                    if _add_feature(key_v2, stat_values[:, ch] - v2_val):
                        metadata.append((key_v2, 6, original_vec_name))

        # Type 7: Transformations: function(matrix_stat - vector_elements)
        for stat_name in matrix_stats:
            for vec_type, current_v, current_g in [('V1', v1, g1), ('V2', v2, g2)]:
                selected_stats = matrix_stats[stat_name][:, current_g]
                diff_stat = selected_stats - current_v
                if diff_stat.ndim != 1:
                    for fname, func in functions:
                        feature_name = f"{fname}({processed_vec_name}_{stat_name}_minus_{vec_type})"
                        value = self._apply_func(diff_stat, func)
                        if _add_feature(feature_name, value):
                            metadata.append((feature_name, 7, original_vec_name))
                else:
                    feature_name = f"{processed_vec_name}_{stat_name}_minus_{vec_type}"
                    if _add_feature(feature_name, diff_stat):
                        metadata.append((feature_name, 7, original_vec_name))

        # Type 8: f(|x - y|) → vector_abs_diff
        for fname, func in functions:
            key = f"{processed_vec_name}_{fname}_vector_abs_diff"
            if _add_feature(key, self._apply_func(abs_diff, func)):
                metadata.append((key, 8, original_vec_name))


# Usage example:
# from Scheme import Scheme
#
# if __name__ == "__main__":
#     np.set_printoptions(precision=5, suppress=True)
#     dots = {
#         "inner left": np.array([
#             ["L1", "L2", "L3"],
#             ["L4", "L5", "L6"],
#             ["L7", "L8", "L9"]
#         ]),
#         "skin left": np.array([
#             ["L1K", "L2K", "L3K"],
#             ["L4K", "L5K", "L6K"],
#             ["L7K", "L8K", "L9K"]
#         ]),
#         "inner right": np.array([
#             ["R1", "R2", "R3"],
#             ["R4", "R5", "R6"],
#             ["R7", "R8", "R9"]
#         ]),
#         "skin right": np.array([
#             ["R1K", "R2K", "R3K"],
#             ["R4K", "R5K", "R6K"],
#             ["R7K", "R8K", "R9K"]
#         ]),
#         "support inner left": np.array([["LO"]]),
#         "support skin left": np.array([["LOK"]]),
#         "support inner right": np.array([["RO"]]),
#         "support skin right": np.array([["ROK"]])
#     }
#
#     df = pd.read_excel("../Data/Ovaries/Ovaries.xlsx")
#     ovaries = Scheme(df, dots, (3, 3), True, True, False)
#     print(ovaries.get_tensors()[0])
#     vecs = SymmetricVectors(ovaries)
#     sym_vectors = vecs.get_symmetric_vectors()
#     print(len(sym_vectors), len(ovaries.get_tensors()[0]))
#     features = Features(ovaries)
#     features.get_features(output_dir="../Data/Ovaries/FinalFeatures/")
#
#     dots = {
#         "inner left": np.array([
#             ["L1", "L2", "L3"],
#             ["L4", "L5", "L6"],
#             ["L7", "L8", "L9"]
#         ]),
#         "skin left": np.array([
#             ["кожа L1", "кожа L2", "кожа L3"],
#             ["кожа L4", "кожа L5", "кожа L6"],
#             ["кожа L7", "кожа L8", "кожа L9"]
#         ]),
#         "inner right": np.array([
#             ["R1", "R2", "R3"],
#             ["R4", "R5", "R6"],
#             ["R7", "R8", "R9"]
#         ]),
#         "skin right": np.array([
#             ["кожа R1", "кожа R2", "кожа R3"],
#             ["кожа R4", "кожа R5", "кожа R6"],
#             ["кожа R7", "кожа R8", "кожа R9"]
#         ])
#     }
#
#     df = pd.read_excel("../Data/Brain/BrainDataCRS.xlsx")
#     brain = Scheme(df, dots, (3, 3), True, False, False)
#     print(brain.get_tensors()[0])
#     vecs = SymmetricVectors(brain)
#     sym_vectors = vecs.get_symmetric_vectors()
#     print(len(sym_vectors), len(brain.get_tensors()[0]))
#     features = Features(brain)
#     features.get_features(output_dir="../Data/Brain/FinalFeatures/")
#
#     dots = {"inner left": np.array([
#         np.array(["t28", "t21", "t22"]),
#         np.array(["t27", "t20", "t23"]),
#         np.array(["t24", "t25", "t26"])
#     ]), "inner right": np.array([
#         np.array(["t8", "t1", "t2"]),
#         np.array(["t7", "t0", "t3"]),
#         np.array(["t4", "t5", "t6"])
#     ]), "skin left": np.array([
#         np.array(["t38", "t31", "t32"]),
#         np.array(["t37", "t30", "t33"]),
#         np.array(["t34", "t35", "t36"])
#     ]), "skin right": np.array([
#         np.array(["t18", "t11", "t12"]),
#         np.array(["t17", "t10", "t13"]),
#         np.array(["t14", "t15", "t16"])
#     ]),
#         "support inner left": np.array([["t29"]]),
#         "support skin left": np.array([["t9"]]),
#         "support inner right": np.array([["t39"]]),
#         "support skin right": np.array([["t19"]]),
#         "central inner support dots": np.array([["t40", "t41"]]),
#         "central skin support dots": np.array([["t42", "t43"]])
#     }
#
#     df = pd.read_excel("../Data/MammaryGlands/MGBothOrgans.xlsx")
#     MG = Scheme(df, dots, (3, 3), True, True, True)
#     print(MG.get_tensors()[0])
#     vecs = SymmetricVectors(MG)
#     sym_vectors = vecs.get_symmetric_vectors()
#     print(len(sym_vectors), len(MG.get_tensors()[0]))
#     features = Features(MG)
#     features.get_features(output_dir="../Data/MammaryGlands/FinalFeatures/")
