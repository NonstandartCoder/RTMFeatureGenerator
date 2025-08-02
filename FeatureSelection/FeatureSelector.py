import numpy as np
from scipy import stats
import os
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
import time
import json
from collections import defaultdict
import warnings

# Игнорируем предупреждения
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="theano")
warnings.filterwarnings("ignore", category=UserWarning, module="pymc")
warnings.filterwarnings("ignore", category=FutureWarning)


class FeatureSelector:
    def __init__(self, base_path, organ, target_path, target_col, random_state=42):
        self.base_path = base_path
        self.organ = organ
        self.target_path = target_path
        self.target_col = target_col
        self.random_state = random_state
        self.feature_groups = None
        self.feature_data = None
        self.informativeness_data = None
        self.selected_features = []
        self.target_data = None
        self.validation_report = {}
        self.evaluation_report = {}

        # Создаем папку для результатов
        self.classification_results_dir = os.path.join(
            base_path, "Data", organ, "ClassificationResults"
        )
        os.makedirs(self.classification_results_dir, exist_ok=True)

        # Параметры для настройки
        self.stability_threshold = 0.5
        self.corr_threshold = 0.7
        self.variance_threshold = 0.01
        self.test_drop_threshold = 0.2
        self.top_k = 70

    def load_data(self):
        """Загрузка всех необходимых данных"""
        print("Загрузка данных...")

        # 1. Загрузка метаданных признаков
        meta_path = os.path.join(
            self.base_path,
            "FinalFeatureGeneration",
            "Data",
            self.organ,
            "FinalFeatures",
            "feature_groups.csv"
        )
        if os.path.exists(meta_path):
            self.feature_groups = pd.read_csv(meta_path)
        else:
            print(f"Предупреждение: Файл метаданных не найден {meta_path}")
            self.feature_groups = pd.DataFrame(columns=['Feature', 'SymmetryVector'])

        # 2. Загрузка данных признаков
        chunk_range = {
            'Brain': range(0, 3),
            'Ovaries': range(0, 3),
            'MammaryGlands': range(0, 5)
        }.get(self.organ, range(0, 3))

        feature_dfs = []
        for chunk_num in chunk_range:
            chunk_path = os.path.join(
                self.base_path,
                "FinalFeatureGeneration",
                "Data",
                self.organ,
                "FinalFeatures",
                f"features_chunk_{chunk_num}.parquet"
            )
            if os.path.exists(chunk_path):
                try:
                    df = pd.read_parquet(chunk_path)
                    feature_dfs.append(df)
                except Exception as e:
                    print(f"Ошибка при чтении {chunk_path}: {str(e)}")
            else:
                print(f"Предупреждение: Файл не найден {chunk_path}")

        self.feature_data = pd.concat(feature_dfs, axis=1) if feature_dfs else pd.DataFrame()

        # 3. Загрузка целевой переменной
        if os.path.exists(self.target_path):
            try:
                # Определяем формат файла по расширению
                if self.target_path.endswith('.xlsx'):
                    self.target_data = pd.read_excel(self.target_path)
                elif self.target_path.endswith('.csv'):
                    self.target_data = pd.read_csv(self.target_path)
                elif self.target_path.endswith('.parquet'):
                    self.target_data = pd.read_parquet(self.target_path)
                else:
                    # Пробуем автоматически определить формат
                    try:
                        self.target_data = pd.read_excel(self.target_path)
                    except:
                        try:
                            self.target_data = pd.read_csv(self.target_path)
                        except:
                            self.target_data = pd.read_parquet(self.target_path)

                # Выравнивание индексов
                if not self.target_data.index.equals(self.feature_data.index):
                    print("Выравнивание индексов признаков и целевой переменной...")
                    common_idx = self.feature_data.index.intersection(self.target_data.index)
                    self.feature_data = self.feature_data.loc[common_idx]
                    self.target_data = self.target_data.loc[common_idx]

                # Добавляем целевую переменную
                self.feature_data[self.target_col] = self.target_data[self.target_col]
            except Exception as e:
                print(f"Ошибка при загрузке целевой переменной: {str(e)}")
                raise
        else:
            raise FileNotFoundError(f"Файл целевой переменной не найден: {self.target_path}")

        # 4. Загрузка информативности признаков (если есть)
        informativeness_dfs = []
        processed_range = range(0, 2)

        for chunk_num in chunk_range:
            for processed_chunk in processed_range:
                for reversed_type in ['_', '_reversed_']:
                    info_path = os.path.join(
                        self.base_path,
                        "FinalFeatureGeneration",
                        "Data",
                        self.organ,
                        "FinalFeatureInformativenessTrainTestSplit",
                        f"file_chunk_{chunk_num}_processed_chunk_{processed_chunk}{reversed_type}ocls.xlsx"
                    )
                    if os.path.exists(info_path):
                        try:
                            df = pd.read_excel(info_path)
                            # Стандартизация названий столбцов
                            if 'Feature' in df.columns and 'MG' not in df.columns:
                                df = df.rename(columns={'Feature': 'MG'})
                            informativeness_dfs.append(df)
                        except Exception as e:
                            print(f"Ошибка при чтении файла информативности {info_path}: {str(e)}")
                    else:
                        pass  # Не выводим предупреждение для отсутствующих файлов

        if informativeness_dfs:
            self.informativeness_data = pd.concat(informativeness_dfs, ignore_index=True)
            # Удаляем дубликаты
            self.informativeness_data = self.informativeness_data.drop_duplicates(subset='MG')
        else:
            print("Предупреждение: Данные информативности не найдены")
            self.informativeness_data = pd.DataFrame(columns=['MG', 'informativeness'])

        print("Данные успешно загружены")
        print(f"Загружено признаков: {self.feature_data.shape[1] - 1}")
        print(f"Загружено образцов: {self.feature_data.shape[0]}")
        return self

    def filter_low_variance(self, threshold=None):
        """Фильтрация признаков с низкой дисперсией"""
        if threshold is None:
            threshold = self.variance_threshold

        print("\nФильтрация по дисперсии...")
        feature_cols = [col for col in self.feature_data.columns if col != self.target_col]

        if not feature_cols:
            print("Нет признаков для фильтрации!")
            self.selected_features = []
            return []

        try:
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(self.feature_data[feature_cols])
            retained_features = np.array(feature_cols)[selector.get_support()].tolist()
        except Exception as e:
            print(f"Ошибка при фильтрации дисперсии: {str(e)}")
            retained_features = feature_cols

        # Обновляем данные
        retained_features.append(self.target_col)
        self.feature_data = self.feature_data[retained_features]

        # Обновляем информативность
        if not self.informativeness_data.empty:
            self.informativeness_data = self.informativeness_data[
                self.informativeness_data['MG'].isin(retained_features)
            ]

        # Обновляем selected_features
        self.selected_features = [f for f in retained_features if f != self.target_col]
        print(f"Оставлено {len(self.selected_features)} признаков после фильтрации дисперсии")
        return self.selected_features

    def select_best_per_symmetry_group(self, test_drop_threshold=None):
        """Выбор лучшего признака для каждой группы симметрии"""
        if test_drop_threshold is None:
            test_drop_threshold = self.test_drop_threshold

        print("\nВыбор лучших признаков по группам симметрии...")

        # Если данные информативности пусты, возвращаем все признаки
        if self.informativeness_data.empty:
            print("Данные информативности отсутствуют, возвращаем все признаки")
            self.selected_features = [col for col in self.feature_data.columns if col != self.target_col]
            return self.selected_features

        # Объединяем данные
        merged_data = pd.merge(
            self.informativeness_data,
            self.feature_groups[['Feature', 'SymmetryVector']],
            left_on='MG',
            right_on='Feature',
            how='left'
        )

        # Если нет информации о симметрии, возвращаем все признаки
        if 'SymmetryVector' not in merged_data.columns or merged_data['SymmetryVector'].isna().all():
            print("Информация о симметрии отсутствует, возвращаем все признаки")
            self.selected_features = merged_data['MG'].unique().tolist()
            return self.selected_features

        best_features = []

        # Для каждой группы симметрии
        for group, group_data in merged_data.groupby('SymmetryVector'):
            # Сортируем по информативности
            group_data = group_data.sort_values('informativeness', ascending=False)

            # Выбираем лучший признак
            for _, row in group_data.iterrows():
                train_info = row['informativeness']
                test_info = row.get('test_informativeness', train_info)

                # Проверка на переобучение
                if (train_info - test_info) / train_info <= test_drop_threshold:
                    best_features.append(row['MG'])
                    break

        print(f"Отобрано {len(best_features)} признаков по группам симметрии")
        self.selected_features = best_features
        return best_features

    def filter_by_correlation(self, corr_threshold=None):
        """Фильтрация по корреляции"""
        if corr_threshold is None:
            corr_threshold = self.corr_threshold

        print("\nФильтрация по корреляции...")

        if not self.selected_features:
            print("Нет признаков для фильтрации")
            return []

        # Создаем словарь информативности
        info_map = {}
        for feature in self.selected_features:
            feat_data = self.informativeness_data[self.informativeness_data['MG'] == feature]
            if not feat_data.empty:
                info_map[feature] = feat_data['informativeness'].iloc[0]
            else:
                # Если признак не найден, используем случайное значение
                info_map[feature] = np.random.rand()
                print(f"Предупреждение: Признак {feature} отсутствует в данных информативности")

        # Сортируем признаки по информативности
        sorted_features = sorted(
            self.selected_features,
            key=lambda x: info_map.get(x, 0),
            reverse=True
        )

        # Вычисляем матрицу корреляций
        try:
            corr_matrix = self.feature_data[sorted_features].corr().abs()
        except Exception as e:
            print(f"Ошибка при вычислении корреляции: {str(e)}")
            return sorted_features

        # Фильтрация коррелирующих признаков
        retained_features = []
        dropped_features = set()

        for feature in sorted_features:
            if feature in dropped_features:
                continue

            retained_features.append(feature)

            # Находим коррелирующие признаки
            if feature in corr_matrix.columns:
                correlated = corr_matrix.index[
                    (corr_matrix[feature] > corr_threshold) &
                    (corr_matrix[feature] < 1.0)
                    ].tolist()
                dropped_features.update(correlated)

        print(f"Оставлено {len(retained_features)} признаков после фильтрации корреляции")
        self.selected_features = retained_features
        return retained_features

    def statistical_validation(self, n_random_features=20, stability_threshold=None, top_k=None):
        """Статистическая валидация признаков"""
        if stability_threshold is None:
            stability_threshold = self.stability_threshold
        if top_k is None:
            top_k = self.top_k

        print("\nСтатистическая валидация признаков...")

        # Если нет признаков для валидации
        if not self.selected_features:
            print("Нет признаков для валидации, возвращаем пустой список")
            self.validation_report = {
                'selected_features': [],
                'fdr': 0.0,
                'stability_scores': {}
            }
            self.selected_features = []
            return []

        # Добавляем случайные признаки
        data = self.feature_data[self.selected_features + [self.target_col]].copy()
        random_features = []

        for i in range(n_random_features):
            rand_col = f'random_{i}'
            data[rand_col] = np.random.normal(size=len(data))
            random_features.append(rand_col)

        # Оценка стабильности
        stability_scores = self._stability_selection(
            data=data,
            features=self.selected_features + random_features,
            target=self.target_col,
            n_iter=50,
            top_k=top_k * 2
        )

        # Отбор кандидатов
        candidate_features = [
                                 f for f in self.selected_features + random_features
                                 if stability_scores.get(f, 0) >= stability_threshold
                             ][:top_k * 2]

        # Байесовская верификация
        try:
            validated_features = self._bayesian_validation(
                data=data,
                features=[f for f in candidate_features if f not in random_features],
                target=self.target_col
            )
        except Exception as e:
            print(f"Ошибка при байесовской верификации: {str(e)}")
            # Fallback: используем простой отбор по стабильности
            validated_features = [
                                     f for f in candidate_features
                                     if f not in random_features and stability_scores.get(f, 0) >= stability_threshold
                                 ][:top_k]

        # Оценка FDR
        n_random_selected = sum(1 for f in validated_features if f in random_features)
        fdr = n_random_selected / max(1, len(validated_features))

        # Сохраняем результаты
        self.validation_report = {
            'selected_features': validated_features,
            'fdr': fdr,
            'stability_scores': stability_scores
        }

        # Обновляем selected_features
        self.selected_features = validated_features

        print(f"Финальный отбор: {len(self.selected_features)} признаков")
        print(f"Оценка FDR: {fdr:.3f}")
        return self.selected_features

    @staticmethod
    def _stability_selection(data, features, target, n_iter=50, top_k=30):
        """Оценка стабильности признаков"""
        print("Оценка стабильности...")
        stability_scores = {f: 0 for f in features}

        def _stability_iteration(iter_idx):
            np.random.seed(iter_idx)
            bootstrap_idx = np.random.choice(len(data), size=int(len(data) * 0.8), replace=True)
            df_sample = data.iloc[bootstrap_idx]

            # Упрощенная модель
            model = RandomForestClassifier(n_estimators=50, max_depth=3, n_jobs=-1, random_state=iter_idx)
            model.fit(df_sample[features], df_sample[target])

            importances = pd.Series(model.feature_importances_, index=features)
            return importances.nlargest(top_k).index.tolist()

        # Параллельный запуск
        results = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(_stability_iteration)(i) for i in tqdm(range(n_iter), desc="Stability Selection")
        )

        # Агрегация результатов
        for feature_list in results:
            for feature in feature_list:
                stability_scores[feature] += 1 / n_iter

        return stability_scores

    @staticmethod
    def _bayesian_validation(data, features, target):
        """Альтернативная валидация с использованием t-теста"""
        if not features:
            return []

        print("Валидация с использованием t-теста...")
        significant_features = []

        # Разделяем данные по классам
        class0 = data[data[target] == 0]
        class1 = data[data[target] == 1]

        for feature in features:
            try:
                # Выполняем t-тест
                t_stat, p_value = stats.ttest_ind(
                    class0[feature].dropna(),
                    class1[feature].dropna(),
                    equal_var=False
                )
                # Проверяем значимость
                if p_value < 0.05:  # стандартный уровень значимости
                    significant_features.append(feature)
            except Exception as e:
                print(f"Ошибка при t-тесте для {feature}: {str(e)}")

        return significant_features

    def save_selected_features(self, step_name):
        """Сохранение отобранных признаков с метаданными"""
        save_dir = os.path.join(
            self.base_path,
            "FinalFeatureGeneration",
            "Data",
            self.organ,
            "SelectedFinalFeaturesSecond"
        )
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{step_name}_features.xlsx")

        if not self.selected_features:
            print(f"Предупреждение: Нет признаков для сохранения на этапе {step_name}")
            return

        # Создаем базовый DataFrame с признаками
        features_df = pd.DataFrame({'feature': self.selected_features})

        # Добавляем метаданные из feature_groups
        if not self.feature_groups.empty:
            # Проверяем наличие всех необходимых колонок
            meta_cols = ['Feature', 'SymmetryVector']
            additional_cols = ['left_interval_limit', 'right_interval_limit', 'label 0', 'label 1']

            # Добавляем недостающие колонки как пустые
            for col in additional_cols:
                if col not in self.feature_groups.columns:
                    self.feature_groups[col] = None
                    print(f"Предупреждение: Колонка {col} отсутствует в feature_groups")

            # Теперь все колонки гарантированно существуют
            all_meta_cols = meta_cols + additional_cols
            available_meta = [col for col in all_meta_cols if col in self.feature_groups.columns]

            features_df = features_df.merge(
                self.feature_groups[available_meta].rename(columns={'Feature': 'feature'}),
                on='feature',
                how='left'
            )
        else:
            # Если нет feature_groups, создаем пустые колонки
            empty_cols = ['SymmetryVector', 'left_interval_limit', 'right_interval_limit', 'label 0', 'label 1']
            for col in empty_cols:
                features_df[col] = None

        # Добавляем информативность
        if not self.informativeness_data.empty:
            info_cols = ['MG', 'informativeness', 'test_informativeness'] + ['left_interval_limit', 'right_interval_limit', 'label 0', 'label 1']
            available_info = [col for col in info_cols if col in self.informativeness_data.columns]

            if 'MG' in available_info:
                features_df = features_df.merge(
                    self.informativeness_data[available_info].rename(columns={'MG': 'feature'}),
                    on='feature',
                    how='left'
                )
        else:
            # Добавляем пустые колонки для информативности
            features_df['informativeness'] = None
            features_df['test_informativeness'] = None

        # Сохраняем
        features_df.to_excel(save_path, index=False)
        print(f"Сохранено {len(features_df)} признаков с метаданными: {save_path}")

    @staticmethod
    def _calculate_sensitivity_specificity(y_true, y_pred):
        """Вычисление чувствительности и специфичности"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return sensitivity, specificity

    def _evaluate_model(self, model, X_train, X_test, y_train, y_test, apply_smote=False):
        class_ratio = min(y_train.mean(), 1 - y_train.mean()) / max(y_train.mean(), 1 - y_train.mean())

        # Создаем пайплайн
        if apply_smote and class_ratio < 0.9:
            pipeline = make_pipeline(
                StandardScaler(),
                SMOTE(random_state=self.random_state),
                model
            )
        else:
            pipeline = make_pipeline(
                StandardScaler(),
                model
            )

        pipeline.fit(X_train, y_train)

        # Получаем вероятности и бинарные предсказания
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        # Вычисляем метрики
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        sensitivity, specificity = self._calculate_sensitivity_specificity(y_test, y_pred)

        # Ключевая метрика: корень из произведения чувствительности и специфичности
        g_mean = np.sqrt(sensitivity * specificity)

        return roc_auc, pr_auc, g_mean, sensitivity, specificity

    def evaluate_final_model(self, test_size=0.2, n_folds=5):
        """Оценка качества финальных признаков"""
        if not self.validation_report.get('selected_features'):
            print("Нет признаков для оценки!")
            return

        features = self.validation_report['selected_features']
        X = self.feature_data[features]
        y = self.feature_data[self.target_col]

        # Проверка наличия данных
        if X.empty or y.empty:
            print("Ошибка: Нет данных для оценки")
            return

        # Определяем применение SMOTE
        class_ratio = min(y.mean(), 1 - y.mean()) / max(y.mean(), 1 - y.mean())
        apply_smote = class_ratio < 0.9
        print(f"Соотношение классов: {class_ratio:.3f}, SMOTE: {'да' if apply_smote else 'нет'}")

        # Модели
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=100, max_depth=3,
                random_state=self.random_state, n_jobs=-1
            ),
            "LogisticRegression": LogisticRegression(
                penalty='l2', C=1.0, solver='liblinear',
                max_iter=1000, random_state=self.random_state, n_jobs=-1
            )
        }

        results = defaultdict(dict)

        # Hold-out оценка
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=self.random_state
            )

            for model_name, model in models.items():
                roc_auc, pr_auc, g_mean, sens, spec = self._evaluate_model(
                    model, X_train, X_test, y_train, y_test, apply_smote
                )
                results[model_name]['holdout_roc_auc'] = roc_auc
                results[model_name]['holdout_pr_auc'] = pr_auc
                results[model_name]['holdout_g_mean'] = g_mean
                results[model_name]['holdout_sensitivity'] = sens
                results[model_name]['holdout_specificity'] = spec
        except Exception as e:
            print(f"Ошибка при hold-out оценке: {str(e)}")

        # Кросс-валидация
        try:
            kf = StratifiedKFold(n_folds, shuffle=True, random_state=self.random_state)

            for model_name, model in models.items():
                cv_roc_auc = []
                cv_pr_auc = []
                cv_g_mean = []
                cv_sensitivity = []
                cv_specificity = []

                for train_idx, test_idx in kf.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    roc_auc, pr_auc, g_mean, sens, spec = self._evaluate_model(
                        model, X_train, X_test, y_train, y_test, apply_smote
                    )
                    cv_roc_auc.append(roc_auc)
                    cv_pr_auc.append(pr_auc)
                    cv_g_mean.append(g_mean)
                    cv_sensitivity.append(sens)
                    cv_specificity.append(spec)

                results[model_name]['cv_roc_auc_mean'] = np.mean(cv_roc_auc)
                results[model_name]['cv_roc_auc_std'] = np.std(cv_roc_auc)
                results[model_name]['cv_pr_auc_mean'] = np.mean(cv_pr_auc)
                results[model_name]['cv_pr_auc_std'] = np.std(cv_pr_auc)
                results[model_name]['cv_g_mean_mean'] = np.mean(cv_g_mean)
                results[model_name]['cv_g_mean_std'] = np.std(cv_g_mean)
                results[model_name]['cv_sensitivity_mean'] = np.mean(cv_sensitivity)
                results[model_name]['cv_specificity_mean'] = np.mean(cv_specificity)
        except Exception as e:
            print(f"Ошибка при кросс-валидации: {str(e)}")

        # Сохраняем результаты
        self.evaluation_report = dict(results)
        report_path = os.path.join(
            self.classification_results_dir,
            f"classification_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(report_path, 'w') as f:
            json.dump(self.evaluation_report, f, indent=4)

        print(f"Отчет классификации сохранен: {report_path}")

        # Итоговый отчет
        print(f"\n{'=' * 50}")
        print(f"ИТОГОВЫЙ ОТЧЕТ ДЛЯ {self.organ.upper()}")
        print(f"{'=' * 50}")
        print(f"Исходные признаки: {self.feature_data.shape[1] - 1}")
        print(
            f"После фильтрации дисперсии: {len([col for col in self.feature_data.columns if col != self.target_col])}")
        print(f"Финальных признаков: {len(self.selected_features)}")
        print(f"FDR: {self.validation_report.get('fdr', 0.0):.3f}")

        # Результаты классификации
        if self.evaluation_report:
            for model_name, metrics in self.evaluation_report.items():
                print(f"\n--- {model_name} ---")
                if 'holdout_roc_auc' in metrics:
                    print(f"Hold-out ROC-AUC: {metrics['holdout_roc_auc']:.3f}")
                    print(f"Hold-out PR-AUC: {metrics['holdout_pr_auc']:.3f}")
                    print(f"Hold-out G-mean: {metrics['holdout_g_mean']:.3f}")
                    print(f"Hold-out Sensitivity: {metrics['holdout_sensitivity']:.3f}")
                    print(f"Hold-out Specificity: {metrics['holdout_specificity']:.3f}")
                if 'cv_roc_auc_mean' in metrics:
                    print(f"CV ROC-AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
                    print(f"CV PR-AUC: {metrics['cv_pr_auc_mean']:.3f} ± {metrics['cv_pr_auc_std']:.3f}")
                    print(f"CV G-mean: {metrics['cv_g_mean_mean']:.3f} ± {metrics['cv_g_mean_std']:.3f}")
                    print(f"CV Sensitivity: {metrics['cv_sensitivity_mean']:.3f}")
                    print(f"CV Specificity: {metrics['cv_specificity_mean']:.3f}")

        return self

    def run_full_pipeline(self):
        """Запуск полного пайплайна"""
        start_time = time.time()
        print(f"\n{'-' * 50}")
        print(f"Начало обработки органа: {self.organ}")
        print(f"{'-' * 50}")

        # 1. Загрузка данных
        self.load_data()

        # Проверка целевой переменной
        if self.target_col not in self.feature_data.columns:
            print(f"Целевая переменная '{self.target_col}' не найдена, попытка найти альтернативу...")
            # Поиск похожих названий
            possible_targets = [col for col in self.feature_data.columns if
                                'target' in col.lower() or 'class' in col.lower()]
            if possible_targets:
                self.target_col = possible_targets[0]
                print(f"Используем альтернативную целевую переменную: {self.target_col}")
            else:
                raise ValueError("Целевая переменная не найдена в данных")

        # 2. Выбор лучших признаков по симметрии
        self.select_best_per_symmetry_group()
        self.save_selected_features("symmetry_selected")

        # 3. Фильтрация по корреляции
        self.filter_by_correlation()
        self.save_selected_features("correlation_filtered")

        # 4. Статистическая валидация
        try:
            self.statistical_validation(
                n_random_features=20,
                stability_threshold=self.stability_threshold,
                top_k=self.top_k
            )
            self.save_selected_features("statistically_validated")
        except Exception as e:
            print(f"Ошибка при статистической валидации: {str(e)}")
            # Используем топ-k признаков из корреляционного отбора
            self.selected_features = self.selected_features[:self.top_k]
            self.save_selected_features("fallback_selection")
            print(f"Используем топ-{self.top_k} признаков из корреляционного отбора")

        # 5. Оценка качества
        self.evaluate_final_model()

        # Итоговый отчет с акцентом на метрики для дисбаланса
        print(f"\n{'=' * 50}")
        print(f"ИТОГОВЫЙ ОТЧЕТ ДЛЯ {self.organ.upper()}")
        print(f"{'=' * 50}")
        print(f"Исходные признаки: {self.feature_data.shape[1] - 1}")
        print(
            f"После фильтрации дисперсии: {len([col for col in self.feature_data.columns if col != self.target_col])}")
        print(f"Финальных признаков: {len(self.selected_features)}")
        print(f"FDR: {self.validation_report.get('fdr', 0.0):.3f}")

        # Анализ дисбаланса
        class_counts = self.feature_data[self.target_col].value_counts()
        minority_class = class_counts.idxmin()
        minority_ratio = class_counts.min() / class_counts.sum()
        print(f"\nДисбаланс классов: {class_counts.to_dict()}")
        print(f"Миноритарный класс ({minority_class}): {minority_ratio:.3f}")

        # Результаты классификации с акцентом на PR-AUC и F1
        if self.evaluation_report:
            for model_name, metrics in self.evaluation_report.items():
                print(f"\n--- {model_name} ---")
                if 'holdout_pr_auc' in metrics:
                    print(f"Hold-out PR-AUC: {metrics['holdout_pr_auc']:.3f} (ключевая метрика при дисбалансе)")
                if 'holdout_f1' in metrics:
                    print(f"Hold-out F1: {metrics['holdout_f1']:.3f}")
                    print(f"Hold-out Precision: {metrics['holdout_precision']:.3f}")
                    print(f"Hold-out Recall: {metrics['holdout_recall']:.3f}")

                if 'cv_pr_auc_mean' in metrics:
                    print(f"CV PR-AUC: {metrics['cv_pr_auc_mean']:.3f} ± {metrics['cv_pr_auc_std']:.3f}")
                if 'cv_f1_mean' in metrics:
                    print(f"CV F1: {metrics['cv_f1_mean']:.3f} ± {metrics['cv_f1_std']:.3f}")

        print(f"\nОбщее время обработки: {(time.time() - start_time) / 60:.1f} мин")
        return self.selected_features


# Пример использования
# if __name__ == "__main__":
#     # Конфигурация органов
#     ORGAN_CONFIG = {
#         "Brain": {
#             "target_path": ".\\Data\\Brain\\BrainDataCRS.xlsx",
#             "target_col": "target"
#         },
#         "Ovaries": {
#             "target_path": ".\\Data\\Ovaries\\Ovaries.xlsx",
#             "target_col": "target"
#         },
#         "MammaryGlands": {
#             "target_path": ".\\Data\\MammaryGlands\\MGBothOrgans.xlsx",
#             "target_col": "target"
#         }
#     }
#
#     for organ, config in ORGAN_CONFIG.items():
#         print(f"\n{'=' * 50}")
#         print(f"Обработка органа: {organ}")
#         print(f"Путь к таргету: {config['target_path']}")
#         print(f"Колонка таргета: {config['target_col']}")
#         print(f"{'=' * 50}")
#
#         selector = FeatureSelector(
#             base_path=BASE_PATH,
#             organ=organ,
#             target_path=config['target_path'],
#             target_col=config['target_col'],
#             random_state=42
#         )
#
#         try:
#             final_features = selector.run_full_pipeline()
#             print(f"Успешно завершено для {organ}: отобрано {len(final_features)} признаков")
#         except Exception as e:
#             print(f"Критическая ошибка при обработке {organ}: {str(e)}")
#             import traceback
#
#             traceback.print_exc()
