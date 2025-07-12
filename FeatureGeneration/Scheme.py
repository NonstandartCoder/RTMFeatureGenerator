import numpy as np
from scipy.signal import convolve2d


class Scheme:
    def __init__(self, temperatures, dot_names, grid_shape, has_paired_organ, has_support_points,
                 has_central_support_points):
        self._validate_inputs(dot_names, grid_shape, has_paired_organ, has_support_points, has_central_support_points)
        self.temperatures = temperatures
        self.dot_names = dot_names
        self.grid_shape = tuple(grid_shape)
        self.has_paired_organ = bool(has_paired_organ)
        self.has_support_points = bool(has_support_points)
        self.has_central_support_points = bool(has_central_support_points)

    def get_dot_names(self):
        return self.dot_names

    def get_temperatures(self):
        return self.temperatures

    def get_has_paired_organ(self):
        return self.has_paired_organ

    def get_grid_shape(self):
        return self.grid_shape

    def get_has_support_points(self):
        return self.has_support_points

    def get_has_central_support_points(self):
        return self.has_central_support_points

    def get_tensors(self):
        try:
            if self.has_paired_organ:
                # Получаем каналы для левой и правой стороны
                left = self._process_side("inner left", "skin left")
                right = self._process_side("inner right", "skin right")

                # Вычисляем разностные каналы (левая сторона - правая сторона)
                diff_channels = [l - r for l, r in zip(left, right)]

                # Объединяем все каналы: левые, правые, разностные
                all_channels = left + right + diff_channels

                if self.has_central_support_points:
                    central_inner_vals = self._get_values("central inner support dots")
                    central_skin_vals = self._get_values("central skin support dots")

                    n_central = central_inner_vals.shape[1]
                    for i in range(n_central):
                        # Обрабатываем центральные точки для левой и правой стороны
                        left_central = self._process_side_with_support(
                            "inner left", "skin left",
                            central_inner_vals[:, i],
                            central_skin_vals[:, i]
                        )
                        right_central = self._process_side_with_support(
                            "inner right", "skin right",
                            central_inner_vals[:, i],
                            central_skin_vals[:, i]
                        )

                        # Вычисляем разностные каналы для центральных точек
                        central_diff = [lc - rc for lc, rc in zip(left_central, right_central)]

                        # Добавляем в общий список каналов
                        all_channels.extend(left_central)
                        all_channels.extend(right_central)
                        all_channels.extend(central_diff)

                return np.stack(all_channels, axis=1)
            else:
                # Обработка для непарного органа
                channels = self._process_side("inner", "skin")

                if self.has_central_support_points:
                    central_inner_vals = self._get_values("central inner support dots")
                    central_skin_vals = self._get_values("central skin support dots")

                    n_central = central_inner_vals.shape[1]
                    for i in range(n_central):
                        central_channels = self._process_side_with_support(
                            "inner", "skin",
                            central_inner_vals[:, i],
                            central_skin_vals[:, i]
                        )
                        channels.extend(central_channels)

                return np.stack(channels, axis=1)
        except KeyError as e:
            raise ValueError(f"Missing temperature columns: {e}") from e

    def _process_side(self, inner_key, skin_key):
        inner_flat = self._get_values(inner_key)
        skin_flat = self._get_values(skin_key)
        n_patients = len(self.temperatures)
        inner = inner_flat.reshape(n_patients, *self.grid_shape)
        skin = skin_flat.reshape(n_patients, *self.grid_shape)
        grad_z = inner - skin

        inner_lap = self._compute_laplacian(inner, skin_values=skin)
        skin_lap = self._compute_laplacian(skin, dirichlet_value=23.0)

        # Для inner-канала
        inner_grad_x, inner_grad_y = self._compute_gradient(inner, skin_values=skin)
        # Для skin-канала
        skin_grad_x, skin_grad_y = self._compute_gradient(skin, dirichlet_value=23.0)

        channels = [inner, skin, grad_z, inner_lap, skin_lap,
                    inner_grad_x, inner_grad_y, skin_grad_x, skin_grad_y]

        if self.has_support_points:
            support_inner_key = f"support {inner_key}"
            support_skin_key = f"support {skin_key}"
            support_inner_vals = self._get_values(support_inner_key)
            support_skin_vals = self._get_values(support_skin_key)

            n_support = support_inner_vals.shape[1]
            if support_skin_vals.shape[1] != n_support:
                raise ValueError(f"Number of support points mismatch for {inner_key} and {skin_key}")

            for j in range(n_support):
                inner_s = inner - support_inner_vals[:, j, None, None]
                skin_s = skin - support_skin_vals[:, j, None, None]
                grad_z_s = inner_s - skin_s

                inner_lap_s = self._compute_laplacian(inner_s, skin_values=skin_s)
                dirichlet_s = 23.0 - support_skin_vals[:, j]
                skin_lap_s = self._compute_laplacian(skin_s, dirichlet_value=dirichlet_s)

                # Для опорных точек inner
                inner_grad_x_s, inner_grad_y_s = self._compute_gradient(inner_s, skin_values=skin_s)
                # Для опорных точек skin
                skin_grad_x_s, skin_grad_y_s = self._compute_gradient(skin_s, dirichlet_value=dirichlet_s)

                support_channels = [inner_s, skin_s, grad_z_s, inner_lap_s, skin_lap_s,
                                    inner_grad_x_s, inner_grad_y_s, skin_grad_x_s, skin_grad_y_s]

                channels.extend(support_channels)

        return channels

    def _process_side_with_support(self, inner_key, skin_key, support_inner, support_skin):
        inner_flat = self._get_values(inner_key)
        skin_flat = self._get_values(skin_key)
        n_patients = len(self.temperatures)
        inner = inner_flat.reshape(n_patients, *self.grid_shape)
        skin = skin_flat.reshape(n_patients, *self.grid_shape)

        # Вычитаем значения опорных точек
        inner = inner - support_inner[:, None, None]
        skin = skin - support_skin[:, None, None]

        grad_z = inner - skin
        inner_lap = self._compute_laplacian(inner, skin_values=skin)
        dirichlet_val = 23.0 - support_skin
        skin_lap = self._compute_laplacian(skin, dirichlet_value=dirichlet_val)

        # Для inner-канала
        inner_grad_x, inner_grad_y = self._compute_gradient(inner, skin_values=skin)
        # Для skin-канала
        skin_grad_x, skin_grad_y = self._compute_gradient(skin, dirichlet_value=dirichlet_val)

        return [inner, skin, grad_z, inner_lap, skin_lap,
                inner_grad_x, inner_grad_y, skin_grad_x, skin_grad_y]

    def _get_values(self, key):
        cols = [pos for vec in self.dot_names[key] for pos in vec]
        return self.temperatures[cols].astype(np.float32).values

    @staticmethod
    def _compute_laplacian(grids, skin_values=None, dirichlet_value=None):
        n_patients, n, m = grids.shape
        laplacians = np.zeros_like(grids)
        for i in range(n_patients):
            grid = grids[i]
            for row in range(n):
                for col in range(m):
                    neighbors = []
                    if row > 0:
                        neighbors.append(grid[row - 1, col])
                    else:
                        if skin_values is not None:
                            neighbors.append(skin_values[i, row, col])
                        elif dirichlet_value is not None:
                            dv = dirichlet_value[i] if isinstance(dirichlet_value, np.ndarray) else dirichlet_value
                            neighbors.append(dv)
                        else:
                            neighbors.append(0)
                    if row < n - 1:
                        neighbors.append(grid[row + 1, col])
                    else:
                        if skin_values is not None:
                            neighbors.append(skin_values[i, row, col])
                        elif dirichlet_value is not None:
                            dv = dirichlet_value[i] if isinstance(dirichlet_value, np.ndarray) else dirichlet_value
                            neighbors.append(dv)
                        else:
                            neighbors.append(0)
                    if col > 0:
                        neighbors.append(grid[row, col - 1])
                    else:
                        if skin_values is not None:
                            neighbors.append(skin_values[i, row, col])
                        elif dirichlet_value is not None:
                            dv = dirichlet_value[i] if isinstance(dirichlet_value, np.ndarray) else dirichlet_value
                            neighbors.append(dv)
                        else:
                            neighbors.append(0)
                    if col < m - 1:
                        neighbors.append(grid[row, col + 1])
                    else:
                        if skin_values is not None:
                            neighbors.append(skin_values[i, row, col])
                        elif dirichlet_value is not None:
                            dv = dirichlet_value[i] if isinstance(dirichlet_value, np.ndarray) else dirichlet_value
                            neighbors.append(dv)
                        else:
                            neighbors.append(0)
                    laplacians[i, row, col] = sum(neighbors) - 4 * grid[row, col]
        return laplacians

    @staticmethod
    def _compute_gradient(grids, skin_values=None, dirichlet_value=None):
        n_patients, n, m = grids.shape
        grad_x = np.zeros_like(grids)
        grad_y = np.zeros_like(grids)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        for i in range(n_patients):
            grid = grids[i]
            # Определяем граничные значения для текущего пациента
            d_val = None
            skin_vals = None

            if skin_values is not None:
                skin_vals = skin_values[i]
            elif dirichlet_value is not None:
                d_val = dirichlet_value[i] if isinstance(dirichlet_value, np.ndarray) else dirichlet_value

            # Создаем расширенную сетку (n+2, m+2)
            expanded = np.zeros((n + 2, m + 2))
            expanded[1:-1, 1:-1] = grid  # Центральная часть

            # Заполняем границы
            if skin_vals is not None:
                # Верхняя и нижняя границы
                expanded[0, 1:-1] = skin_vals[0, :]  # Верх
                expanded[-1, 1:-1] = skin_vals[-1, :]  # Низ
                # Левая и правая границы
                expanded[1:-1, 0] = skin_vals[:, 0]  # Лево
                expanded[1:-1, -1] = skin_vals[:, -1]  # Право
                # Углы
                expanded[0, 0] = skin_vals[0, 0]  # Верх-лево
                expanded[0, -1] = skin_vals[0, -1]  # Верх-право
                expanded[-1, 0] = skin_vals[-1, 0]  # Низ-лево
                expanded[-1, -1] = skin_vals[-1, -1]  # Низ-право
            elif d_val is not None:
                expanded[0, 1:-1] = d_val  # Верх
                expanded[-1, 1:-1] = d_val  # Низ
                expanded[1:-1, 0] = d_val  # Лево
                expanded[1:-1, -1] = d_val  # Право
                expanded[0, 0] = d_val  # Верх-лево
                expanded[0, -1] = d_val  # Верх-право
                expanded[-1, 0] = d_val  # Низ-лево
                expanded[-1, -1] = d_val  # Низ-право

            # Вычисляем градиенты на расширенной сетке
            gx = convolve2d(expanded, sobel_x, mode='valid')
            gy = convolve2d(expanded, sobel_y, mode='valid')

            grad_x[i] = gx
            grad_y[i] = gy

        return grad_x, grad_y

    @staticmethod
    def _validate_inputs(dot_names, grid_shape, has_paired_organ, has_support_points, has_central_support_points):
        required_paired = ["inner left", "inner right", "skin left", "skin right"]
        required_single = ["inner", "skin"]
        required_keys = required_paired if has_paired_organ else required_single
        if not all(k in dot_names for k in required_keys):
            raise ValueError(f"Missing required keys: {required_keys}")

        if has_support_points:
            support_required = [f"support {k}" for k in required_keys]
            if not all(k in dot_names for k in support_required):
                raise ValueError(f"Missing support keys: {support_required}")

            for key in support_required:
                if not isinstance(dot_names[key], np.ndarray):
                    raise TypeError(f"Support dots for {key} must be numpy arrays.")

        if has_central_support_points:
            central_required = ["central inner support dots", "central skin support dots"]
            if not all(k in dot_names for k in central_required):
                raise ValueError(
                    "Both central inner and skin support dots must be provided when has_central_support_points is True.")

            for key in central_required:
                if not isinstance(dot_names[key], np.ndarray):
                    raise TypeError(f"Central support dots for {key} must be numpy arrays.")

            central_inner = dot_names["central inner support dots"]
            central_skin = dot_names["central skin support dots"]

            if has_paired_organ:
                if central_inner.size < 2 or central_skin.size < 2:
                    raise ValueError("For paired organs, central support dots must have at least two elements.")
                if central_inner.shape[1] % 2 != 0:
                    raise ValueError("For paired organs, central support dots must come in pairs")
            else:
                if central_inner.size < 1 or central_skin.size < 1:
                    raise ValueError("Central support dots must have at least one element for single organ.")

        if not isinstance(grid_shape, (tuple, list)) or len(grid_shape) != 2:
            raise ValueError("grid_shape must be a 2-element tuple.")

        all_dots = []
        for key in required_keys + (
                ["central inner support dots", "central skin support dots"] if has_central_support_points else []) + (
                           [f"support {k}" for k in required_keys] if has_support_points else []):
            matrix = dot_names[key]
            if not isinstance(matrix, np.ndarray):
                raise TypeError(f"Dot names for {key} must be numpy arrays.")
            for row in matrix:
                for dot in row:
                    all_dots.append(str(dot).strip())

        seen = {}
        duplicates = set()
        for idx, dot in enumerate(all_dots):
            if dot in seen:
                duplicates.add(f"'{dot}' (positions {seen[dot]} and {idx})")
            else:
                seen[dot] = idx
        if duplicates:
            raise ValueError(f"Duplicate dot names: {', '.join(duplicates)}")

