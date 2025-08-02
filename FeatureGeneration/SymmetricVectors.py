import numpy as np
from itertools import combinations
from math import gcd


class SymmetricVectors:
    def __init__(self, scheme):
        self.scheme = scheme
        self.grid_shape = self.scheme.get_grid_shape()
        self.has_paired_organ = self.scheme.get_has_paired_organ()
        self.matrix_names = self.get_matrix_names()

    def get_matrix_names(self):
        base_names = [
            "Inner", "Skin", "Gradient",
            "Laplacian Inner", "Laplacian Skin",
            "Inner Gradient X", "Inner Gradient Y",
            "Skin Gradient X", "Skin Gradient Y"
        ]

        dot_names = self.scheme.get_dot_names()
        names = []

        def get_dot_str(base_name, dots):
            """Returns appropriate dot string based on matrix type and context"""
            if "Gradient" in base_name or "Laplacian" in base_name:
                return "/".join([d for d in dots.values() if d])
            elif "Inner" in base_name:
                return dots.get("inner", dots.get("left_inner", dots.get("right_inner", "")))
            elif "Skin" in base_name:
                return dots.get("skin", dots.get("left_skin", dots.get("right_skin", "")))
            return "/".join([d for d in dots.values() if d])

        if self.has_paired_organ:
            # Base channels: left, right, diff (9 × 3 = 27)
            for base in base_names:
                names.append(f"{base} left")
                names.append(f"{base} right")
                names.append(f"{base} diff")

            # Regular support points
            if self.scheme.get_has_support_points():
                inner_left = dot_names["support inner left"].ravel()
                skin_left = dot_names["support skin left"].ravel()
                inner_right = dot_names["support inner right"].ravel()
                skin_right = dot_names["support skin right"].ravel()

                for i in range(len(inner_left)):
                    lo = inner_left[i]  # Left inner dot (e.g., "LO")
                    lok = skin_left[i]  # Left skin dot (e.g., "LOK")
                    ro = inner_right[i]  # Right inner dot (e.g., "RO")
                    rok = skin_right[i]  # Right skin dot (e.g., "ROK")

                    for base in base_names:
                        # Left side
                        dots = {"left_inner": lo, "left_skin": lok}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} Support({dot_str}) left")

                        # Right side - corrected
                        dots = {"right_inner": ro, "right_skin": rok}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} Support({dot_str}) right")

                        # Diff side
                        if "Inner" in base or "Laplacian Inner" in base:
                            dot_str = f"{lo}/{ro}"
                        elif "Skin" in base or "Laplacian Skin" in base:
                            dot_str = f"{lok}/{rok}"
                        else:
                            dot_str = f"{lo}/{lok}/{ro}/{rok}"
                        names.append(f"{base} Support({dot_str}) diff")

            # Central support points
            if self.scheme.get_has_central_support_points():
                central_inner = dot_names["central inner support dots"].ravel()
                central_skin = dot_names["central skin support dots"].ravel()

                # For paired organs, central support dots come in pairs
                n_central = len(central_inner) // 2
                for i in range(n_central):
                    # Left central dots
                    inner_left = central_inner[2 * i]
                    skin_left = central_skin[2 * i]
                    # Right central dots
                    inner_right = central_inner[2 * i + 1]
                    skin_right = central_skin[2 * i + 1]

                    for base in base_names:
                        # Left central
                        dots = {"left_inner": inner_left, "left_skin": skin_left}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} CentralSupport({dot_str}) left")

                        # Right central
                        dots = {"right_inner": inner_right, "right_skin": skin_right}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} CentralSupport({dot_str}) right")

                        # Central diff
                        if "Inner" in base or "Laplacian Inner" in base:
                            dot_str = f"{inner_left}/{inner_right}"
                        elif "Skin" in base or "Laplacian Skin" in base:
                            dot_str = f"{skin_left}/{skin_right}"
                        else:
                            dot_str = f"{inner_left}/{skin_left}/{inner_right}/{skin_right}"
                        names.append(f"{base} CentralSupport({dot_str}) diff")

        else:  # Single organ
            # Base channels (9)
            names.extend(base_names)

            # Regular support points
            if self.scheme.get_has_support_points():
                inner_dots = dot_names["support inner"].ravel()
                skin_dots = dot_names["support skin"].ravel()

                for i in range(len(inner_dots)):
                    inner_dot = inner_dots[i]
                    skin_dot = skin_dots[i]

                    for base in base_names:
                        dots = {"inner": inner_dot, "skin": skin_dot}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} Support({dot_str})")

            # Central support points
            if self.scheme.get_has_central_support_points():
                central_inner = dot_names["central inner support dots"].ravel()
                central_skin = dot_names["central skin support dots"].ravel()

                for i in range(len(central_inner)):
                    inner_dot = central_inner[i]
                    skin_dot = central_skin[i]

                    for base in base_names:
                        dots = {"inner": inner_dot, "skin": skin_dot}
                        dot_str = get_dot_str(base, dots)
                        names.append(f"{base} CentralSupport({dot_str})")

        return names

    def reflect_point(self, x, y, line_p1, line_p2):
        x1, y1 = line_p1
        x2, y2 = line_p2

        if x1 == x2:
            reflected_x = 2 * x1 - x
            reflected_y = y
            return self.validate_reflection(reflected_x, reflected_y)

        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2

        denominator = a ** 2 + b ** 2
        if denominator == 0:
            return None

        proj_x = (b * (b * x - a * y) - a * c) / denominator
        proj_y = (a * (-b * x + a * y) - b * c) / denominator

        reflected_x = 2 * proj_x - x
        reflected_y = 2 * proj_y - y

        return self.validate_reflection(reflected_x, reflected_y)

    def validate_reflection(self, x, y):
        EPSILON = 1e-6
        x_round = round(x)
        y_round = round(y)

        if abs(x - x_round) > EPSILON or abs(y - y_round) > EPSILON:
            return None

        if 0 <= x_round < self.grid_shape[1] and 0 <= y_round < self.grid_shape[0]:
            return int(x_round), int(y_round)
        return None

    def get_outer_cycle(self):
        n, m = self.grid_shape
        cycle = []

        if n < 1 or m < 1:
            return cycle

        for col in range(m):
            cycle.append((0, col))
        for row in range(1, n):
            cycle.append((row, m-1))
        if n > 1:
            for col in range(m-2, -1, -1):
                cycle.append((n-1, col))
        if m > 1:
            for row in range(n-2, 0, -1):
                cycle.append((row, 0))

        return cycle

    def is_outer_point(self, row, col):
        n, m = self.grid_shape
        return row == 0 or row == n-1 or col == 0 or col == m-1

    def rotate_point(self, row, col, k=1):
        n, m = self.grid_shape
        if n < 1 or m < 1:
            return None

        cycle = self.get_outer_cycle()
        if not cycle:
            return None

        cycle_length = len(cycle)

        if not self.is_outer_point(row, col):
            return row, col

        try:
            idx = cycle.index((row, col))
            new_idx = (idx + k) % cycle_length
            return cycle[new_idx]
        except ValueError:
            return None

    def get_dot_names(self, matrix_name):
        """Для разностных матриц используем сетку точек левой стороны"""
        # Определяем базовое имя матрицы без указания стороны/типа
        base_name = matrix_name
        if 'diff' in matrix_name.lower():
            base_name = matrix_name.replace('diff', '').strip()
        elif 'left' in matrix_name.lower() or 'right' in matrix_name.lower():
            base_name = matrix_name.rsplit(' ', 1)[0].strip()

        # Определяем тип матрицы
        if 'inner' in base_name.lower():
            key = "inner left" if self.scheme.get_has_paired_organ() else "inner"
        elif 'skin' in base_name.lower():
            key = "skin left" if self.scheme.get_has_paired_organ() else "skin"
        elif 'gradient' in base_name.lower():
            key = "inner left" if self.scheme.get_has_paired_organ() else "inner"
        elif 'laplacian' in base_name.lower():
            if 'inner' in base_name.lower():
                key = "inner left" if self.scheme.get_has_paired_organ() else "inner"
            else:
                key = "skin left" if self.scheme.get_has_paired_organ() else "skin"
        else:
            raise ValueError(f"Unknown matrix type: {matrix_name}")

        return self.scheme.get_dot_names()[key]

    def get_intra_matrix_symmetries(self):
        symmetries = {}

        for matrix_idx, matrix_name in enumerate(self.matrix_names):
            dot_grid = self.get_dot_names(matrix_name)
            grid_points = [(r, c) for r in range(self.grid_shape[0])
                           for c in range(self.grid_shape[1])]

            line_registry = {}

            for p1, p2 in combinations(grid_points, 2):
                try:
                    dot1 = dot_grid[p1[0]][p1[1]]
                    dot2 = dot_grid[p2[0]][p2[1]]
                except IndexError:
                    continue

                x1, y1 = p1[1], p1[0]
                x2, y2 = p2[1], p2[0]

                a = y2 - y1
                b = x1 - x2
                c = x2 * y1 - x1 * y2

                divisor = gcd(gcd(abs(a), abs(b)), abs(c)) or 1
                a_red, b_red, c_red = a // divisor, b // divisor, c // divisor

                if a_red < 0 or (a_red == 0 and b_red < 0):
                    a_red, b_red, c_red = -a_red, -b_red, -c_red

                line_hash = (a_red, b_red, c_red)
                segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                axis_pairs = []
                for row in range(self.grid_shape[0]):
                    for col in range(self.grid_shape[1]):
                        reflected = self.reflect_point(
                            x=col, y=row,
                            line_p1=(x1, y1),
                            line_p2=(x2, y2)
                        )
                        if reflected is not None:
                            ref_col, ref_row = reflected
                            try:
                                # Проверка существования точки в матрице
                                _ = dot_grid[ref_row][ref_col]
                                original = (matrix_idx, row, col)
                                reflected_pt = (matrix_idx, ref_row, ref_col)
                                if original != reflected_pt:
                                    axis_pairs.append(tuple(sorted([original, reflected_pt])))
                            except IndexError:
                                continue

                axis_pairs = list(set(axis_pairs))

                if line_hash in line_registry:
                    if segment_length > line_registry[line_hash][0]:
                        line_registry[line_hash] = (segment_length, axis_pairs, dot1, dot2)
                else:
                    line_registry[line_hash] = (segment_length, axis_pairs, dot1, dot2)

            for line_hash, (_, axis_pairs, dot1, dot2) in line_registry.items():
                if len(axis_pairs) > 0:
                    key = f"{matrix_name} | Axis {dot1}-{dot2}"
                    symmetries[key] = axis_pairs

        return symmetries

    def get_rotation_symmetries(self):
        symmetries = {}
        cycle = self.get_outer_cycle()
        cycle_length = len(cycle)
        if cycle_length < 2:
            return symmetries

        n, m = self.grid_shape

        # Обработка вращений внутри одной матрицы
        for matrix_idx, matrix_name in enumerate(self.matrix_names):
            dot_grid = self.get_dot_names(matrix_name)

            for k in range(1, cycle_length):
                pairs = []
                for i, (row1, col1) in enumerate(cycle):
                    rotated = self.rotate_point(row1, col1, k)
                    if rotated is None:
                        continue
                    row2, col2 = rotated
                    try:
                        dot1 = dot_grid[row1][col1]
                        dot2 = dot_grid[row2][col2]
                        if dot1 != dot2:
                            pair = ((matrix_idx, row1, col1), (matrix_idx, row2, col2))
                            pairs.append(tuple(sorted([pair[0], pair[1]])))
                    except IndexError:
                        continue
                pairs = list(set(pairs))
                if pairs:
                    symmetries[f"{matrix_name} | Cyclic Rotation k={k}"] = pairs

        # Обработка пар лево-право (исключая разностные матрицы)
        if self.has_paired_organ:
            group_size = len(self.matrix_names) // 3  # left, right, diff группы
            for left_idx in range(group_size):
                right_idx = left_idx + group_size  # соответствущая правая матрица
                left_name = self.matrix_names[left_idx]
                right_name = self.matrix_names[right_idx]
                left_dots = self.get_dot_names(left_name)
                right_dots = self.get_dot_names(right_name)

                for k in range(1, cycle_length):
                    pairs = []
                    # Внешние точки с ротацией
                    for i, (row1, col1) in enumerate(cycle):
                        rotated = self.rotate_point(row1, col1, k)
                        if rotated is None:
                            continue
                        row2, col2 = rotated
                        try:
                            dot_left = left_dots[row1][col1]
                            dot_right = right_dots[row2][col2]
                            if dot_left != dot_right:
                                pair = ((left_idx, row1, col1), (right_idx, row2, col2))
                                pairs.append(tuple(sorted([pair[0], pair[1]])))
                        except IndexError:
                            continue

                    # Внутренние точки без ротации
                    for row in range(n):
                        for col in range(m):
                            if not self.is_outer_point(row, col):
                                try:
                                    dot_left = left_dots[row][col]
                                    dot_right = right_dots[row][col]
                                    if dot_left != dot_right:
                                        pairs.append(((left_idx, row, col), (right_idx, row, col)))
                                except IndexError:
                                    continue

                    pairs = list(set(pairs))
                    if pairs:
                        key = f"Left-Right | {left_name} ↔ {right_name} | Cyclic Rotation k={k}"
                        symmetries[key] = pairs

        return symmetries

    def get_symmetric_vectors(self):
        symmetries = {
            **self.get_intra_matrix_symmetries(),
            **self.get_rotation_symmetries()
        }

        subvector_registry = set()
        res = symmetries.copy()

        for key, pairs in symmetries.items():
            if "Cyclic Rotation" in key:
                continue

            n = len(pairs)
            for mask in range(1, 1 << n):
                subset = [pairs[i] for i in range(n) if (mask & (1 << i))]
                dot_pairs = []
                for pair in subset:
                    (m1, r1, c1), (m2, r2, c2) = pair
                    matrix1 = self.matrix_names[m1]
                    matrix2 = self.matrix_names[m2]
                    dots1 = self.get_dot_names(matrix1)
                    dots2 = self.get_dot_names(matrix2)
                    dot1 = dots1[r1][c1]
                    dot2 = dots2[r2][c2]
                    sorted_dots = tuple(sorted([dot1, dot2]))
                    dot_pairs.append(sorted_dots)

                sorted_dot_pairs = sorted(dot_pairs)
                subset_hash = frozenset(sorted_dot_pairs)

                if subset_hash in subvector_registry:
                    continue
                subvector_registry.add(subset_hash)

                desc_parts = [f"{p[0]}-{p[1]}" for p in sorted_dot_pairs]
                desc = ", ".join(desc_parts)
                res_key = f"{key} | Subvector [{desc}]"
                res[res_key] = subset

        return res
