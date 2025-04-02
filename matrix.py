from tensor import Tensor

class Matrix(Tensor):
    def __init__(self, rows, cols, data):
        if len(data) != rows * cols:
            raise ValueError("Data length does not match the specified dimensions.")

        self.rows = rows
        self.cols = cols
        self.data = [data[i * cols:(i + 1) * cols] for i in range(rows)]
        super().__init__((rows, cols), self.data)

    def conv_rc2i(self, r, c):
        if r < 0 or r >= self.rows or c < 0 or c >= self.cols:
            raise IndexError("Row or column index out of bounds.")
        return r * self.cols + c

    def conv_i2rc(self, i):
        if i < 0 or i >= self.rows * self.cols:
            raise IndexError("Index out of bounds.")
        return divmod(i, self.cols)

    def __str__(self):
        flat_data = [item for row in self.data for item in row]
        max_width = max(len(str(item)) for item in flat_data)
        format_str = "{:>" + str(max_width) + "}"

        lines = ["["]
        for row in self.data:
            line = "  ".join(format_str.format(item) for item in row)
            lines.append("    " + line)
        lines.append("]")
        return "\n".join(lines)

    def __getitem__(self, key):
        rows, cols = self.rows, self.cols

        if isinstance(key, tuple) and len(key) == 2:
            r_key, c_key = key

            if isinstance(r_key, int):
                selected_rows = [self.data[r_key]]
            elif isinstance(r_key, slice):
                selected_rows = self.data[r_key]
            elif isinstance(r_key, list):
                selected_rows = [self.data[i] for i in r_key]
            else:
                raise TypeError("Unsupported row key type.")

            if isinstance(c_key, int):
                return Matrix(len(selected_rows), 1, [row[c_key] for row in selected_rows])
            elif isinstance(c_key, slice):
                sliced_cols = [row[c_key] for row in selected_rows]
                return Matrix(len(selected_rows), len(sliced_cols[0]), [item for row in sliced_cols for item in row])
            elif isinstance(c_key, list):
                selected_cols = [[row[c] for c in c_key] for row in selected_rows]
                return Matrix(len(selected_rows), len(c_key), [item for row in selected_cols for item in row])
            else:
                raise TypeError("Unsupported column key type.")

        if isinstance(key, int):
            return Matrix(1, cols, self.data[key])
        elif isinstance(key, slice):
            sliced_data = self.data[key]
            return Matrix(len(sliced_data), cols, [item for row in sliced_data for item in row])
        elif isinstance(key, list):
            selected_rows = [self.data[i] for i in key]
            return Matrix(len(selected_rows), cols, [item for row in selected_rows for item in row])

        raise TypeError("Unsupported key type or format.")
