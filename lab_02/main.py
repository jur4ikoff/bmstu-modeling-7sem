"""Построение графиков равномерного и эрланговского распределений."""

from __future__ import annotations

import math
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk


PLOT_WIDTH = 820
PLOT_HEIGHT = 270
PLOT_MARGIN = 52
SAMPLES_COUNT = 500


@dataclass(frozen=True)
class UniformParameters:
    """Параметры равномерного распределения на отрезке [left, right]."""

    left: float
    right: float


@dataclass(frozen=True)
class ErlangParameters:
    """Параметры распределения Эрланга: порядок и интенсивность."""

    order: int
    rate: float


def uniform_density(x: float, parameters: UniformParameters) -> float:
    """Возвращает плотность равномерного распределения."""
    if parameters.left <= x <= parameters.right:
        return 1 / (parameters.right - parameters.left)
    return 0.0


def uniform_cdf(x: float, parameters: UniformParameters) -> float:
    """Возвращает функцию распределения равномерной величины."""
    if x <= parameters.left:
        return 0.0
    if x >= parameters.right:
        return 1.0
    return (x - parameters.left) / (parameters.right - parameters.left)


def erlang_density(x: float, parameters: ErlangParameters) -> float:
    """Возвращает плотность распределения Эрланга."""
    if x < 0:
        return 0.0
    if x == 0:
        return parameters.rate if parameters.order == 1 else 0.0

    logarithm = (
        parameters.order * math.log(parameters.rate)
        + (parameters.order - 1) * math.log(x)
        - parameters.rate * x
        - math.lgamma(parameters.order)
    )
    return math.exp(logarithm) if logarithm > -745 else 0.0


def erlang_cdf(x: float, parameters: ErlangParameters) -> float:
    """Возвращает функцию распределения Эрланга для целого порядка."""
    if x <= 0:
        return 0.0

    rate_x = parameters.rate * x
    series = sum(rate_x**index / math.factorial(index) for index in range(parameters.order))
    return max(0.0, min(1.0, 1 - math.exp(-rate_x) * series))


class DistributionPlot(tk.Canvas):
    """Холст с осями и графиком одной функции распределения."""

    def __init__(self, parent: tk.Misc, title: str) -> None:
        super().__init__(
            parent,
            width=PLOT_WIDTH,
            height=PLOT_HEIGHT,
            background="white",
            highlightthickness=1,
            highlightbackground="#b8c2cc",
        )
        self.title = title

    def draw(
        self,
        function: callable,
        x_minimum: float,
        x_maximum: float,
        y_maximum: float,
        function_name: str,
        markers: tuple[tuple[float, str, str], ...] = (),
    ) -> None:
        """Очищает холст и строит функцию на заданном интервале."""
        self.delete("all")
        self.create_text(
            PLOT_WIDTH / 2,
            18,
            text=f"{self.title}: {function_name}",
            font=("Arial", 12, "bold"),
        )

        left = PLOT_MARGIN
        right = PLOT_WIDTH - 18
        top = 38
        bottom = PLOT_HEIGHT - 38
        y_minimum = 0.0
        x_span = x_maximum - x_minimum
        y_span = y_maximum - y_minimum

        def to_canvas_x(value: float) -> float:
            return left + (value - x_minimum) / x_span * (right - left)

        def to_canvas_y(value: float) -> float:
            return bottom - (value - y_minimum) / y_span * (bottom - top)

        self.create_line(left, bottom, right, bottom, fill="#404040", arrow=tk.LAST)
        self.create_line(left, bottom, left, top, fill="#404040", arrow=tk.LAST)
        self.create_text(right - 4, bottom + 18, text="x", anchor=tk.E)
        self.create_text(left - 14, top + 4, text="y", anchor=tk.N)

        for tick in range(6):
            x_value = x_minimum + x_span * tick / 5
            x = to_canvas_x(x_value)
            self.create_line(x, bottom - 4, x, bottom + 4, fill="#404040")
            self.create_text(x, bottom + 17, text=f"{x_value:.2g}", anchor=tk.N)

        for tick in range(6):
            y_value = y_maximum * tick / 5
            y = to_canvas_y(y_value)
            self.create_line(left - 4, y, left + 4, y, fill="#404040")
            self.create_text(left - 8, y, text=f"{y_value:.2g}", anchor=tk.E)
            if tick:
                self.create_line(left, y, right, y, fill="#e8edf2")

        for index, (value, label, color) in enumerate(markers):
            if not x_minimum <= value <= x_maximum:
                continue
            x = to_canvas_x(value)
            self.create_line(x, top, x, bottom, fill=color, dash=(4, 3), width=2)
            self.create_text(
                x,
                bottom - 6 - index * 16,
                text=label,
                fill=color,
                anchor=tk.S,
                font=("Arial", 9, "bold"),
            )

        points = []
        for index in range(SAMPLES_COUNT + 1):
            x_value = x_minimum + x_span * index / SAMPLES_COUNT
            y_value = max(y_minimum, min(y_maximum, function(x_value)))
            points.extend((to_canvas_x(x_value), to_canvas_y(y_value)))
        self.create_line(*points, fill="#006fb9", width=2, smooth=True)


class DistributionApplication(ttk.Frame):
    """Графический интерфейс лабораторной работы."""

    def __init__(self, root: tk.Tk) -> None:
        super().__init__(root, padding=14)
        self.root = root
        self.uniform_left = tk.StringVar(value="0")
        self.uniform_right = tk.StringVar(value="1")
        self.erlang_order = tk.StringVar(value="3")
        self.erlang_rate = tk.StringVar(value="1")
        self.function_kind = tk.StringVar(value="density")
        self.status = tk.StringVar()

        root.title("ЛР №2 — распределения")
        root.minsize(PLOT_WIDTH + 40, 700)
        self.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._create_controls()
        self.uniform_plot = DistributionPlot(self, "Равномерное распределение")
        self.uniform_plot.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        self.erlang_plot = DistributionPlot(self, "Распределение Эрланга")
        self.erlang_plot.grid(row=2, column=0, sticky="ew")
        ttk.Label(self, textvariable=self.status, foreground="#a00000").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        self.redraw()

    def _create_controls(self) -> None:
        controls = ttk.LabelFrame(self, text="Параметры", padding=10)
        controls.grid(row=0, column=0, sticky="ew")

        ttk.Label(controls, text="Равномерное: a").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.uniform_left, width=9).grid(
            row=0, column=1, padx=(5, 12)
        )
        ttk.Label(controls, text="b").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.uniform_right, width=9).grid(
            row=0, column=3, padx=(5, 20)
        )

        ttk.Label(controls, text="Эрланга: k").grid(row=0, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.erlang_order, width=7).grid(
            row=0, column=5, padx=(5, 12)
        )
        ttk.Label(controls, text="λ").grid(row=0, column=6, sticky="w")
        ttk.Entry(controls, textvariable=self.erlang_rate, width=9).grid(
            row=0, column=7, padx=(5, 20)
        )

        ttk.Radiobutton(
            controls,
            text="Плотность",
            variable=self.function_kind,
            value="density",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Radiobutton(
            controls,
            text="Функция распределения",
            variable=self.function_kind,
            value="cdf",
        ).grid(row=1, column=2, columnspan=3, sticky="w", pady=(8, 0))
        ttk.Button(controls, text="Построить графики", command=self.redraw).grid(
            row=1, column=6, columnspan=2, sticky="e", pady=(8, 0)
        )

        for column in range(8):
            controls.columnconfigure(column, weight=1 if column in (3, 7) else 0)

    def _read_parameters(self) -> tuple[UniformParameters, ErlangParameters]:
        try:
            uniform = UniformParameters(
                left=float(self.uniform_left.get().replace(",", ".")),
                right=float(self.uniform_right.get().replace(",", ".")),
            )
            erlang = ErlangParameters(
                order=int(self.erlang_order.get()),
                rate=float(self.erlang_rate.get().replace(",", ".")),
            )
        except ValueError as error:
            raise ValueError("Параметры должны быть числами; порядок k должен быть целым") from error

        if not math.isfinite(uniform.left) or not math.isfinite(uniform.right):
            raise ValueError("Границы равномерного распределения должны быть конечными")
        if uniform.left >= uniform.right:
            raise ValueError("Для равномерного распределения необходимо a < b")
        if not 1 <= erlang.order <= 50:
            raise ValueError("Порядок Эрланга k должен быть целым от 1 до 50")
        if not math.isfinite(erlang.rate) or erlang.rate <= 0:
            raise ValueError("Интенсивность λ должна быть положительным числом")
        return uniform, erlang

    def redraw(self) -> None:
        """Строит оба графика по значениям полей интерфейса."""
        try:
            uniform, erlang = self._read_parameters()
        except ValueError as error:
            self.status.set(str(error))
            return

        self.status.set("")
        if self.function_kind.get() == "density":
            uniform_height = 1 / (uniform.right - uniform.left)
            uniform_padding = (uniform.right - uniform.left) * 0.2
            self.uniform_plot.draw(
                lambda x: uniform_density(x, uniform),
                uniform.left - uniform_padding,
                uniform.right + uniform_padding,
                uniform_height * 1.2,
                "плотность f(x)",
                (
                    (uniform.left, f"a = {uniform.left:g}", "#b22222"),
                    (uniform.right, f"b = {uniform.right:g}", "#b22222"),
                ),
            )
            x_maximum = (erlang.order + 6 * math.sqrt(erlang.order)) / erlang.rate
            mode = max(0, (erlang.order - 1) / erlang.rate)
            erlang_height = erlang_density(mode, erlang)
            mean = erlang.order / erlang.rate
            erlang_markers = ((mean, f"E[X] = {mean:g}", "#228b22"),)
            if erlang.order > 1:
                erlang_markers += ((mode, f"мода = {mode:g}", "#b06000"),)
            self.erlang_plot.draw(
                lambda x: erlang_density(x, erlang),
                0.0,
                x_maximum,
                erlang_height * 1.2 if erlang_height else 1.0,
                "плотность f(x)",
                erlang_markers,
            )
        else:
            uniform_padding = (uniform.right - uniform.left) * 0.2
            self.uniform_plot.draw(
                lambda x: uniform_cdf(x, uniform),
                uniform.left - uniform_padding,
                uniform.right + uniform_padding,
                1.05,
                "функция распределения F(x)",
                (
                    (uniform.left, f"a = {uniform.left:g}", "#b22222"),
                    (uniform.right, f"b = {uniform.right:g}", "#b22222"),
                ),
            )
            x_maximum = (erlang.order + 6 * math.sqrt(erlang.order)) / erlang.rate
            mean = erlang.order / erlang.rate
            erlang_markers = ((mean, f"E[X] = {mean:g}", "#228b22"),)
            if erlang.order > 1:
                mode = (erlang.order - 1) / erlang.rate
                erlang_markers += ((mode, f"мода = {mode:g}", "#b06000"),)
            self.erlang_plot.draw(
                lambda x: erlang_cdf(x, erlang),
                0.0,
                x_maximum,
                1.05,
                "функция распределения F(x)",
                erlang_markers,
            )


def main() -> None:
    """Запускает графический интерфейс."""
    root = tk.Tk()
    DistributionApplication(root)
    root.mainloop()


if __name__ == "__main__":
    main()
