"""Чтение исходных данных из файла или с клавиатуры."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from math import log10
from pathlib import Path
from random import randint
from typing import Sequence


TABLE_ROWS = 10
DigitGroups = tuple[int, int, int]
BENFORD_PROBABILITIES = tuple(log10(1 + 1 / digit) for digit in range(1, 10))
VALUE_RANGES = ((0, 9), (0, 99), (0, 999)) * 2


def generate_random_numbers(count: int) -> list[int]:
    """Возвращает ``count`` случайных чисел от 100000 до 999999."""
    if count < 0:
        raise ValueError("Количество чисел не может быть отрицательным")

    return [randint(100_000, 999_999) for _ in range(count)]


def benford_randomness_score(numbers: Sequence[int | float | str]) -> float:
    """Возвращает оценку соответствия закону Бенфорда от 0.0 до 1.0.

    Оценка основана на первых значащих цифрах ненулевых чисел. Значение 1.0
    означает точное совпадение распределения с законом Бенфорда, а 0.0 —
    максимально возможное отклонение от него.
    """
    counts = [0] * 9
    valid_numbers = 0

    for number in numbers:
        try:
            value = Decimal(str(number))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Ожидалось число, получено: {number!r}") from error

        if not value.is_finite():
            raise ValueError(f"Число должно быть конечным, получено: {number!r}")
        if value.is_zero():
            continue

        first_digit = value.copy_abs().as_tuple().digits[0]
        counts[first_digit - 1] += 1
        valid_numbers += 1

    if valid_numbers == 0:
        return 0.0

    observed = [count / valid_numbers for count in counts]
    total_variation_distance = sum(
        abs(actual - expected)
        for actual, expected in zip(observed, BENFORD_PROBABILITIES)
    ) / 2
    max_distance = 1 - min(BENFORD_PROBABILITIES)

    return max(0.0, min(1.0, 1 - total_variation_distance / max_distance))


def order_score(
    numbers: Sequence[int | float | str], lower_bound: int, upper_bound: int
) -> float:
    """Возвращает оценку порядка чисел с учётом обратных подпоследовательностей.

    Для каждой непрерывной тройки чисел строится пара соседних разностей.
    Одинаковые пары, а также пары, полученные при чтении тройки в обратном
    направлении, считаются одним шаблоном. Длинные монотонные фрагменты и
    повторяющиеся прямые или обратные подпоследовательности снижают оценку.
    """
    if lower_bound >= upper_bound:
        raise ValueError("Нижняя граница должна быть меньше верхней")

    lower = Decimal(lower_bound)
    upper = Decimal(upper_bound)
    values: list[Decimal] = []

    for number in numbers:
        try:
            value = Decimal(str(number))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Ожидалось число, получено: {number!r}") from error

        if not value.is_finite():
            raise ValueError(f"Число должно быть конечным, получено: {number!r}")
        if not lower <= value <= upper:
            raise ValueError(
                f"Число {number!r} вне допустимого диапазона "
                f"от {lower_bound} до {upper_bound}"
            )
        values.append(value)

    if len(values) < 3:
        return 0.0

    patterns: set[tuple[Decimal, Decimal]] = set()
    monotonic_triplets = 0
    windows_count = len(values) - 2

    for first, second, third in zip(values, values[1:], values[2:]):
        first_difference = second - first
        second_difference = third - second
        if (first_difference > 0 and second_difference > 0) or (
            first_difference < 0 and second_difference < 0
        ):
            monotonic_triplets += 1

        pattern = (first_difference, second_difference)
        reversed_pattern = (-second_difference, -first_difference)
        patterns.add(min(pattern, reversed_pattern))

    monotonic_score = 1 - monotonic_triplets / windows_count
    subsequence_score = len(patterns) / windows_count
    return monotonic_score * subsequence_score


def variance_score(
    numbers: Sequence[int | float | str], lower_bound: int, upper_bound: int
) -> float:
    """Возвращает близость дисперсии к равномерному распределению.

    Для равномерно распределённых целых чисел от ``lower_bound`` до
    ``upper_bound`` ожидаемая дисперсия равна ``(M ** 2 - 1) / 12``, где
    ``M`` — число возможных значений. Точное совпадение с ней даёт 1.0;
    при большем или меньшем разбросе оценка снижается до 0.0.
    """
    if lower_bound >= upper_bound:
        raise ValueError("Нижняя граница должна быть меньше верхней")

    lower = Decimal(lower_bound)
    upper = Decimal(upper_bound)
    values: list[Decimal] = []

    for number in numbers:
        try:
            value = Decimal(str(number))
        except (InvalidOperation, ValueError) as error:
            raise ValueError(f"Ожидалось число, получено: {number!r}") from error

        if not value.is_finite():
            raise ValueError(f"Число должно быть конечным, получено: {number!r}")
        if not lower <= value <= upper:
            raise ValueError(
                f"Число {number!r} вне допустимого диапазона "
                f"от {lower_bound} до {upper_bound}"
            )
        values.append(value)

    if not values:
        return 0.0

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    values_count = upper - lower + 1
    expected_variance = (values_count**2 - 1) / 12
    score = 1 - abs(variance - expected_variance) / expected_variance
    return max(0.0, min(1.0, float(score)))


def randomness_score(
    numbers: Sequence[int | float | str], lower_bound: int, upper_bound: int
) -> float:
    """Возвращает составной коэффициент случайности от 0.0 до 1.0.

    Критерий поровну учитывает соответствие закону Бенфорда, порядок чисел
    и близость дисперсии к ожидаемой.
    """
    return sum(randomness_criteria(numbers, lower_bound, upper_bound)) / 3


def randomness_criteria(
    numbers: Sequence[int | float | str], lower_bound: int, upper_bound: int
) -> tuple[float, float, float]:
    """Возвращает значения трёх составляющих критерия случайности."""
    return (
        benford_randomness_score(numbers),
        order_score(numbers, lower_bound, upper_bound),
        variance_score(numbers, lower_bound, upper_bound),
    )


def read_file_data() -> list[str]:
    project_dir = Path(__file__).resolve().parent
    candidates = (Path("data.txt"), project_dir / "data.txt")

    for path in candidates:
        if path.is_file():
            with path.open(encoding="utf-8") as file:
                return [
                    value
                    for line in file
                    if (value := line.strip()).isdigit() and len(value) == 6
                ]

    searched_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Не найден файл data.txt. Проверены: {searched_paths}")


def read_manual_data() -> list[str]:
    print("Введите данные по одному значению в строке. Пустая строка завершает ввод.")
    data: list[str] = []

    while True:
        try:
            value = input().strip()
        except EOFError:
            break

        if not value:
            break
        data.append(value)

    return data


def get_digit_groups(number: str | int) -> DigitGroups:
    """Преобразует число ABCDEF в числовой массив (A, BC, DEF)."""
    value = str(number)
    if len(value) != 6 or not value.isdigit():
        raise ValueError(f"Ожидалось шестизначное число, получено: {number!r}")

    return int(value[0]), int(value[1:3]), int(value[3:6])


def prepare_table_data(numbers: list[str | int]) -> list[DigitGroups]:
    """Подготавливает числа для таблицы в едином числовом формате."""
    return [get_digit_groups(number) for number in numbers]


def format_table(table_data: list[DigitGroups], generated_data: list[DigitGroups]) -> str:
    """Формирует таблицу табличного и алгоритмического способов."""
    rows = [
        from_file + generated
        for from_file, generated in zip(table_data, generated_data)
    ]
    rows = rows[:TABLE_ROWS]
    if not rows:
        return "Нет корректных шестизначных чисел для построения таблицы."

    headers = (
        "1 разряд",
        "2 и 3 разряды",
        "4, 5 и 6 разряды",
        "1 разряд",
        "2 и 3 разряды",
        "4, 5 и 6 разряды",
    )
    min_widths = (12, 17, 20, 12, 17, 20)
    widths = [
        max(min_widths[index], len(headers[index]), *(len(str(row[index])) for row in rows))
        for index in range(6)
    ]

    def line(left: str, separator: str, right: str) -> str:
        return left + separator.join("─" * (width + 2) for width in widths) + right

    def row(values: tuple[int | str, ...]) -> str:
        return "│" + "│".join(
            f" {value:^{width}} " for value, width in zip(values, widths)
        ) + "│"

    group_widths = (sum(widths[:3]) + 8, sum(widths[3:]) + 8)
    group_line = (
        "┌" + "─" * group_widths[0] + "┬" + "─" * group_widths[1] + "┐"
    )
    group_separator = (
        "├" + "─" * group_widths[0] + "┼" + "─" * group_widths[1] + "┤"
    )
    title_row = (
        "│"
        + f" {'Табличный способ':^{group_widths[0] - 2}} "
        + "│"
        + f" {'Алгоритмический':^{group_widths[1] - 2}} "
        + "│"
    )
    middle = "├" + "┬".join("─" * (width + 2) for width in widths) + "┤"
    table_criteria = tuple(
        randomness_criteria(
            [values[index] for values in table_data], *VALUE_RANGES[index]
        )
        for index in range(3)
    )
    generated_criteria = tuple(
        randomness_criteria(
            [values[index] for values in generated_data], *VALUE_RANGES[index + 3]
        )
        for index in range(3)
    )
    criteria_by_column = table_criteria + generated_criteria
    criteria_scores = tuple(
        tuple(
            f"{score:.4f}"
            for score in criteria
        )
        for criteria in criteria_by_column
    )
    total_scores = tuple(
        f"{sum(criteria) / 3:.4f}" for criteria in criteria_by_column
    )
    table_width = sum(widths) + 17
    def result_row(label: str) -> str:
        return f"│ {label:^{table_width - 2}} │"

    return "\n".join(
        [
            group_line,
            title_row,
            group_separator,
            row(headers),
            middle,
            *(row(values) if index == len(rows) - 1 else row(values) + "\n" + middle
              for index, values in enumerate(rows)),
            middle,
            result_row("Результат"),
            middle,
            result_row("Закон Бенфорда"),
            middle,
            row(tuple(scores[0] for scores in criteria_scores)),
            middle,
            result_row("Критерий порядка"),
            middle,
            row(tuple(scores[1] for scores in criteria_scores)),
            middle,
            result_row("Близость дисперсии к ожидаемой"),
            middle,
            row(tuple(scores[2] for scores in criteria_scores)),
            middle,
            result_row("Итоговый коэффициент"),
            middle,
            row(total_scores),
            line("└", "┴", "┘"),
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Чтение данных из data.txt или ввод с клавиатуры."
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="считать данные, введённые пользователем",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manual:
        data = read_manual_data()
        benford, order, variance = randomness_criteria(data, 0, 9)
        score = (benford + order + variance) / 3
        print(f"Закон Бенфорда: {benford:.4f}")
        print(f"Критерий порядка: {order:.4f}")
        print(f"Близость дисперсии к ожидаемой: {variance:.4f}")
        print(f"Коэффициент случайности: {score:.4f}")
        return

    data = read_file_data()
    table_data = prepare_table_data(data)
    generated_numbers = generate_random_numbers(len(data))
    generated_data = prepare_table_data(generated_numbers)
    print(format_table(table_data, generated_data))


if __name__ == "__main__":
    main()
