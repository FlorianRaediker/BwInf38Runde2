"""
Aufgabe 2: Geburtstag

Dieses Skript ist Teil der Einsendung für die

    2. Runde des
    38. Bundeswettbewerbs Informatik

von

    Florian Rädiker.

Teilnahme-ID: 52570
"""
import itertools
import math
import operator
import os
import time
import traceback
import warnings
from collections import ChainMap
from fractions import Fraction
from functools import lru_cache
from typing import Callable, Type, NamedTuple, Tuple, List, Optional, Dict, Any, Set, Iterable, Sequence, TextIO

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    warnings.warn("tqdm module is not installed")
    class tqdm:
        def __init__(self, *_, **__):
            pass
        def update(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
try:
    import pyfiglet
    # noinspection SpellCheckingInspection
    str2asciiart = pyfiglet.figlet_format
except ModuleNotFoundError:
    # noinspection SpellCheckingInspection
    def str2asciiart(x, *_, **__):
        return x


DO_PRINT = False  # whether to print output in BirthdayNumberFinder.find_*_term or not

ENABLE_NON_INTEGERS = True


class Operation:
    _SIGN: str = None
    _TERM_COUNT: int = None
    OPERATOR: Callable = None

    def __init__(self, *_):
        pass


class BinaryOperation(Operation):
    _TERM_COUNT = 2

#    def __init__(self, term1, term2):  # stats
#        super().__init__()  # stats
    def __init__(self, term1, term2):  # non-stats
        super().__init__()  # non-stats
        self.term1 = term1
        self.term2 = term2
#        self.factorials = self.term1.factorials | self.term2.factorials  # stats

    def __str__(self):
        return "(" + self.term1.__str__() + self._SIGN + self.term2.__str__() + ")"

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.term1.__repr__() + ", " + self.term2.__repr__() + ")"


class Summation(BinaryOperation):
    _SIGN = "+"
    OPERATOR = operator.add


class Subtraction(BinaryOperation):
    _SIGN = "-"
    OPERATOR = operator.sub


class Multiplication(BinaryOperation):
    _SIGN = "*"
    OPERATOR = operator.mul


class Division(BinaryOperation):
    _SIGN = "/"
    OPERATOR = Fraction


class Power(BinaryOperation):
    _SIGN = "^"
    OPERATOR = operator.pow


class UnaryOperation(Operation):
    _TERM_COUNT = 1

#    def __init__(self, term1):  # stats
#        super().__init__()  # stats
    def __init__(self, term1):  # non-stats
        super().__init__()  # non-stats
        self.term1 = term1

    def __str__(self):
        return "(" + self.term1.__str__() + self._SIGN + ")"

    def __repr__(self):
        return self.__class__.__name__ + "(" + self.term1.__repr__() + ")"


#class Number(UnaryOperation):  # stats
#    _SIGN = ""  # stats
#    OPERATOR = lambda n: n  # stats

#    def __init__(self, num):  # stats
#        super().__init__(num)  # stats
#        self.factorials = set()  # stats

#    def __str__(self):  # stats
#        return self.term1.__str__()  # stats


class Factorial(UnaryOperation):
    _SIGN = "!"
    OPERATOR = math.factorial

#    def __init__(self, num, term1):  # stats
#        super().__init__(term1)  # stats
#        self.factorials = term1.factorials.copy()  # stats
#        self.factorials.add(num)  # stats


class BirthdayNumberFinder:
    digit: int
    str_digit: str
    max_number: int
    found_simple_terms: Dict[int, Tuple[Any, int, Optional[int], set]]
    simple_search_count_limit: int
    extended_terms: List[Optional[list]]
    taboo_numbers: Set[int]

    def __init__(self, digit, max_number, max_simple_count):
        self.digit = digit
        self.max_number = max_number
        self.max_simple_count = max_simple_count
        self.found_simple_terms = {}
        self.str_digit = str(self.digit)
        # fill self.found_simple_terms with repdigits
        for length in range(1, max_simple_count + 1):
            repdigit = get_repdigit(self.str_digit, length)
            self.found_simple_terms[repdigit] = (repdigit, length, None, set())
        # the algorithms do not normally need a term for 0; the following is only necessary if 0 is passed to find_term
        #   or find_extended_term directly
        self.found_simple_terms[0] = (Subtraction(self.digit, self.digit), 2, None, set())  # non-stats
#        self.found_simple_terms[0] = (Subtraction(Number(self.digit), Number(self.digit)), 2, None, set())  # stats
        self.found_extended_terms = {}
        self.all_found_terms = ChainMap(self.found_extended_terms, self.found_simple_terms)
        self.simple_search_count_limit = 0
        self.taboo_numbers = set()

    def search_terms(self, extended_count_limit, simple_count_limit, max_found_term):
        assert extended_count_limit <= simple_count_limit, \
            "extended_count_limit must be smaller than or equal to simple_count_limit"

        MAX_FACTORIAL_N = 70  # non-stats
#        MAX_FACTORIAL_N = 100  # stats

        found_simple_numbers = set(self.found_simple_terms.keys())  # keys are repdigits (filled in __init__)
        found_extended_numbers = set()

        def add_simple_number(operand1, operand2, term1, term2, operation: Type[Operation]):
            nonlocal biggest_num
            num = operation.OPERATOR(operand1, operand2)
            # Überprüfung der Schnapszahlen-Optimalität
            if (c := (str_num := str(num)).count(self.str_digit)) == len(str_num) and c > count:
                print(f"WARNING: Found better term for repdigit {self.str_digit*c}: {operation(term1, term2)}")
            if num > max_num:
                return
            if num not in found_simple_numbers:
                found_simple_numbers.add(num)
                term = operation(term1, term2)
                if num > biggest_num:
                    biggest_num = num
                if type(num) == Fraction:
                    if num.denominator == 1:
                        num = num.numerator
                        is_int = True
                    else:
                        is_int = False
                else:
                    is_int = True
                if is_int:
                    if num <= max_found_term:
                        self.found_simple_terms[num] = (term, count, None, set())
                if num not in found_extended_numbers:
                    new_simple_terms.append((num, term))
                    if is_int:
                        add_factorial(num, term)
                else:
                    # term is worse than some extended term
                    new_bad_simple_terms.append((num, term))

        def add_extended_number(operand1, operand2, term1, term2, operation: Type[Operation]):
            nonlocal biggest_num
            num = operation.OPERATOR(operand1, operand2)
            # Überprüfung der Schnapszahlen-Optimalität
            if (c := (str_num := str(num)).count(self.str_digit)) == len(str_num) and c > count:
                print(f"WARNING: Found better term for repdigit {self.str_digit*c}: {operation(term1, term2)}")
            if num > max_num:
                return
            if num not in found_simple_numbers and num not in found_extended_numbers:
                found_extended_numbers.add(num)
                term = operation(term1, term2)
                if num > biggest_num:
                    biggest_num = num
                if type(num) == Fraction:
                    if num.denominator == 1:
                        # num is a whole number
                        num = num.numerator
                        is_int = True
                    else:
                        is_int = False
                else:
                    is_int = True
                if is_int:
                    if num <= max_found_term:
                        self.found_extended_terms[num] = (term, count)
                    add_factorial(num, term)
                new_extended_terms.append((num, term))

        def add_simple_number_without_factorial(operand1, operand2, term1, term2, operation: Type[Operation]):
            nonlocal biggest_num
            num = operation.OPERATOR(operand1, operand2)
            # Überprüfung der Schnapszahlen-Optimalität
            if (c := (str_num := str(num)).count(self.str_digit)) == len(str_num) and c > count:
                print(f"WARNING: Found better term for repdigit {self.str_digit*c}: {operation(term1, term2)}")
            if num > max_num:
                return
            if num not in found_simple_numbers:
                found_simple_numbers.add(num)
                term = operation(term1, term2)
                if num > biggest_num:
                    biggest_num = num
                if type(num) == Fraction:
                    if num.denominator == 1:
                        # num is a whole number
                        num = num.numerator
                        is_int = True
                    else:
                        is_int = False
                else:
                    is_int = True
                if is_int and num <= max_found_term:
                    self.found_simple_terms[num] = (term, count, None, set())
                if num not in found_extended_numbers:
                    new_simple_terms.append((num, term))
                else:
                    new_bad_simple_terms.append((num, term))

        def add_extended_number_without_factorial(operand1, operand2, term1, term2, operation: Type[Operation]):
            nonlocal biggest_num
            num = operation.OPERATOR(operand1, operand2)
            # Überprüfung der Schnapszahlen-Optimalität
            if (c := (str_num := str(num)).count(self.str_digit)) == len(str_num) and c > count:
                print(f"WARNING: Found better term for repdigit {self.str_digit*c}: {operation(term1, term2)}")
            if num > max_num:
                return
            if num not in found_simple_numbers and num not in found_extended_numbers:
                found_extended_numbers.add(num)
                term = operation(term1, term2)
                if num > biggest_num:
                    biggest_num = num
                if type(num) == Fraction:
                    if num.denominator == 1:
                        num = num.numerator
                        is_int = True
                    else:
                        is_int = False
                else:
                    is_int = True
                if is_int and num <= max_found_term:
                    self.found_extended_terms[num] = (term, count)
                new_extended_terms.append((num, term))

        def add_factorial(num, term):
            nonlocal biggest_num
            if 2 < num <= MAX_FACTORIAL_N:
                factorial = math.factorial(num)
                # Überprüfung der Schnapszahlen-Optimalität
                if (c := (str_num := str(num)).count(self.str_digit)) == len(str_num) and c > count:
                    print(f"WARNING: Found better term for repdigit {self.str_digit * c}: {Factorial(term)}")
                if factorial > max_num:
                    return
                if factorial not in found_simple_numbers and factorial not in found_extended_numbers:
                    found_extended_numbers.add(factorial)
                    factorial_term = Factorial(term)  # non-stats
#                    factorial_term = Factorial(num, term)  # stats
                    if factorial > biggest_num:
                        biggest_num = factorial
                    new_extended_terms.append((factorial, factorial_term))
                    if factorial <= max_found_term:
                        self.found_extended_terms[factorial] = (factorial_term, count)
                    add_factorial(factorial, factorial_term)

        def combine_extended_terms(terms1: Iterable, terms2: Sequence):
            # all combinations are extended terms
            for t1_num, t1 in terms1:
                for t2_num, t2 in terms2:
                    pbar.update()
                    # SUMMATION
                    add_extended_number(t1_num, t2_num, t1, t2, Summation)

                    # MULTIPLICATION
                    add_extended_number(t1_num, t2_num, t1, t2, Multiplication)

                    # DIVISION
                    if ENABLE_NON_INTEGERS or t1_num % t2_num == 0:
                        add_extended_number(t1_num, t2_num, t1, t2, Division)
                    if ENABLE_NON_INTEGERS or t2_num % t1_num == 0:
                        add_extended_number(t2_num, t1_num, t2, t1, Division)

                    # SUBTRACTION
                    if t1_num > t2_num:
                        add_extended_number(t1_num, t2_num, t1, t2, Subtraction)
                    elif t1_num != t2_num:
                        add_extended_number(t2_num, t1_num, t2, t1, Subtraction)

                    # POWER
                    if t2_num < 100 and type(t1) != Power and type(t2_num) != Fraction:
                        add_extended_number(t1_num, t2_num, t1, t2, Power)
                    if t1_num < 100 and type(t2) != Power and type(t1_num) != Fraction:
                        add_extended_number(t2_num, t1_num, t2, t1, Power)

        def combine_simple_terms(terms1: Iterable, terms2: Callable[[], Iterable]):
            # all combinations are simple terms, but the terms may be combined to extended terms
            for t1_num, t1 in terms1:
                for t2_num, t2 in terms2():
                    pbar.update()
                    # SUMMATION
                    add_simple_number(t1_num, t2_num, t1, t2, Summation)

                    # MULTIPLICATION
                    add_simple_number(t1_num, t2_num, t1, t2, Multiplication)

                    # DIVISION
                    if ENABLE_NON_INTEGERS or t1_num % t2_num == 0:
                        add_simple_number(t1_num, t2_num, t1, t2, Division)
                    if ENABLE_NON_INTEGERS or t2_num % t1_num == 0:
                        add_simple_number(t2_num, t1_num, t2, t1, Division)

                    # SUBTRACTION
                    if t1_num > t2_num:
                        add_simple_number(t1_num, t2_num, t1, t2, Subtraction)
                    elif t1_num != t2_num:
                        add_simple_number(t2_num, t1_num, t2, t1, Subtraction)

                    # POWER
                    if t2_num < 100 and type(t1) != Power and type(t2_num) != Fraction:
                        add_extended_number(t1_num, t2_num, t1, t2, Power)
                    if t1_num < 100 and type(t2) != Power and type(t1_num) != Fraction:
                        add_extended_number(t2_num, t1_num, t2, t1, Power)

        def combine_extended_terms_only_basic_operations(terms1: Iterable, terms2: Sequence):
            # all combinations are extended terms, but the terms may only be combined using basic arithmetic operations
            for t1_num, t1 in terms1:
                for t2_num, t2 in terms2:
                    pbar.update()
                    # SUMMATION
                    add_extended_number_without_factorial(t1_num, t2_num, t1, t2, Summation)

                    # MULTIPLICATION
                    add_extended_number_without_factorial(t1_num, t2_num, t1, t2, Multiplication)

                    # DIVISION
                    if ENABLE_NON_INTEGERS or t1_num % t2_num == 0:
                        add_extended_number_without_factorial(t1_num, t2_num, t1, t2, Division)
                    if ENABLE_NON_INTEGERS or t2_num % t1_num == 0:
                        add_extended_number_without_factorial(t2_num, t1_num, t2, t1, Division)

                    # SUBTRACTION
                    if t1_num > t2_num:
                        add_extended_number_without_factorial(t1_num, t2_num, t1, t2, Subtraction)
                    elif t1_num != t2_num:
                        add_extended_number_without_factorial(t2_num, t1_num, t2, t1, Subtraction)

        def combine_simple_terms_only_basic_operations(terms1: Iterable, terms2: Callable[[], Iterable]):
            # all combinations are simple terms and the terms may only be combined using basic arithmetic operations
            for t1_num, t1 in terms1:
                for t2_num, t2 in terms2():
                    pbar.update()
                    # SUMMATION
                    add_simple_number_without_factorial(t1_num, t2_num, t1, t2, Summation)

                    # MULTIPLICATION
                    add_simple_number_without_factorial(t1_num, t2_num, t1, t2, Multiplication)

                    # DIVISION
                    if ENABLE_NON_INTEGERS or t1_num % t2_num == 0:
                        add_simple_number_without_factorial(t1_num, t2_num, t1, t2, Division)
                    if ENABLE_NON_INTEGERS or t2_num % t1_num == 0:
                        add_simple_number_without_factorial(t2_num, t1_num, t2, t1, Division)

                    # SUBTRACTION
                    if t1_num > t2_num:
                        add_simple_number_without_factorial(t1_num, t2_num, t1, t2, Subtraction)
                    elif t1_num != t2_num:
                        add_simple_number_without_factorial(t2_num, t1_num, t2, t1, Subtraction)

        def new_progress_bar(c1, c2):
            return tqdm(total=(len(simple_terms[c1]) + len(bad_simple_terms[c1])) *
                              (len(simple_terms[c2]) + len(bad_simple_terms[c2]))
                               + len(self.extended_terms[c1]) * len(self.extended_terms[c2])
                               + len(self.extended_terms[c1]) * len(simple_terms[c2])
                               + len(simple_terms[c1]) * len(self.extended_terms[c2]), leave=False)

        def to_scientific(number: Any, ndigits: int = 6):
            """ f"{number:e}" does not work for large ints. This function does """
            if type(number) == Fraction and number.denominator == 1:
                number = number.numerator
            if type(number) == int:
                string = str(number)
                return f"{string[0]}.{string[1:1+ndigits]}e+{str(len(string)-1):0>2}"
            return str(number)

        self.simple_search_count_limit = simple_count_limit

        simple_terms = [[] for _ in range(simple_count_limit + 1)]
        bad_simple_terms = [[] for _ in range(simple_count_limit + 1)]
        self.extended_terms = [[] for _ in range(simple_count_limit + 1)]
        biggest_nums = [math.inf for _ in range(self.max_simple_count + 1)]  # save the biggest number per digit count
        biggest_nums[0] = 1

        for count in range(1, extended_count_limit + 1):
            max_num = biggest_nums[self.max_simple_count - count] * self.max_number
            print(f"DIGIT COUNT {count}, maximum number is "
                  f"~{to_scientific(biggest_nums[self.max_simple_count - count])} * {self.max_number} ≈ "
                  f"{to_scientific(max_num)}")
            new_simple_terms = simple_terms[count]
            new_bad_simple_terms = bad_simple_terms[count]
            new_extended_terms = self.extended_terms[count]
            repdigit_num = get_repdigit(self.str_digit, count)
            if repdigit_num <= max_num:
                biggest_num = repdigit_num
#                repdigit_num_term = Number(repdigit_num)  # stats
                repdigit_num_term = repdigit_num  # non-stats
                new_simple_terms.append((repdigit_num, repdigit_num_term))
                add_factorial(repdigit_num, repdigit_num_term)
            else:
                biggest_num = 0
            for c1, c2 in zip(
                    range(1, math.floor(count / 2) + 1),
                    range(count - 1, math.ceil(count / 2) - 1, -1)
            ):
                print("count", count, " = count", c1, " + count", c2, sep="")
                with new_progress_bar(c1, c2) as pbar:
                    combine_simple_terms(itertools.chain(simple_terms[c1], bad_simple_terms[c1]),
                                            lambda: itertools.chain(simple_terms[c2], bad_simple_terms[c2]))
                    combine_extended_terms(self.extended_terms[c1], self.extended_terms[c2])
                    combine_extended_terms(self.extended_terms[c1], simple_terms[c2])
                    combine_extended_terms(simple_terms[c1], self.extended_terms[c2])
            biggest_nums[count] = biggest_num
            print(f"biggest number for count{count} is ~{to_scientific(biggest_num)}")

        for count in range(extended_count_limit + 1, simple_count_limit + 1):
            max_num = biggest_nums[self.max_simple_count - count] * self.max_number
            print(f"DIGIT COUNT {count}, maximum number is "
                  f"~{to_scientific(biggest_nums[self.max_simple_count - count])} * {self.max_number} ≈ "
                  f"{to_scientific(max_num)}")
            new_simple_terms = simple_terms[count]
            new_bad_simple_terms = []
            new_bad_simple_terms = bad_simple_terms[count]
            new_extended_terms = []
            new_extended_terms = self.extended_terms[count]
            repdigit_num = get_repdigit(self.str_digit, count)
            if repdigit_num <= max_num:
                biggest_num = repdigit_num
#                repdigit_num_term = Number(repdigit_num)  # stats
                repdigit_num_term = repdigit_num  # non-stats
                new_simple_terms.append((repdigit_num, repdigit_num_term))
                add_factorial(repdigit_num, repdigit_num_term)
            else:
                biggest_num = 0

            for c1, c2 in zip(
                    range(1, math.floor(count / 2) + 1),
                    range(count - 1, math.ceil(count / 2) - 1, -1)
            ):
                print("count", count, " = count", c1, " + count", c2, sep="")
                with new_progress_bar(c1, c2) as pbar:
                    combine_simple_terms_only_basic_operations(
                        itertools.chain(simple_terms[c1], bad_simple_terms[c1]),
                        lambda: itertools.chain(simple_terms[c2], bad_simple_terms[c2]))
                    combine_extended_terms_only_basic_operations(self.extended_terms[c1], self.extended_terms[c2])
                    combine_extended_terms_only_basic_operations(self.extended_terms[c1], simple_terms[c2])
                    combine_extended_terms_only_basic_operations(simple_terms[c1], self.extended_terms[c2])
            biggest_nums[count] = biggest_num
            print(f"biggest number for count{count} is ~{to_scientific(biggest_num)}")

    def save_initial_terms(self):
        """
        Saves the Term d/d=1 in self.found_simple_terms if self.digit != 1.

        DIESE METHODE MUSS AUFGERUFEN WERDEN, WENN TIEFENSUCHE OHNE VORHERIGE TERMBILDUNG GENUTZT WIRD. Ansonsten ist
        die abgerundete Zahl bei der Zerlegung (analyse_number) 0 und range_with_required_digits wird mit begin=0
        aufgerufen. Das führt zu einem 'ValueError: math domain error', weil versucht wird, math.log10(0) zu berechnen.
        """
        if self.digit != 1:
            self.found_simple_terms[1] = (Division(self.digit, self.digit), 2, None, set())

    def analyse_number(self, number):
        results = []  # all terms with summation and multiplication, n-th element contains terms with digit count n

        # MUL
        for factor1 in factors(number):
            min_digit_count1 = get_min_required_digits(factor1, self.str_digit)
            factor2 = number // factor1
            min_digit_count2 = get_min_required_digits(factor2, self.str_digit)
            count = min_digit_count1 + min_digit_count2
            term = Multiplication((factor1, min_digit_count1), (factor2, min_digit_count2))
            try:
                results[count].append(term)
            except IndexError:
                results.extend([] for _ in range(count - len(results) + 1))
                results[count].append(term)

        # SUM
        half_number = number / 2
        for (addend1, min_digit_count1), (addend2, min_digit_count2) in zip(
                range_with_required_digits_from(math.floor(half_number), self.digit, self.str_digit),
                range_with_required_digits(math.ceil(half_number), number - 1, self.digit, self.str_digit),
        ):
            count = min_digit_count1 + min_digit_count2
            term = Summation((addend1, min_digit_count1), (addend2, min_digit_count2))
            try:
                results[count].append(term)
            except IndexError:
                results.extend([] for _ in range(count - len(results) + 1))
                results[count].append(term)

        def sub():
            # generator for subtraction terms
            for (minuend, minuend_min_count), (subtrahend, subtrahend_min_count) in zip(
                    range_with_required_digits_unlimited(number + 1, self.digit, self.str_digit),
                    range_with_required_digits_unlimited(1, self.digit, self.str_digit)
            ):
                yield (minuend_min_count + subtrahend_min_count,
                       Subtraction((minuend, minuend_min_count), (subtrahend, subtrahend_min_count)))

        def div():
            # generator for division terms
            dividend_min_count = get_min_required_digits(number*2, self.str_digit)  # first dividend is number*2
            next_dividend_repdigit = get_repdigit(self.str_digit, dividend_min_count)

            for divisor, divisor_min_count in range_with_required_digits_unlimited(2, self.digit, self.str_digit):
                dividend = number * divisor
                if dividend > next_dividend_repdigit:
                    dividend_min_count += 1
                    next_dividend_repdigit = next_dividend_repdigit * 10 + self.digit
                yield (divisor_min_count + dividend_min_count,
                       Division((dividend, dividend_min_count), (divisor, divisor_min_count)))

        sub_gen = sub()
        div_gen = div()
        # sub_count is the count for term next_sub
        sub_count, next_sub = next(sub_gen)
        div_count, next_div = next(div_gen)

        for count in itertools.count(2):
            yield None  # yield None to indicate start of new count
            yield count
            try:
                # yield all results from summation and multiplication
                yield from results[count]
            except IndexError:
                # results is not long enough, so there are no summation or multiplication terms with this count
                pass
            if sub_count == count:
                # yield subtraction terms until next_count differs from count
                yield next_sub
                for sub_count, next_sub in sub_gen:
                    if sub_count != count:
                        break
                    yield next_sub
            if div_count == count:
                yield next_div
                for div_count, next_div in div_gen:
                    if div_count != count:
                        break
                    yield next_div

    def find_simple_term(self, number, max_count, best_possible_count, indent=""):
#        raise ValueError("Stats are enabled, can't find term")  # stats
        if DO_PRINT and len(indent) < 8: print(indent, "SEARCH ", number, " max_count: ", max_count, " taboo: ",
                                               self.taboo_numbers, sep="")

        def new_term(term):
            nonlocal best_term, best_count
            if term.term1[0] < term.term2[0]:
                small_term, small_min_count = term.term1
                big_term, big_min_count = term.term2
                swap = False
            else:
                small_term, small_min_count = term.term2
                big_term, big_min_count = term.term1
                swap = True
            term1, length1 = self.find_simple_term(small_term, best_count - big_min_count - 1, small_min_count,
                                                   indent + "    ")
            if term1 is not None:
                term2, length2 = self.find_simple_term(big_term, best_count - length1 - 1, big_min_count, indent +
                                                       "    ")
                if term2 is not None:
                    best_count = length1 + length2
                    best_term = term
                    if swap:
                        term.term1 = term2
                        term.term2 = term1
                    else:
                        term.term1 = term1
                        term.term2 = term2
                    return True
            return False

        if max_count < best_possible_count:
            return None, math.inf

        if number in self.found_simple_terms:
            best_term, best_count, old_max_count, old_taboo_numbers = self.found_simple_terms[number]

            if old_taboo_numbers.issubset(self.taboo_numbers):
                if best_term is not None:
                    if best_count <= max_count:
                        return best_term, best_count
                    return None, math.inf
                if max_count <= old_max_count:
                    return None, math.inf
            if best_term is None and old_max_count < max_count:
                best_count = max_count + 1
                old_taboo_numbers = None
            if best_count > max_count + 1:
                best_term = None
                best_count = max_count + 1
        else:
            if max_count <= self.simple_search_count_limit:
                return None, math.inf
            old_taboo_numbers = None
            best_count = max_count + 1
            best_term = None

        self.taboo_numbers.add(number)

        term_generator = self.analyse_number(number)
        while True:
            term = next(term_generator)
            while term is None:
                # new count starts here
                min_count = next(term_generator)
                if min_count >= best_count:
                    break
                term = next(term_generator)
            else:
                # 'else' block is entered if while loop did not break
                if DO_PRINT and len(indent) < 8: print(indent, " ", min_count, " TERM ", term, sep="")
                if term.term1[0] not in self.taboo_numbers and term.term2[0] not in self.taboo_numbers:
                    if new_term(term):
                        if min_count == best_count:
                            break
                continue
            break
        self.taboo_numbers.remove(number)

        taboo_numbers = self.taboo_numbers.copy()
        if old_taboo_numbers:
            new_taboo_numbers = taboo_numbers & old_taboo_numbers
            taboo_numbers = new_taboo_numbers

        res = (best_term, best_count, max_count, taboo_numbers)
        self.found_simple_terms[number] = res
        if DO_PRINT and len(indent) < 8: print(indent, "RESULT ", number, " ", res, sep="")
        return best_term, best_count

    def find_extended_term(self, number, best_term, best_count, indent=""):
#        raise ValueError("Stats are enabled, can't find term")  # stats
        extended_term_count = 1
        if number in self.found_extended_terms:
            return self.found_extended_terms[number]
        while extended_term_count < best_count-1:
            max_count2 = best_count - extended_term_count - 1
            try:
                extended_terms = self.extended_terms[extended_term_count]
            except IndexError:
                break
            for num, term in extended_terms:
                if num < number:
                    # SUM
                    addend2 = number - num
                    if addend2 in self.all_found_terms:
                        term_addend2, addend2_count, *_ = self.all_found_terms[addend2]
                        if term_addend2 is not None and addend2_count <= max_count2:
                            best_count = extended_term_count + addend2_count
                            best_term = Summation(term, term_addend2)

                    # DIV
                    # dividend / num = number <=> dividend = number * num
                    dividend = number * num
                    if dividend in self.all_found_terms:
                        term_dividend, dividend_count, *_ = self.all_found_terms[dividend]
                        if term_dividend is not None and dividend_count <= max_count2:
                            best_count = extended_term_count + dividend_count
                            best_term = Division(term_dividend, term)
                else:
                    # SUB
                    # num - subtrahend = number <=> subtrahend = num - number
                    subtrahend = num - number
                    if subtrahend in self.all_found_terms:
                        term_subtrahend, subtrahend_count, *_ = self.all_found_terms[subtrahend]
                        if term_subtrahend is not None and subtrahend_count <= max_count2:
                            best_count = extended_term_count + subtrahend_count
                            best_term = Subtraction(term, term_subtrahend)
                    # DIV
                    # num / divisor = number <=> divisor = num / number
                    if num % number == 0:
                        divisor = num // number
                        if divisor in self.all_found_terms:
                            term_divisor, divisor_count, *_ = self.all_found_terms[divisor]
                            if term_divisor is not None and divisor_count <= max_count2:
                                best_count = extended_term_count + divisor_count
                                best_term = Division(term, term_divisor)

                # SUB
                # minuend - num = number <=> minuend = num + number
                minuend = num + number
                if minuend in self.all_found_terms:
                    term_minuend, minuend_count, *_ = self.all_found_terms[minuend]
                    if term_minuend is not None and minuend_count <= max_count2:
                        best_count = extended_term_count + minuend_count
                        best_term = Subtraction(term_minuend, term)

                # MUL
                if number % num == 0:
                    factor = number // num
                    if factor in self.all_found_terms:
                        term_factor, factor_count, *_ = self.all_found_terms[factor]
                        if term_factor is not None and factor_count <= max_count2:
                            best_count = extended_term_count + factor_count
                            best_term = Multiplication(term, term_factor)
            extended_term_count += 1

        res = (best_term, best_count)
        self.found_extended_terms[number] = res
        if DO_PRINT: print(indent, "RESULT ", number, " (", best_term, ", ", best_count, ")", sep="")
        return res

    def print_factorial_terms(self, out: TextIO):
        raise ValueError("Stats are disabled, can't print factorial terms")  # non-stats
        out.write("Alle Terme, die Fakultäten enthalten und eine Zahl darstellen, die kleiner als 100000 ist, "
                  "mit Ziffer " + self.str_digit + "\n")
        all_n_values = set()
        for n_values, num, term in sorted(
                ((term.factorials, num, term) for extended_terms in self.extended_terms
                 for num, term in extended_terms if num < 10000 and type(num) != Fraction and term.factorials),
                key=operator.itemgetter(1)):
            out.write("n: " + str(n_values) + "; " + str(num) + "=" + str(term) + "\n")
            all_n_values.update(n_values)
        out.write("n!: n ∈ {" + ", ".join(str(n) for n in sorted(all_n_values)) + "}" + "\n")


@lru_cache(maxsize=None)
def get_repdigit(str_digit: str, length: int):
    """
    >>> get_repdigit("3", 10)
    3333333333

    :param str_digit:
    :param length:
    :return:
    """
    """
    Using strings to create repdigits is faster than calculating (works for other lengths instead of 10 as well). 
    >>> timeit.timeit("f(10, '2')", "f=lambda l,d:int(d*l)")
    0.5736585570002717
    >>> timeit.timeit("f(10, 2)", "f=lambda l,d:d*((10**l-1)//9)")
    1.0174948559997574
    """
    return int(str_digit * length)


@lru_cache(maxsize=None)
def get_digit_count(number: int):
    return math.floor(math.log10(number)) + 1


@lru_cache(maxsize=None)
def get_min_required_digits(number: int, str_digit: str):
    length = get_digit_count(number)
    repdigit = get_repdigit(str_digit, length)
    if number <= repdigit:
        return length
    return length + 1


def range_with_required_digits(begin: int, end: int, digit: int, str_digit: str):
    do_return = False
    # range to the next repdigit
    current_count = get_digit_count(begin)
    next_repdigit = get_repdigit(str_digit, current_count)
    if begin > next_repdigit:
        current_count += 1
        next_repdigit = next_repdigit * 10 + digit
    if end <= next_repdigit:
        next_repdigit = end
        do_return = True
    for x in range(begin, next_repdigit + 1):
        yield x, current_count
    if do_return:
        return
    current_repdigit = next_repdigit
    current_count += 1
    while True:
        next_repdigit = current_repdigit * 10 + digit
        if end <= next_repdigit:
            next_repdigit = end
            do_return = True
        for x in range(current_repdigit + 1, next_repdigit + 1):
            yield x, current_count
        if do_return:
            return
        current_count += 1
        current_repdigit = next_repdigit


def range_with_required_digits_unlimited(start: int, digit: int, str_digit: str):
    # range to the next repdigit
    current_count = get_digit_count(start)
    next_repdigit = get_repdigit(str_digit, current_count)
    if start > next_repdigit:
        current_count += 1
        next_repdigit = next_repdigit * 10 + digit
    for x in range(start, next_repdigit + 1):
        yield x, current_count
    current_repdigit = next_repdigit
    current_count += 1
    while True:
        next_repdigit = current_repdigit * 10 + digit
        for x in range(current_repdigit + 1, next_repdigit + 1):
            yield x, current_count
        current_count += 1
        current_repdigit = next_repdigit


def range_with_required_digits_from(begin: int, digit: int, str_digit: str):
    current_count = get_digit_count(begin)
    repdigit = get_repdigit(str_digit, current_count)
    if begin > repdigit:
        current_count += 1
        repdigit = repdigit * 10 + digit
    current_repdigit = begin
    next_repdigit = repdigit // 10
    while True:
        for x in range(current_repdigit, next_repdigit, -1):
            yield x, current_count
        current_count -= 1
        if current_count == 0:
            return
        current_repdigit = next_repdigit
        next_repdigit = int(current_repdigit / 10)


def factors(number):
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            yield i


class DigitForSearching(NamedTuple):
    digit: int
    extended_count_limit: int
    simple_count_limit: int
    simple_max_counts: List[int]
    extended_max_counts: List[int]


def main(save_path=None, wait_for_enter=input):  # specify `wait_for_enter=lambda x: None` to remove waiting for enter
    numbers = (2019, 2020, 2030, 2080, 2980)
    digits = (
        DigitForSearching(1, 4, 8, [11, 10, 12, 12, 13], [11, 10, 10, 10, 11]),
        DigitForSearching(2, 4, 6, [10, 8, 9, 9, 11], [9, 8, 8, 8, 8]),
        DigitForSearching(3, 2, 4, [ 7, 9, 9, 9,  9], [5, 6, 6, 5, 7]),
        DigitForSearching(4, 2, 5, [10, 8, 10, 7,  9], [7, 5, 7, 5, 5]),
        DigitForSearching(5, 2, 5, [10, 8, 8, 8,  8], [8, 7, 7, 7, 5]),
        DigitForSearching(6, 2, 6, [9, 10, 10, 10, 11], [8, 7, 7, 8, 8]),
        DigitForSearching(7, 1, 7, [10, 9, 8, 9, 11], [9,
                                                       8,  # 9 with only whole numbers
                                                       7, 9,
                                                       8  # 9 with only whole numbers
                                                       ]),
        DigitForSearching(8, 1, 7, [11, 9, 10, 8, 11], [9, 9, 8, 8, 9]),
        DigitForSearching(9, 6, 6, [10, 9, 10, 9, 10], [8, 9, 8, 8, 9]),
    )

    max_number = 2980

    all_results = []
    for digit_search in digits:
        print(str2asciiart("digit  =  " + str(digit_search.digit)))
        wait_for_enter("Press enter to start term search.")

        simple_max_count = max(digit_search.simple_max_counts)

        print("Searching terms...")
        finder = BirthdayNumberFinder(digit_search.digit, max_number, simple_max_count)
        t1 = time.perf_counter_ns()
        finder.search_terms(digit_search.extended_count_limit, digit_search.simple_count_limit, 1000000)
        t2 = time.perf_counter_ns()
        time_search = t2-t1
        print(f"\nsearched terms in {time_search/1e+9}s")

        # SIMPLE TERMS
        print("\n\n############\nSIMPLE TERMS")

        t1_simple = time.perf_counter_ns()
        simple_results = []
        for number, max_count_for_number in zip(numbers, digit_search.simple_max_counts):
            print("\nFinding term for", number)
            t1 = time.perf_counter_ns()
            res = finder.find_simple_term(number, max_count_for_number,
                                          get_min_required_digits(number, str(digit_search.digit)))
            if res[1] != max_count_for_number:
                print("WARNING: SIMPLE TERM HAS UNEXPECTED DIGIT COUNT: got", res[1], "expected:", max_count_for_number)
            t2 = time.perf_counter_ns()
            print(f"  found term in {(t2-t1)/1e+9:.4f}s")
            print(number, " = ", res[0], " (digit count: ", res[1], ")", sep="")
            simple_results.append(res)
        t2_simple = time.perf_counter_ns()
        time_simple = t2_simple-t1_simple
        print(f"\nfound simple terms in {time_simple/1e+9:.4f}s\n")

        wait_for_enter("Simple terms finished. Press enter.")

        # EXTENDED TERMS
        print("\n\n############\nEXTENDED TERMS")

        t1_extended = time.perf_counter_ns()
        extended_results = []
        for number, max_count_for_number, (best_simple_term, best_simple_count) in \
                zip(numbers, digit_search.extended_max_counts, simple_results):
            print("\nFinding term for", number)
            t1 = time.perf_counter_ns()
            res = finder.find_extended_term(number, best_simple_term, best_simple_count)
            t2 = time.perf_counter_ns()
            print(f"  found term in {(t2 - t1) / 1e+9:.4f}s")
            print(number, " = ", res[0], " (digit count: ", res[1], ")", sep="", end="\n\n")
            if res[1] != max_count_for_number:
                print("WARNING: EXTENDED TERM HAS UNEXPECTED DIGIT COUNT: got", res[1],
                      "expected:", max_count_for_number)
            extended_results.append(res)
        t2_extended = time.perf_counter_ns()
        time_extended = t2_extended-t1_extended
        print(f"found extended terms in {time_extended/1e+9:.4f}s")

        print(f"complete time: {(time_search+time_simple+time_extended)/1e+9:.4f}s\n")
        wait_for_enter(f"Found all terms for digit {digit_search.digit}. Press enter.")

        if save_path:
            try:
                with open(os.path.join(save_path, f"digit{digit_search.digit}.txt"), "w") as f:
                    for number, simple_result, extended_result in zip(numbers, simple_results, extended_results):
                        for name, result in (("simple  ", simple_result), ("extended", extended_result)):
                            f.write(f"{name}: {number} = {result[0]} (digit count: {result[1]})\n")
                        f.write("\n")
            except Exception:
                warnings.warn("Saving to file failed")
                traceback.print_exc()
        all_results.append((digit_search.digit, simple_results, extended_results))


def save_extended_terms_up_to_number(digit, limit, extended_count_limit, simple_count_limit, filename):
    finder = BirthdayNumberFinder(digit, limit, simple_count_limit)
    finder.search_terms(extended_count_limit, simple_count_limit, limit)
    with open(filename, "w", encoding="utf-8") as f:
        for i in range(1, limit+1):
            if i in finder.all_found_terms:
                print(i)
                res = finder.all_found_terms[i]
                print("", res[0], res[1])
                f.write(str(i) + "\t" + str(res[0]) + "\t" + str(res[1]) + "\n")#


if __name__ == "__main__":
    # TERME FÜR VORGEGEBENE ZAHLEN BERECHNEN
    main(save_path="results")

    # (FAST) ALLE ERWEITERTEN TERME BIS 3000 SUCHEN UND IM VERZEICHNIS 'extended_terms' SPEICHERN
    """for digit, extended_count_limit, simple_count_limit in (
            (1, 11, 11), (2, 9, 9), (3, 7, 7), (4, 6, 6), (5, 7, 8), (6, 7, 8), (7, 7, 8), (8, 7, 8), (9, 7, 8),
            ):
        print(str2asciiart("digit  =  " + str(digit)))
        save_extended_terms_up_to_number(digit, 3000, extended_count_limit, simple_count_limit,
                                         f"extended_terms/digit{digit}.txt")"""

    # HERAUSFINDEN, WELCHE FAKULTÄTEN VERWENDET WERDEN UND ERGEBNISSE IM VERZEICHNIS 'factorials_in_terms_max100'
    # SPEICHERN
    # ACHTUNG: DAMIT FOLGENDER CODE FUNKTIONIERT, BITTE STATS "EINSCHALTEN" (siehe Dokumentation)
    # INFO: Für die Ergebnisse im Ordner 'factorials_in_terms_max200' wurde MAX_FACTORIAL_N auf 200 gesetzt
    # Für eine angemessene Laufzeit ist auch empfehlenswert, ENABLE_NON_INTEGERS auf False zu setzen oder die
    # count_limits zu reduzieren.
    """for digit, extended_count_limit, simple_count_limit in (
            (1, 11, 11), (2, 9, 9), (3, 7, 7), (4, 6, 6), (5, 7, 8), (6, 7, 8), (7, 7, 8), (8, 7, 8), (9, 7, 8),
    ):
        print(str2asciiart("digit  =  " + str(digit)))
        finder = BirthdayNumberFinder(digit, 100000, simple_count_limit)
        finder.search_terms(extended_count_limit, simple_count_limit, 100000)
        with open(f"factorials_in_terms_max100/digit{digit}.txt", "w", encoding="utf-8") as f:
            finder.print_factorial_terms(f)"""
