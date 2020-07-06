"""
Aufgabe 2: Geburtstag

Dieses Skript ist Teil der Einsendung für die

    2. Runde des
    38. Bundeswettbewerbs Informatik

von

    Florian Rädiker.

Teilnahme-ID: 52570
"""
MAX_NUMBER = 3000
for digit in range(1, 10):
    str_digit = str(digit)
    print("digit", digit)
    with open(f"extended_terms/digit{digit}.txt", "r") as f:
        numbers = {int((split_line := line.split())[0]): (int(split_line[2]), split_line[1]) for line in f.readlines()}

    # Berechne Zifferanzahlen für Exponenten
    exponent_counts = {}
    for a in range(1, 11):
        for b in range(2, 11):
            if b == 1:
                continue
            a_ = a
            b_ = b
            count = numbers[a_][0] + numbers[b_][0]
            best = (a_, b_)
            while True:
                # Bruch (a_/b_) erweitern
                a_ += a
                b_ += b
                if a_ > 3000 or b_ > 3000:
                    break
                try:
                    new_count = numbers[a_][0] + numbers[b_][0]
                except KeyError:
                    # a_ oder b_ sind nicht in numbers enthalten, weil offenbar ihre Ziffernanzahl zu groß ist
                    continue
                if new_count < count:
                    count = new_count
                    best = (a_, b_)
            exponent_counts[(a, b)] = (count, numbers[best[0]][1] + "/" + numbers[best[1]][1])
    print("Anzahl der von Exponenten benötigten Ziffern:", exponent_counts)

    for number, (count, term) in numbers.items():
        # Versuche Werte für den Exponenten der Form a/b (wobei x^(a/b) = number, also berechnet sich x aus
        # (a-te Wurzel aus (number^b)). x sollte ganzzahlig sein)
        for a in range(1, 11):
            for b in range(2, 11):
                if a == b:
                    continue
                exponent_count, exponent_term = exponent_counts[(a, b)]
                power = number ** b
                x = int(round(power ** (1 / a)))  # power**(1/a) könnte nicht-ganzzahlig sein
                if x ** a == power:  # testen, ob der nicht-gerundete Wert für x ganzzahlig ist
                    if x in numbers:
                        if numbers[x][0] + exponent_count < count:
                            print("GEFUNDEN: Zahl", number, "wird dargestellt durch", term,
                                  f"(Ziffernanzahl {count}), könnte aber besser durch",
                                  f"{numbers[x][1]}^({exponent_term}) dargestellt werden")
