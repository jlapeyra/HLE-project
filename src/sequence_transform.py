from collections import Counter

def transform(seq1, seq2):
    """
    Retorna una llista d'operacions per transformar seq1 -> seq2
    usant DELETE, INSERT i MOVE. També imprimeix el pas a pas.
    """
    a = list(seq1)
    b = list(seq2)
    ops = []

    def do_delete(i):
        ops.append(f"DELETE({i})")
        a.pop(i)

    def do_insert(i, x):
        ops.append(f"INSERT({i}, {x})")
        a.insert(i, x)

    def do_move(i, j):
        ops.append(f"MOVE({i}, {j})")
        x = a.pop(i)
        a.insert(j, x)

    # 1) Elimina sobrants segons comptatges (multiset)
    ca, cb = Counter(a), Counter(b)
    extra = ca - cb  # elements que sobren a "a"
    if extra:
        for i in range(len(a) - 1, -1, -1):
            x = a[i]
            if extra.get(x, 0) > 0:
                do_delete(i)
                extra[x] -= 1

    # 2) Alinea de esquerra a dreta amb MOVE/INSERT
    i = 0
    while i < len(b):
        want = b[i]
        if i < len(a) and a[i] == want:
            i += 1
            continue

        # busca "want" més endavant a 'a'
        j = None
        for k in range(i + 1, len(a)):
            if a[k] == want:
                j = k
                break

        if j is not None:
            do_move(j, i)     # porta'l a la posició correcta
        else:
            do_insert(i, want) # no hi és -> inserta'l
        i += 1

    # 3) Si encara sobra alguna cosa al final, esborra-la
    while len(a) > len(b):
        do_delete(len(a) - 1)

    return ops

if __name__ == "__main__":
    while True:
        input_seq = input("Introdueix la primera seqüència (separada per espais): ")
        s1 = input_seq.strip().split()
        input_seq = input("Introdueix la segona seqüència (separada per espais): ")
        s2 = input_seq.strip().split()
        ops = transform(s1, s2)

        print("Operacions:")
        for op in ops:
            print(" -", op)

        # Opcional: comprova resultat
        print("\nResultat final:", s2)
