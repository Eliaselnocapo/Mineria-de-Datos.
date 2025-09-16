from itertools import combinations

# Aqui calculamos el soporte
def get_support(transactions, itemset):
    count = sum(1 for t in transactions if itemset.issubset(t))
    return count / len(transactions)

# Aqui el algoritmo Apriori:
def apriori(transactions, min_support):

    # Generamos los ítems únicos
    items = set(item for t in transactions for item in t)
    itemsets = [{i} for i in items]

    freq_itemsets = []
    k = 1

    while itemsets:
        print(f"\n--- Iteración {k} ---")
        valid_itemsets = []
        for itemset in itemsets:
            support = get_support(transactions, itemset)
            if support >= min_support:
                valid_itemsets.append(itemset)
                print(f"{itemset} → Soporte: {support:.2f}")
        freq_itemsets.extend(valid_itemsets)

        # Generamos las combinaciones de k+1 ítems
        itemsets = [a.union(b) for i, a in enumerate(valid_itemsets) 
                    for b in valid_itemsets[i+1:] if len(a.union(b)) == k+1]
        
        # Eliminamos si hay duplicados
        itemsets = list(map(set, set(frozenset(i) for i in itemsets)))
        k += 1

    return freq_itemsets

# Ejemplos de transacciones
transactions = [
    {"procesador", "tarjeta_madre", "memoria_RAM"},
    {"procesador", "tarjeta_madre"},
    {"tarjeta_grafica", "memoria_RAM"},
    {"disco_duro", "memoria_RAM"},
    {"procesador", "tarjeta_madre", "disco_duro"},
]

# Ejecutamos aqui el Apriori con soporte mínimo del 0.5 (50%)
frequent_itemsets = apriori(transactions, min_support=0.5)

print("\n✅ Items frecuentes:")
for fi in frequent_itemsets:
    print(fi)
