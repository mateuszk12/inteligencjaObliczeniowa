import time
import matplotlib.pyplot as plt
import random

from aco import AntColony

plt.style.use("dark_background")

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (26, 22),
    (40, 80),
    (81, 14),
)


def random_coord():
    r = random.randint(0, len(COORDS))
    return r


def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])


def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)

    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))


start = time.time()
plot_nodes()

colony = AntColony(COORDS, ant_count=300, alpha=1, beta=2,
                   pheromone_evaporation_rate=0.20, pheromone_constant=1000.0,
                   iterations=200)

optimal_nodes = colony.get_path()

end = time.time()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

print('czas: ', end - start)
plt.show()

# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:5.061120271682739
# wynik:296.6056997227529
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.12, pheromone_constant=1000.0,iterations=100)
# czas:5.357610702514648
# wynik:312.5045798730912
# colony = AntColony(COORDS, ant_count=300, alpha=0.8, beta=1.2,pheromone_evaporation_rate=0.12, pheromone_constant=1000.0,iteration2s=100)
# czas:5.1018922328948975
# wynik:303.4918895995388
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=200)
# czas:9.89323091506958
# wynik:325.31303283303765
# colony = AntColony(COORDS, ant_count=100, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:3.4740262031555176
# wynik:338.93936626845255
# colony = AntColony(COORDS, ant_count=300, alpha=0.8, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:9.931202173233032
# wynik:296.6056997227529
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:10.174883365631104
# wynik:325.31303283303765
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:10.34669828414917
# wynik:296.6056997227529
# colony = AntColony(COORDS, ant_count=300, alpha=0.5, beta=1.2,pheromone_evaporation_rate=0.10, pheromone_constant=1000.0,iterations=100)
# czas:10.4085373878479
# wynik:303.4918895995388
#najlepszy czas wykonania to: 3.4740262031555176 (osiągnięto dla 100 mrówek)
#najdłuzszy czas działania to: 10.4085373878479
#najlepszy wynik to: 296.6056997227529
#najgorszy wynik to: 338.93936626845255