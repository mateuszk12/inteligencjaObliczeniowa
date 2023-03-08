import pandas as pa
import matplotlib.pyplot as plt
miasta = pa.read_csv("miasta.csv")
print(miasta.values)
miasta.loc[len(miasta.index)] = [2010,460,555,405]
plt.ylabel("Liczba ludnosci w tys.")
plt.xlabel("Lata")
plt.title("Liczba ludnosci Gdanska")
plt.plot(miasta["Rok"],miasta["Gdansk"],color="red",marker="o")
plt.show()


