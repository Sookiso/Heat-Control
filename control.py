import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Gompertz-hatásfok modellezése
def gompertz_efficiency(u, eta_max=1.0, beta=5):
    """A fűtési rendszer hatásfoka Gompertz-függvény alapján."""
    return eta_max * np.exp(-np.exp(beta * (u - 0.5)))

# Hőmérséklet dinamikája
def room_temperature_dynamics(x, u, x_out, dt=1, C=100, k_loss=0.05, eta_max=1.0, beta=5):
    """
    A szoba hőmérsékletének változása.
    x: aktuális hőmérséklet
    u: fűtési vezérlés (0-1 között)
    x_out: külső hőmérséklet
    dt: időlépés
    C: hőkapacitás
    k_loss: hőveszteségi együttható
    """
    Q_heat = gompertz_efficiency(u, eta_max, beta)  # Fűtésből származó hatásos hő
    Q_loss = k_loss * (x - x_out)  # Hőveszteség
    dx_dt = (Q_heat - Q_loss) / C
    return x + dx_dt * dt

# Büntetési függvény az MPC optimalizáláshoz
def mpc_objective(u, x, x_ref, x_out, N, dt=1):
    """
    MPC költségfüggvény. Minimalizálja a hőmérsékleti eltérést és a fűtési intenzitást.
    u: vezérlési jel (vektor)
    x: kezdeti hőmérséklet
    x_ref: célhőmérséklet
    x_out: külső hőmérséklet
    N: predikciós horizont
    """
    cost = 0
    for k in range(N):
        # Szimuláljuk a szoba hőmérsékletét
        x = room_temperature_dynamics(x, u[k], x_out, dt)
        # Költségek: eltérés a célhőmérséklettől (negyedik hatvány) + vezérlés büntetése
        cost += ((x_ref - x)**4) + (u[k]**2)
    return cost

# Szimulációs paraméterek
x_initial = 19.0          # Kezdeti szobahőmérséklet [Celsius]
x_ref = 21.0              # Célhőmérséklet [Celsius]
x_out = 15.0              # Külső hőmérséklet [Celsius]
N = 30                    # Predikciós horizont
dt = 1                    # Időlépés [óra]
bounds = [(0, 1)] * N     # Vezérlési jel korlátai
time_horizon = np.arange(1, N + 1)

# Optimalizáció
u_initial_guess = np.zeros(N)  # Kezdeti tipp a vezérlési jelre
result = minimize(mpc_objective, u_initial_guess, args=(x_initial, x_ref, x_out, N, dt),
                  bounds=bounds, method='SLSQP')

# Ellenőrizni az optimalizáció eredményét
if result.success:
    u_optimal = result.x
else:
    raise ValueError("Optimalizáció sikertelen!")

# Szimuláció a vezérlési jellel
x_simulated = [x_initial]
for u in u_optimal:
    x_new = room_temperature_dynamics(x_simulated[-1], u, x_out, dt)
    x_simulated.append(x_new)

# Ábrázolás: 3D diagram
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Adatok előkészítése a 3D diagramhoz
time_steps = np.arange(1, len(u_optimal) + 1)
u_values, t_values = np.meshgrid(u_optimal, time_steps)
z_values = np.array(x_simulated[1:])  # Felmelegedési hőmérséklet

# 3D felület rajzolása
ax.plot_surface(u_values, t_values, z_values, cmap='viridis', alpha=0.8)
ax.set_xlabel("Vezérlési jel (u)")
ax.set_ylabel("Idő (óra)")
ax.set_zlabel("Hőmérséklet (Celsius)")
ax.set_title("MPC alapú szoba hőmérséklet-szabályozás")
plt.show()
