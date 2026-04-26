import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. PARAMETRY LASERA (Rzeczywista wiązka)
# ==========================================
wavelength = 1e-6          # Długość fali (1 um)
k = 2 * np.pi / wavelength # Liczba falowa
w0 = 0.05                  # Początkowy promień (5 cm - duży teleskop nadawczy)
F = 5000.0                 # Ogniskowa w metrach (skupiamy wiązkę na dystans 5 km)

# Siatka poprzeczna
N = 256
L = 0.4                    # Okno 40x40 cm
x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)
R_sq = X**2 + Y**2

dx = x[1] - x[0]
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, kx)
K_perp_sq = KX**2 + KY**2

# --- RZECZYWISTY GAUSS ---
# Amplituda zanikająca (Gaussian) pomnożona przez fazę sferyczną (skupienie soczewką)
E = np.exp(-R_sq / w0**2) * np.exp(-1j * k * R_sq / (2 * F))

I_initial = np.abs(E)**2
# ==========================================
# 2. PARAMETRY TRASY (Wykładnicza atmosfera)
# ==========================================
z_total = 10000.0      # Zasięg 10 km (Nadajnik na pokładzie samolotu)
z_current = 0.0        # Zaczynamy na górze

z0_atm = 7350.0        # Skala grubości turbulencji optycznych (1.5 km)
dz_start = 2000.0      # Duże skoki w stratosferze (co 2 km)     # Początkowy duży skok (rzadka atmosfera na górze)
d_screen = 0.1         # Fizyczna grubość pojedynczego zaburzenia (np. 1 m)

side_view = []         # Tablica na widok 3D
z_positions = []       # Zapisujemy rzeczywiste odległości, by narysować wykres

print("Rozpoczynam propagację w dół przez gęstniejącą atmosferę...")

# ==========================================
# 3. PĘTLA PROPAGACJI
# ==========================================
while z_current < z_total:
    
    # Obliczamy dynamiczny krok - maleje eksponencjalnie z przebytym dystansem
    # Skoro startujemy z góry, z_current rośnie w dół, więc d_prop maleje.
    d_prop = dz_start * np.exp(-z_current / z0_atm)
    
    # Zabezpieczenie: ostatni krok dobija równo do ziemi
    if z_current + d_prop > z_total:
        d_prop = z_total - z_current
        
    # ETAP 1: Propagacja w próżni (pusta przestrzeń między ekranami)
    E_fft = np.fft.fft2(E)
    E = np.fft.ifft2(E_fft * np.exp(-1j * K_perp_sq * d_prop / (2 * k)))
    
    z_current += d_prop
    
    # ETAP 2: Ekran fazowy o grubości d_screen (zaburzenie turbulencyjne)
    # Przy ziemi turbulencje są silniejsze (zwiększamy amplitudę bąbli)
    turb_strength = 5e-7 * np.exp(z_current / z0_atm) # Rośnie bliżej ziemi!
    
    n1 = gaussian_filter(np.random.normal(0, 1, (N, N)), sigma=4)
    n1 = turb_strength * (n1 / np.max(np.abs(n1)))
    
    phase_screen = np.exp(1j * k * d_prop * n1)
    E = E * phase_screen
    
    # Zapis danych do wizualizacji
    side_view.append(np.abs(E[N//2, :])**2)
    z_positions.append(z_current)
    
    print(f"Przeleciano: {z_current:.1f} m | Krok dz: {d_prop:.1f} m")

print("Dotarto do celu.")

# ==========================================
# 4. WIZUALIZACJA 3D (Siatka nieliniowa)
# ==========================================
# Konwertujemy dane, Z_grid i X_grid posłużą do poprawnego rysowania
Z_grid, X_grid = np.meshgrid(z_positions, x)
side_view = np.array(side_view).T

I_final = np.abs(E)**2

plt.figure(figsize=(14, 6))

plt.subplot(1, 1, 1)
plt.title("Wiazka poczatkowa (z = 0)")
plt.imshow(I_initial, extent=[-L/2, L/2, -L/2, L/2], cmap='inferno')
plt.colorbar(label='Natezenie')
plt.xlabel('x [m]')
plt.ylabel('y [m]')

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.title("Przekrój boczny (Gęstniejące ekrany)")
# Używamy pcolormesh, bo osie nie mają równego rozstawu (d_prop się zmieniał!)
plt.pcolormesh(Z_grid, X_grid * 100, side_view, shading='auto', cmap='hot')
plt.xlabel("Przebyty dystans Z [m]")
plt.ylabel("Przekrój X [cm]")
plt.colorbar(label="Natężenie")

plt.subplot(1, 2, 2)
plt.title(f"Plamka końcowa na Ziemi (Z = {z_total} m)")
plt.imshow(np.abs(E)**2, extent=[-L*100/2, L*100/2, -L*100/2, L*100/2], cmap='hot')
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")
plt.colorbar(label="Natężenie")

plt.tight_layout()
plt.show()