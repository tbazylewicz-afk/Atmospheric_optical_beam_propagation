import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GENERATOR EKRANÓW KOŁMOGOROWA
# ==========================================
def generate_kolmogorov_screen(N, L, r0):
    """
    Generuje fizycznie poprawny ekran fazowy zgodny ze spektrum Kolmogorowa.
    N  - rozdzielczosc siatki (np. 256)
    L  - fizyczny rozmiar siatki w metrach
    r0 - parametr Frieda (sila turbulencji) w metrach
    """
    # Krok w przestrzeni czestotliwosci
    delta_k = 2 * np.pi / L
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, kx)
    K_sq = KX**2 + KY**2
    
    # Zapobieganie dzieleniu przez zero dla stalej skladowej
    K_sq[0, 0] = 1e-10
    
    # Widmo gęstości mocy (PSD) dla fazy Kolmogorowa
    PSD_phi = 0.023 * r0**(-5/3) * K_sq**(-11/6)
    PSD_phi[0, 0] = 0.0
    
    # Generowanie zespolonego, bialego szumu
    noise_real = np.random.normal(0, 1, (N, N))
    noise_imag = np.random.normal(0, 1, (N, N))
    complex_noise = (noise_real + 1j * noise_imag) / np.sqrt(2)
    
    # Filtrowanie szumu przez widmo Kolmogorowa
    screen_fft = complex_noise * np.sqrt(PSD_phi) * delta_k
    
    # Odwrotna transformata (mnozymy przez N^2 ze wzgledu na konwencje FFT w numpy)
    phase_screen = np.real(np.fft.ifft2(screen_fft)) * (N**2)
    
    return phase_screen

# ==========================================
# 2. PARAMETRY LASERA I SIATKI
# ==========================================
wavelength = 1e-6          # 1 mikrometr
k = 2 * np.pi / wavelength # Liczba falowa
w0 = 0.05                  # Poczatkowy promien 5 cm (nadajnik)
F = 10000.0                # Ogniskowa na 10 km (celujemy w Ziemie)

N = 256                    # Siatka 256x256 pikseli
L = 0.5                    # Rozmiar okna obliczeniowego 50x50 cm
x = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, x)
R_sq = X**2 + Y**2

# Przestrzen Fouriera dla propagacji w prozni (dyfrakcja)
dx = L / N
kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
KX, KY = np.meshgrid(kx, kx)
K_perp_sq = KX**2 + KY**2

# RZECZYWISTA WIAZKA POCZATKOWA (Gauss + soczewka)
E = np.exp(-R_sq / w0**2) * np.exp(-1j * k * R_sq / (2 * F))

# ==========================================
# 3. PARAMETRY TRASY ATMOSFERYCZNEJ
# ==========================================
z_total = 10000.0      # Startujemy z 10 km, lecimy na wysokosc 0
z_current = 0.0        # Przelatany dystans

# Profil turbulencji (model uproszczony HV)
Cn2_ground = 1e-14     # Sila turbulencji przy Ziemi (silna)
h0 = 1500.0            # Skala spadku grubosci turbulencji (1.5 km)

side_view = []
z_positions = []

print("Rozpoczynam symulacje propagacji (Downlink: 10 km -> Ziemia)...")

# ==========================================
# 4. GLOWNA PETLA (Split-Step Fourier Method)
# ==========================================
while z_current < z_total:
    
    # KROK 1: Obliczanie dynamicznego skoku dz
    # Na poczatku przeskakujemy szybko (np. co 2000m), przy Ziemi zwalniamy do 20m
    dz = 2000.0 * np.exp(-z_current / 2000.0)
    if dz < 20.0: 
        dz = 20.0
    if z_current + dz > z_total:
        dz = z_total - z_current
        
    # KROK 2: Propagacja w pustej przestrzeni (Dyfrakcja)
    E_fft = np.fft.fft2(E)
    E = np.fft.ifft2(E_fft * np.exp(-1j * K_perp_sq * dz / (2 * k)))
    
    z_current += dz
    
    # KROK 3: Wyznaczanie sily turbulencji (r0) dla tego kroku
    # Obliczamy aktualna wysokosc nad Ziemia
    altitude = z_total - z_current
    
    # Cn2 rosnie wykladniczo im blizej Ziemi
    Cn2 = Cn2_ground * np.exp(-altitude / h0)
    
    # Wzor na parametr Frieda r0 dla fali plaskiej na dystansie dz
    # Im wieksze dz i wieksze Cn2, tym mniejsze r0 (silniejsze zaburzenia)
    r0 = (0.423 * (k**2) * Cn2 * dz)**(-3/5)
    
    # KROK 4: Generacja i nalozenie ekranu fazowego Kolmogorowa
    phase = generate_kolmogorov_screen(N, L, r0)
    E = E * np.exp(1j * phase)
    
    # Zapis danych
    side_view.append(np.abs(E[N//2, :])**2)
    z_positions.append(z_current)
    
    print(f"Przebyto: {z_current:5.0f} m | Wysokosc: {altitude:5.0f} m | Krok dz: {dz:4.0f} m | r0: {r0:5.2f} m")

print("Zakonczono propagacje!")

# ==========================================
# 5. WIZUALIZACJA
# ==========================================
Z_grid, X_grid = np.meshgrid(z_positions, x)
side_view = np.array(side_view).T

plt.figure(figsize=(14, 6))

# Widok boczny
plt.subplot(1, 2, 1)
plt.title("Przekroj boczny wiazki")
plt.pcolormesh(Z_grid, X_grid * 100, side_view, shading='auto', cmap='inferno')
plt.xlabel("Przebyty dystans Z [m] (Z Ziemi = 10000)")
plt.ylabel("Os X [cm]")
plt.colorbar(label="Natezenie")

# Widok 2D koncowej plamki
plt.subplot(1, 2, 2)
plt.title(f"Plamka na Ziemi (Wysokosc = 0 m)")
plt.imshow(np.abs(E)**2, extent=[-L*100/2, L*100/2, -L*100/2, L*100/2], cmap='inferno')
plt.xlabel("X [cm]")
plt.ylabel("Y [cm]")
plt.colorbar(label="Natezenie")

plt.tight_layout()
plt.show()