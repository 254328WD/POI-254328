import numpy as np

# Funkcje generujące chmury punktów

def generate_horizontal_surface(num_points, width, length):
    x = np.random.uniform(-width/2, width/2, num_points)
    y = np.random.uniform(-length/2, length/2, num_points)
    z = np.zeros(num_points)  # Z-axis points are 0 on a horizontal surface
    return np.column_stack((x, y, z))

def generate_vertical_surface(num_points, width, height):
    x = np.random.uniform(-width/2, width/2, num_points)
    y = np.zeros(num_points)  # Y-axis points are 0 on a vertical surface
    z = np.random.uniform(0, height, num_points)
    return np.column_stack((x, y, z))

def generate_cylindrical_surface(num_points, radius, height):
    angle = np.random.uniform(0, 2*np.pi, num_points)  # Random angle for cylindrical coordinates
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = np.random.uniform(0, height, num_points)  # Uniform height distribution
    return np.column_stack((x, y, z))

# Funkcja zapisująca chmurę punktów do pliku .xyz
def save_to_xyz_file(points, file_path):
    with open(file_path, 'w') as file:
        for point in points:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")

# Przykładowe użycie funkcji i zapis do plików
num_points = 100000  # Liczba punktów w chmurze

# Generowanie chmury punktów
horizontal_surface_points = generate_horizontal_surface(num_points, width=10, length=20)
vertical_surface_points = generate_vertical_surface(num_points, width=10, height=20)
cylindrical_surface_points = generate_cylindrical_surface(num_points, radius=5, height=20)

# Ścieżki do plików .xyz
horizontal_surface_file_path = 'horizontal_surface.xyz'
vertical_surface_file_path = 'vertical_surface.xyz'
cylindrical_surface_file_path = 'cylindrical_surface.xyz'

# Zapis do plików
save_to_xyz_file(horizontal_surface_points, horizontal_surface_file_path)
save_to_xyz_file(vertical_surface_points, vertical_surface_file_path)
save_to_xyz_file(cylindrical_surface_points, cylindrical_surface_file_path)

print("Chmury punktów zostały zapisane do plików .xyz")
