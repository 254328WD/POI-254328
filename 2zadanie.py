import csv
import numpy as np
from sklearn.cluster import KMeans


# Funkcja wczytująca punkty z pliku .xyz
def load_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                points.append(list(map(float, line.split())))
    return np.array(points)


# Funkcja do znajdowania rozłącznych chmur punktów za pomocą algorytmu k-średnich
def find_clusters(points, k=3):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    clusters = [points[kmeans.labels_ == i] for i in range(k)]
    return clusters


# Funkcja RANSAC do dopasowania płaszczyzny
def fit_plane_ransac(points, num_iterations, distance_threshold):
    best_inliers_count = 0
    best_plane = None
    best_distance = np.inf

    for _ in range(num_iterations):
        # Losowe próbkowanie 3 punktów
        sample_points = points[np.random.choice(len(points), 3, replace=False), :]
        # Obliczanie równania płaszczyzny Ax+By+Cz+D=0
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)
        d = -np.dot(normal, sample_points[0])
        # Szukanie inlierów
        distances = np.abs(np.dot(points, normal) + d) / np.linalg.norm(normal)
        inliers = distances < distance_threshold
        inliers_count = np.sum(inliers)

        # Aktualizacja najlepszego modelu
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_plane = normal, d
            best_distance = np.mean(distances[inliers])

    # Normalizacja wektora normalnego
    normal_vector = best_plane[0]
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector_normalized, best_distance


# Ścieżki do plików .xyz
file_paths = [
    'horizontal_surface.xyz',
    'vertical_surface.xyz',
    'cylindrical_surface.xyz'
]

# Parametr określający, kiedy uważamy chmurę za płaszczyznę
threshold_distance = 0.05 

# Wczytywanie i dopasowanie płaszczyzny dla każdej chmury
for file_path in file_paths:
    points = load_xyz_file(file_path)
    clusters = find_clusters(points)

    for i, cluster in enumerate(clusters):
        normal_vector, mean_distance = fit_plane_ransac(cluster, num_iterations=100, distance_threshold=0.01)
        is_plane = mean_distance < threshold_distance
        # Określenie czy płaszczyzna jest pionowa czy pozioma
        if np.isclose(normal_vector[2], 0, atol=0.1):
            plane_type = 'pionowa'
        else:
            plane_type = 'pozioma'

        print(f"Chmura {i + 1} z pliku {file_path}:")
        print(f" Wektor normalny: {np.array_str(normal_vector, precision=4)}")
        print(f" Średnia odległość od płaszczyzny: {mean_distance}")
        print(f" Czy jest płaszczyzną: {'tak' if is_plane else 'nie'}")
        if is_plane:
            print(f" Typ płaszczyzny: {plane_type}")
        print()

