import numpy as np
from sklearn.cluster import DBSCAN
import pyransac3d as pyrsc


# Funkcja wczytująca punkty z pliku .xyz
def load_xyz_file(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip():
                points.append(list(map(float, line.split())))
    return np.array(points)


# Funkcja do znajdowania klastrów za pomocą algorytmu DBSCAN
def find_clusters_dbscan(points, eps=0.1, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    unique_labels = np.unique(labels)
    clusters = [points[labels == label] for label in unique_labels if
                label != -1]  # Pomijamy punkty nieprzypisane do żadnego klastra (-1)
    return clusters[:10]  # Zwracamy tylko 10 pierwszych klastrów


# Funkcja RANSAC do dopasowania płaszczyzny przy użyciu pyransac3d
def fit_plane_ransac_py(points, distance_threshold=0.01):
    plane = pyrsc.Plane()
    best_eq, best_inliers = plane.fit(points, distance_threshold)
    return best_eq, points[best_inliers]


# Ścieżki do plików .xyz
file_paths = [
    'horizontal_surface.xyz',
    'vertical_surface.xyz',
    'cylindrical_surface.xyz'
]

# Parametry dla algorytmu DBSCAN
eps = 0.1
min_samples = 10

# Parametr określający, kiedy uważamy chmurę za płaszczyznę
threshold_distance = 0.05  # Możesz dostosować ten próg

# Wczytywanie i dopasowanie płaszczyzny dla każdej chmury
for file_path in file_paths:
    print(f"Analiza pliku {file_path}:")
    points = load_xyz_file(file_path)

    # Znajdowanie klastrów za pomocą DBSCAN
    clusters = find_clusters_dbscan(points, eps, min_samples)

    for i, cluster in enumerate(clusters):
        # Dopasowanie płaszczyzny za pomocą pyransac3d
        best_eq, best_inliers = fit_plane_ransac_py(cluster)
        mean_distance = np.mean(np.abs(np.dot(best_inliers, best_eq[:3]) + best_eq[3]) / np.linalg.norm(best_eq[:3]))
        is_plane = mean_distance < threshold_distance

        # Określenie typu płaszczyzny
        if np.isclose(best_eq[2], 0, atol=0.1):
            plane_type = 'pionowa'
        else:
            plane_type = 'pozioma'

        print(f"  Chmura {i + 1}:")
        print(f"   Wektor normalny: {np.array_str(np.array(best_eq[:3]), precision=4)}")
        print(f"   Średnia odległość od płaszczyzny: {mean_distance}")
        print(f"   Czy jest płaszczyzną: {'tak' if is_plane else 'nie'}")
        if is_plane:
            print(f"   Typ płaszczyzny: {plane_type}")
        print()
