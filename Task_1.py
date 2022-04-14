from scipy.stats import norm
from csv import writer


num_points = 5000

def generate_coordinates_vert(points: int = num_points):
    x_coord = norm(loc=-100, scale=0.1)
    y_coord = norm(loc=-150, scale=70)
    z_coord = norm(loc=10, scale=70)

    x = x_coord.rvs(size=points)
    y = y_coord.rvs(size=points)
    z = z_coord.rvs(size=points)

    points_vert = zip(x, y, z)
    return points_vert


def generate_coordinates_horiz(points: int = num_points):
    x_coord = norm(loc=50, scale=70)
    y_coord = norm(loc=300, scale=0.1)
    z_coord = norm(loc=150, scale=70)

    x = x_coord.rvs(size=points)
    y = y_coord.rvs(size=points)
    z = z_coord.rvs(size=points)

    points_horiz = zip(x, y, z)
    return points_horiz


def generate_coordinates_cylind(points: int = 2*num_points):
    x_coord = norm(loc=140, scale=20)
    y_coord = norm(loc=-150, scale=70)
    z_coord = norm(loc=50, scale=20)

    x = x_coord.rvs(size=points)
    y = y_coord.rvs(size=points)
    z = z_coord.rvs(size=points)

    points_cylind = zip(x, y, z)
    return points_cylind


if __name__ == '__main__':
    cloud_vert = generate_coordinates_vert(num_points)
    with open('LidarData_vert.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_vert:
            csvwriter.writerow(p)

    cloud_horiz = generate_coordinates_horiz(num_points)
    with open('LidarData_horiz.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_horiz:
            csvwriter.writerow(p)

    cloud_cylind = generate_coordinates_cylind(2*num_points)
    with open('LidarData_cylind.xyz', 'w', encoding='utf8', newline='') as csvfile:
        csvwriter = writer(csvfile)
        for p in cloud_cylind:
            csvwriter.writerow(p)


    filenames = ['LidarData_vert.xyz', 'LidarData_horiz.xyz', 'LidarData_cylind.xyz']
    with open('LidarData_concat.xyz', 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)