import re


def getData(fileName):
    f = open(fileName, "r")
    content = f.read()
    whole = re.findall(r"(\d+) (-*\d+.*d*) (-*\d+.*d*) (\d+.*d*) (\d+.*d*) (\d+)",
                       content, re.MULTILINE)
    coords = []
    x_data = []
    y_data = []
    visit_time = []
    benefit = []
    open_time = []
    close_time = []
    for i in whole:
        coords.append(i[1].split(' '))
        open_time.append(int(i[-2]))
        close_time.append(int(i[-1]))
    for i in coords:
        x_data.append(float(i[0]))
        y_data.append(float(i[1]))
        visit_time.append(float(i[2]))
        benefit.append(float(i[3]))

    XY = zip(x_data, y_data)
    coords = list(XY)

    return coords, benefit, open_time, close_time, visit_time
