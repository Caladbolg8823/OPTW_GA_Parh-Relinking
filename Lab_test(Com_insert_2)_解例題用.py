import re
from typing import Text
import numpy as np
import Parse_OPTW
import matplotlib.pyplot as plt
import datetime
from tqdm import trange


S = datetime.datetime.now()
fileName = 'OPTW/instance/c_r_rc_100_100/r105.txt'

BKS = 247

coords, benefit, open_time, close_time, visit_time = Parse_OPTW.getData(
    fileName)


# TODO 距離矩陣
def compute_dis_mat(coords):
    dis_array = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i == j:
                dis_array[i][j] = np.inf
                continue
            a = np.array(coords[i])
            b = np.array(coords[j])
            tmp = np.linalg.norm(a-b)
            dis_array[i][j] = round(tmp, 2)
    return dis_array


# TODO 初始路徑(貪婪法)
def init_greedy(dis_array, benefit):
    path = []
    use_time = 0
    node = [x for x in range(0, len(coords))]
    current = 0
    node.remove(current)
    result_one = [current]

    while use_time <= max_time:  # ? 尋找每步最大獲益、最小距離的點
        point_max = -1
        min_time = 0
        ex = []
        for x in node:
            if open_time[x] <= use_time + dis_array[current][x] <= close_time[x]:
                greedy_point = benefit[x] / dis_array[current][x]
                if greedy_point > point_max:
                    point_max = greedy_point  # ? 當前最大獲益
                    min_time = dis_array[current][x] + \
                        visit_time[x]   # ? 當前最短時間
                    tmp_choose = x
            elif open_time[x] > use_time + dis_array[current][x] and close_time[x] >= open_time[x]:
                wait_time = open_time[x] - (use_time + dis_array[current][x])
                greedy_point = benefit[x] / \
                    (dis_array[current][x] + wait_time)
                if greedy_point > point_max:
                    point_max = greedy_point  # ? 當前最大獲益
                    min_time = open_time[x] + visit_time[x]  # ?當前最短時間
                    tmp_choose = x
            # elif close_time[x] < use_time + dis_array[current][x]:
            else:
                ex.append(x)
                if len(ex) == len(node):
                    break
        if len(ex) == len(node):
            break

        use_time += min_time
        current = tmp_choose
        result_one.append(tmp_choose)
        node.remove(tmp_choose)
    result_one.append(0)
    use_time, result_one = travel_time(dis_array, result_one)
    path.append(result_one)
    return path


# TODO 初始路徑(GRASP)
def init_grasp(dis_array, population_num, benefit, path):
    # path = []
    for i in range(population_num - 1):
        use_time = 0
        node = [x for x in range(0, len(coords))]
        current = 0
        node.remove(current)
        result_one = [current]
        while use_time <= max_time:
            random_ratio = np.random.randint(0, 2)  # ? 0：貪婪、1：隨機
            point_max = -1
            min_time = 0
            ex = []  # 超過關門時間的點集合
            candidate = []    # 候選點，用於GRASP
            for x in node:
                if open_time[x] <= use_time + dis_array[current][x] <= close_time[x] or (open_time[x] > use_time + dis_array[current][x] and close_time[x] >= open_time[x]):
                    candidate.append(x)
                # if close_time[x] < use_time + dis_array[current][x]:
                else:
                    ex.append(x)
                    if len(ex) == len(node):
                        break
            if len(ex) == len(node):
                break

            if random_ratio == 1:  # ? ratio == 1，可行解隨機
                aim = np.random.choice(candidate)
                if open_time[aim] <= use_time + dis_array[current][aim] <= close_time[aim]:
                    min_time = dis_array[current][aim] + visit_time[aim]
                    tmp_choose = aim
                else:
                    wait_time = open_time[aim] - \
                        (use_time + dis_array[current][aim])
                    min_time = open_time[aim] + visit_time[aim]
                    tmp_choose = aim

                use_time += min_time
                current = tmp_choose
                result_one.append(tmp_choose)
                node.remove(tmp_choose)

            else:  # ? ratio == 0，普通貪婪
                for ccc in candidate:
                    if open_time[ccc] <= use_time + dis_array[current][ccc] <= close_time[ccc]:
                        greedy_point = benefit[ccc] / dis_array[current][ccc]
                        if greedy_point > point_max:
                            point_max = greedy_point  # ? 當前最大獲益
                            min_time = dis_array[current][ccc] + \
                                visit_time[ccc]  # ? 當前最短時間
                            tmp_choose = ccc

                    # elif open_time[ccc] > use_time + dis_array[current][ccc] and close_time[ccc] >= open_time[ccc] + visit_time[ccc]:
                    else:
                        wait_time = open_time[ccc] - \
                            (use_time + dis_array[current][ccc])
                        greedy_point = benefit[ccc] / \
                            (dis_array[current][ccc] +
                             wait_time)
                        if greedy_point > point_max:
                            point_max = greedy_point  # ? 當前最大獲益
                            min_time = open_time[ccc] + \
                                visit_time[ccc]  # ? 當前最短時間
                            tmp_choose = ccc

                use_time += min_time
                current = tmp_choose
                result_one.append(tmp_choose)
                node.remove(tmp_choose)

        result_one.append(0)
        use_time, result_one = travel_time(dis_array, result_one)

        path.append(result_one)
    return path


#! 選擇
def choice(path):
    n = len(path)
    fit_pros = []
    [fit_pros.append(get_fitness(benefit, path[i])) for i in range(n)]
    choice_gens = []
    for i in range(n):
        j = roulette_wheel(fit_pros, path)
        choice_gens.append(path[j])
    path = choice_gens
    return path


# TODO 計算每條路徑的分數
def Evaluation(benefit, gens):
    gens = np.copy(gens)
    P = 0
    for i in gens:
        P += benefit[i]
    return P


#! 計算適應值
def get_fitness(benefit, gens):
    TT, G = travel_time(dis_array, gens)
    P = 0
    for i in G:
        P += benefit[i]
    fitness = (P**3)/TT
    return P


# TODO 輪盤法
def roulette_wheel(fitness, path):
    total_fitness = sum(fitness)
    pick = np.random.random()
    p = 0
    for j in range(len(path)):
        p += fitness[j] / total_fitness
        # print(p)
        if p >= pick:
            return j


# TODO 計算旅遊總時間(不合理時間剔除) 歸還時間與路線
def travel_time(dis_array, route):
    for i in route:
        if dummy in route:
            route.remove(dummy)

    a = 1
    use_time = 0
    while a < len(route):
        now = route[a-1]
        next = route[a]
        if open_time[next] <= use_time + dis_array[now][next] <= close_time[next]:
            use_time += dis_array[now][next] + visit_time[next]
        elif open_time[next] > use_time + dis_array[now][next] and close_time[next] >= open_time[next]:
            use_time = open_time[next] + visit_time[next]
        # elif open_time[next] <= use_time + dis_array[now][next] <= close_time[next] and close_time[next] < visit_time[next]+use_time + dis_array[now][next]:
        #     use_time = close_time[next]
        else:
            if next == route[0]:
                use_time -= dis_array[route[a-2]][now] + visit_time[now]
                route.remove(now)
                a -= 2
            else:
                route.remove(next)
                a -= 1
        a += 1
    # print('加入點後的時間：', use_time)
    return use_time, route


# TODO 路徑重新連結
def path_relinking(dis_array, route):
    utmost = 0
    count = 0
    R = np.random.randint(1, 3)

    use_time = 0
    while use_time < close_time[0]:

        PR_list = [i for i in range(len(coords))]
        PR_list.append(0)
        for i in route:
            PR_list.remove(i)

        best_candidate = -9
        a = 1
        now = route[a-1]
        next = route[a]
        use_time = 0
        best_ratio = 0
        while a < len(route):  # ? 在整條路徑中，找出最好候選點
            candidate_point = []
            w_candidate_point = []
            for i in PR_list:
                arrival_time = dis_array[now][i] + use_time
                if open_time[i] <= arrival_time <= close_time[i]:  # ! 找出不影響後面入場時間的點
                    if open_time[next] <= arrival_time + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                        candidate_point.append(i)

                elif open_time[i] > arrival_time and open_time[i] < close_time[i]:
                    if open_time[next] <= open_time[i] + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                        w_candidate_point.append(i)
            if len(candidate_point) != 0:  # !可以剛好進去玩的
                candidate_point, candidate_ratio, current, go = cal_ratio(
                    candidate_point, now, next)
                if candidate_ratio > best_ratio:
                    # refer_point = route.index(current, 1, -1)
                    insert_point = route.index(go, 1)
                    best_ratio = candidate_ratio
                    best_candidate = candidate_point
            if len(w_candidate_point) != 0:  # ! 要等開門的
                candidate_point, candidate_ratio, current, go = w_cal_ratio(
                    w_candidate_point, now, next, use_time)
                if candidate_ratio > best_ratio:
                    # refer_point = route.index(current, 1, -1)
                    insert_point = route.index(go, 1)
                    best_ratio = candidate_ratio
                    best_candidate = candidate_point

            use_time, route = cal_time(dis_array, now, next, use_time, route)
            a += 1
            if a < len(route):
                now = route[a-1]
                next = route[a]
            # else:
            #     use_time = cal_time(dis_array, now, next, use_time)
            #     a += 1
            #     now = route[a-1]
            #     next = route[a]

        if best_candidate != -9:
            ori_route = route
            route.insert(insert_point, best_candidate)
            use_time, route = travel_time(dis_array, route)
            if use_time > utmost:
                utmost = use_time
                best_route = route
                count = 0
            else:
                count += 1
                if count >= R:
                    break
            if use_time > close_time[0]:
                route = ori_route
                break
        else:
            return route
    # ? print(utmost)

    return best_route


# TODO 路徑重新連結(輪盤法)
def NPR(dis_array, route):
    utmost = 0
    count = 0

    use_time = 0
    while use_time < close_time[0]:

        PR_list = [i for i in range(len(coords))]
        PR_list.append(0)
        for i in route:
            PR_list.remove(i)

        best_candidate = -9
        a = 1
        now = route[a-1]
        next = route[a]
        use_time = 0

        insert_point = []
        best_candidate = []
        best_ratio = []

        while a < len(route):  # ? 在整條路徑中，找出最好候選點
            candidate_point = []
            w_candidate_point = []
            for i in PR_list:
                arrival_time = dis_array[now][i] + use_time
                if open_time[i] <= arrival_time <= close_time[i]:  # ! 找出不影響後面入場時間的點
                    if open_time[next] <= arrival_time + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                        candidate_point.append(i)

                elif open_time[i] > arrival_time and open_time[i] < close_time[i] - visit_time[i]:
                    if open_time[next] < open_time[i] + visit_time[i] + dis_array[i][next] < close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                        w_candidate_point.append(i)
            if len(candidate_point) != 0:  # !可以剛好進去玩的
                candidate_point, candidate_ratio, go = ratio1(
                    candidate_point, now, next)

                #!!!!!!!!
                insert_point.append(route.index(go, 1))
                best_ratio.append(candidate_ratio)
                best_candidate.append(candidate_point)

            if len(w_candidate_point) != 0:  # ! 要等開門的
                candidate_point, candidate_ratio, go = ratio_w2(
                    w_candidate_point, now, next, use_time)
                #!!!!
                insert_point.append(route.index(go, 1))
                best_ratio.append(candidate_ratio)
                best_candidate.append(candidate_point)

            use_time, route = cal_time(dis_array, now, next, use_time, route)
            a += 1
            if a < len(route):
                now = route[a-1]
                next = route[a]
            # else:
            #     use_time = cal_time(dis_array, now, next, use_time)
            #     a += 1
            #     now = route[a-1]
            #     next = route[a]

        if len(best_candidate) != 0:
            ori_route = route

            best_candidate, insert_point = wheel(
                best_ratio, best_candidate, insert_point)

            route.insert(insert_point, best_candidate)
            use_time, route = travel_time(dis_array, route)
            if use_time > utmost:
                utmost = use_time
                best_route = route
                count = 0
            else:
                count += 1
                if count >= 1:
                    break
            if use_time > close_time[0]:
                route = ori_route
                break
        else:
            return route
    # ? print(utmost)

    return best_route


# ! PR專用輪盤法
def wheel(fitness, path, insert_point):
    all_pro = [sum(i) for i in fitness]
    total_fitness = sum(all_pro)
    pick = np.random.random()
    p = 0
    for i in range(len(fitness)):
        p += all_pro[i] / total_fitness
        if p >= pick:
            path = path[i]
            fitness = fitness[i]
            insert_point = insert_point[i]
            break

    pick = np.random.random()
    p = 0
    total_fitness = sum(fitness)
    for i, j in enumerate(path):
        p += fitness[i] / total_fitness
        if p >= pick:
            # insert_point = insert_point[i]
            return j, insert_point


#! 輪盤PR2專用
def wheel2(fitness, path):
    total_fitness = sum(fitness)
    pick = np.random.random()
    p = 0
    for i, j in enumerate(path):
        p += fitness[i] / total_fitness
        if p >= pick:
            return j


# ! 路徑重新連結(輪盤法找最爛一點插入)(突變後專用)
def NPR2(dis_array, a, route):
    if route[-1] == 0:
        del route[-1]
    use_time, route = travel_time(dis_array, route)
    route.append(0)

    PR_list = [i for i in range(len(coords))]
    PR_list.append(0)
    for i in route:
        PR_list.remove(i)

    now = route[-2]
    next = route[-1]

    insert_point = []
    best_candidate = []
    best_ratio = []

    candidate_point = []
    w_candidate_point = []
    for i in PR_list:
        arrival_time = dis_array[now][i] + use_time
        if open_time[i] <= arrival_time <= close_time[i]:  # ! 找出不影響後面入場時間的點
            if open_time[next] <= arrival_time + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                candidate_point.append(i)

        elif open_time[i] > arrival_time and open_time[i] < close_time[i] - visit_time[i]:
            if open_time[next] < open_time[i] + visit_time[i] + dis_array[i][next] < close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                w_candidate_point.append(i)

    if len(candidate_point) != 0:  # !可以剛好進去玩的
        candidate_point, candidate_ratio, go = ratio1(
            candidate_point, now, next)

        insert_point.append(route.index(go, 1))
        best_ratio.append(candidate_ratio)
        best_candidate.append(candidate_point)

    if len(w_candidate_point) != 0:  # ! 要等開門的
        candidate_point, candidate_ratio,  go = ratio_w2(
            w_candidate_point, now, next, use_time)

        insert_point.append(route.index(go, 1))
        best_ratio.append(candidate_ratio)
        best_candidate.append(candidate_point)

    if len(candidate_point)+len(w_candidate_point) != 0:
        best_candidate, insert_point = wheel(
            best_ratio, best_candidate, insert_point)

        route.insert(insert_point, best_candidate)
        use_time, route = travel_time(dis_array, route)
    return route, len(candidate_point)+len(w_candidate_point)


# ? 路徑重新連結(貪心插入)(突變後專用)
def NPR_greedy(dis_array, a, route):
    if route[-1] == 0:
        del route[-1]
    use_time, route = travel_time(dis_array, route)
    route.append(0)

    PR_list = [i for i in range(len(coords))]
    PR_list.append(0)
    for i in route:
        PR_list.remove(i)

    now = route[-2]
    next = route[-1]

    insert_point = []
    best_candidate = []
    best_ratio = 0

    candidate_point = []
    w_candidate_point = []
    for i in PR_list:
        arrival_time = dis_array[now][i] + use_time
        if open_time[i] <= arrival_time <= close_time[i]:  # ! 找出不影響後面入場時間的點
            if open_time[next] <= arrival_time + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                candidate_point.append(i)

        elif open_time[i] > arrival_time and open_time[i] < close_time[i]:
            if open_time[next] < open_time[i] + visit_time[i] + dis_array[i][next] < close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and use_time + dis_array[now][i] + visit_time[i] + dis_array[i][0] <= close_time[0]:
                w_candidate_point.append(i)

    if len(candidate_point) != 0:  # !可以剛好進去玩的
        C, candidate_ratio, current, go = cal_ratio(
            candidate_point, now, next)

        if candidate_ratio > best_ratio:
            insert_point = route.index(go, 1)
            best_ratio = candidate_ratio
            best_candidate = C

    if len(w_candidate_point) != 0:  # ! 要等開門的
        C, candidate_ratio, current,  go = w_cal_ratio(
            w_candidate_point, now, next, use_time)
        if candidate_ratio > best_ratio:
            insert_point = route.index(go, 1)
            best_ratio = candidate_ratio
            best_candidate = C
    if len(candidate_point)+len(w_candidate_point) != 0:
        route.insert(insert_point, best_candidate)
        use_time, route = travel_time(dis_array, route)
    return route, len(candidate_point)+len(w_candidate_point)


# ! 路徑重新連結(突變後專用)(自插入點找最好)
def path_relinking2(dis_array, a, route):
    del route[-1]
    use_time, route = travel_time(dis_array, route)
    route.append(0)

    utmost = 0
    count = 0
    # a += 1
    a -= 1

    while use_time < close_time[0]:

        PR_list = [i for i in range(len(coords))]
        PR_list.append(0)
        for i in route:
            PR_list.remove(i)

        best_candidate = -9
        now = route[a-1]
        next = route[a]
        best_ratio = 0
        while a < len(route):  # ? 在整條路徑中，找出最好候選點
            candidate_point = []
            w_candidate_point = []
            for i in PR_list:
                arrival_time = dis_array[now][i] + use_time
                if open_time[i] <= arrival_time <= close_time[i]:  # ! 找出不影響後面入場時間的點
                    if open_time[next] <= arrival_time + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]):
                        candidate_point.append(i)

                elif open_time[i] > arrival_time and open_time[i] < close_time[i]:
                    if open_time[next] <= open_time[i] + visit_time[i] + dis_array[i][next] <= close_time[next] or (open_time[next] > arrival_time + visit_time[i] + dis_array[i][next] and close_time[next] >= open_time[next]) and close_time[next] < visit_time[next]+use_time + dis_array[now][next]:
                        w_candidate_point.append(i)
            if len(candidate_point) != 0:  # !可以剛好進去玩的
                candidate_point, candidate_ratio, go = ratio1(
                    candidate_point, now, next)
                insert_point = route.index(go, 1)
                best_ratio = candidate_ratio
                best_candidate = candidate_point
            if len(w_candidate_point) != 0:  # ! 要等開門的
                candidate_point, candidate_ratio,  go = ratio_w2(
                    w_candidate_point, now, next, use_time)
                insert_point = route.index(go, 1)
                best_ratio = candidate_ratio
                best_candidate = candidate_point

            a += 1

            # use_time, route = cal_time(dis_array, now, next, use_time, route)
            # a += 1
            # if a < len(route):
            #     now = route[a-1]
            #     next = route[a]
        if best_candidate != -9:
            ori_route = route

            best_candidate = wheel2(
                best_ratio, best_candidate)

            route.insert(insert_point, best_candidate)
            next = route[a-1]
            use_time, route = cal_time(dis_array, now, next, use_time, route)

            if use_time > utmost:
                utmost = use_time
                best_route = route
                count = 0
            else:
                count += 1
                if count >= 1:
                    break
            if use_time > close_time[0]:
                route = ori_route
                break
        else:
            use_time, route = travel_time(dis_array, route)
            return route

        # use_time, route = cal_time(dis_array, now, next, use_time, route)
        # a += 1
        if a < len(route):
            now = route[a-1]
            next = route[a]

    return best_route


# TODO 計算走到目前為止的時間
def cal_time(dis_array, now, next, use_time, route):
    if open_time[next] <= use_time + dis_array[now][next] <= close_time[next]:
        use_time += dis_array[now][next] + visit_time[next]
    elif open_time[next] > use_time + dis_array[now][next] and close_time[next] >= open_time[next]:
        use_time = open_time[next] + visit_time[next]
    else:  # !!!!!!!!!!!!!!!!!!!!!! 有錯誤
        if next != 0:
            route.remove(next)
            # print('時間錯誤')
        else:
            route.remove(now)
            # print('時間錯誤')
    return use_time, route


#! 計算候選點的比值，並輪盤選一(剛好進去玩的專用)
def ratio1(candidate_point, now, next):
    ratio = []
    for g in candidate_point:
        ratio.append(benefit[g] / (dis_array[now][g]))

    return candidate_point, ratio,  next


# ! 計算候選點的比值，並輪盤選一(等開門專用)
def ratio_w2(candidate_point, now, next, use_time):
    ratio = []
    for g in candidate_point:
        ratio.append(benefit[g] / (open_time[g] - use_time))

    return candidate_point, ratio,  next


# TODO 計算候選點的比值，取最高(剛好進去玩的專用)
def cal_ratio(candidate_point, now, next):
    best_ratio = 0
    for g in candidate_point:
        ratio = benefit[g] / (dis_array[now][g])
        if ratio > best_ratio:
            current = now
            go = next
            best_ratio = ratio
            B = g
    return B, best_ratio, current, go


# TODO 計算候選點的比值，取最高(等開門的專用)
def w_cal_ratio(candidate_point, now, next, use_time):
    best_ratio = 0
    for g in candidate_point:
        ratio = benefit[g] / (open_time[g] - use_time)
        if ratio > best_ratio:
            current = now
            go = next
            best_ratio = ratio
            B = g
    return B, best_ratio, current, go


# TODO 交配
def cross(path):
    z = 0
    L = np.random.randint(1, len(coords)/2)
    while z < 1:
        cur_pro = np.random.random()
        if cur_pro <= cross_pro:
            n = len(path)
            fitness = []
            [fitness.append(get_fitness(benefit, path[i])) for i in range(n)]
            fitness = np.array(fitness)

            parent1 = roulette_wheel(fitness, path)  # ? 利用輪盤法選出好的染色體
            parent2 = np.random.randint(0, len(path))

            if parent1 == parent2:
                continue

            r_1 = path[parent1].copy()  # ? 複製池(不會動到母體)
            r_2 = path[parent2].copy()

            while len(r_1) - len(r_2) < 0:  # ? 讓要複製的染色體長度一致
                r_1.insert(-1, dummy)
            while len(r_1) - len(r_2) > 0:
                r_2.insert(-1, dummy)

            gens_len = len(r_1)

            index1 = np.random.randint(1, gens_len-2)  # ? 第一切點
            index2 = np.random.randint(index1, gens_len-2)  # ? 第二切點

            if index1 == index2:
                continue

            temp_gen1 = r_1[index1:index2+1]  # 交配的基因片段1
            temp_gen2 = r_2[index1:index2+1]  # 交配的基因片段2

            # 交配
            r_1[index1:index2+1] = temp_gen2
            r_2[index1:index2+1] = temp_gen1

            # 消除衝突
            pos = index1 + len(temp_gen1)  # 插入交配基因片段的结束位置
            conflict1_ids, conflict2_ids = [], []
            [conflict1_ids.append(i) for i, v in enumerate(r_1) if v in r_1[index1:pos]
             and i not in list(range(index1, pos))]  # ? i引數；v為值
            [conflict2_ids.append(i) for i, v in enumerate(r_2) if v in r_2[index1:pos]
             and i not in list(range(index1, pos))]

            for i in conflict1_ids:
                r_1[i] = dummy
            for j in conflict2_ids:
                r_2[j] = dummy

            for i in range(len(r_1)):
                if dummy in r_1:
                    r_1.remove(dummy)
            for i in range(len(r_2)):
                if dummy in r_2:
                    r_2.remove(dummy)

            # 修正路線使之在規定時間內
            t, r_1 = travel_time(dis_array, r_1)
            t, r_2 = travel_time(dis_array, r_2)

            r_1 = path_relinking(dis_array, r_1)
            r_2 = path_relinking(dis_array, r_2)

            path[parent1] = r_1
            path[parent2] = r_2

        z += 1
    return path


#! 突變
def mutation(path):
    n = len(path)
    for i in range(n):
        cur_pro = np.random.random()  # ? 決定是否要突變
        # 不突變
        if cur_pro > mutation_pro:
            continue
        else:
            mutate_point = np.random.randint(1, len(path[i])-1)
            if mutate_point == 1:
                P = path[i].copy()
                del P[mutate_point]  # ! 選定基因以後刪除
                path[i] = NPR(dis_array, P)
            else:
                P = path[i].copy()
                del P[mutate_point:]
                point_num = -1
                while point_num != 0:
                    split = np.random.random()
                    if split >= 0.5:
                        P, point_num = NPR2(dis_array, mutate_point, P)
                    else:
                        P, point_num = NPR_greedy(dis_array, mutate_point, P)
                # path[i] = path_relinking2(dis_array, mutate_point, P)
                path[i] = P

    return path


def draw_H(citys, best_gens):
    plt.ion()
    x_data = [(v[0]) for i, v in enumerate(citys)]
    y_data = [(v[1]) for i, v in enumerate(citys)]
    parent1, parent2 = [], []
    plt.cla()
    plt.scatter(x_data, y_data, s=10, c='blue')
    for i, v in enumerate(best_gens):
        parent2.append(citys[v])
        plt.annotate(text=v, xy=citys[v], xytext=(-4, 5),
                     textcoords='offset points', fontsize=10)
    x_data = [(v[0]) for i, v in enumerate(parent2)]
    y_data = [(v[1]) for i, v in enumerate(parent2)]
    # plt.title("OPTW", fontsize=25)
    # plt.xlim(min(x_data) * 1.2, max(x_data)*1.2)
    # plt.ylim(min(y_data) * 1.2, max(y_data)*1.2)
    plt.plot(x_data, y_data, 'r-')
    plt.pause(0.001)


if __name__ == '__main__':
    max_time = close_time[0]
    population_num = 100
    max_evolution_num = 500
    # choice_pro = 0.2
    cross_pro = 0.8
    mutation_pro = 0.15
    best_step_index = 0
    max_fitness = 0
    dummy = -1
    dis_array = compute_dis_mat(coords)

    for CCC in range(5):
        S = datetime.datetime.now()
        iter = [0]
        H_scores = [0]
        B_scores = [0]
        path = []
        best_step_index = 0
        max_fitness = 0
        path = init_greedy(dis_array, benefit)
        path = init_grasp(dis_array, population_num, benefit, path)


#!----------------------演算法程序-----------------------
        for step in trange(max_evolution_num):
            path = choice(path)
            path = cross(path)
            path = mutation(path)

            Fitness = []
            [Fitness.append(get_fitness(benefit, path[i]))
             for i in range(len(path))]

            best_path_idx = np.argmax(Fitness)  # ? 找到最好染色體位置

            if Fitness[best_path_idx] > max_fitness:
                max_fitness = Fitness[best_path_idx]
                Total_time, best_path = travel_time(
                    dis_array, path[best_path_idx])
                max_score = Evaluation(benefit, best_path)  # ? 更新最高獲益值
                # best_path = path[best_path_idx]  # ? 更新最佳染色體
                if best_step_index <= step:
                    best_step_index = step + 1

            if (sum(Fitness) / len(path)) == (Fitness[best_path_idx]):
                break

            # draw_H(coords, path[best_path_idx])
            # draw_H(coords, best_path)
            iter.append(step+1)
            H_scores.append(Evaluation(benefit, path[best_path_idx]))
            B_scores.append(max_score)
            if H_scores[-2] != H_scores[-1]:
                Count = 0
            else:
                Count += 1
                if Count == max_evolution_num*0.1:
                    break
            if max_score == BKS:
                break

            # print('第{}代：{}分'.format(step + 1, max_score))
            # print('第{}代：{}分，當前最佳：{}分'.format(
            # step + 1, max_score, Evaluation(benefit, path[best_path_idx])))

        print('{:=^100s}'.format('這是分隔線'))
        print('最佳分數為：{}分'.format(max_score))
        print('最佳路徑出現在第{}代，共拜訪了{}個點，其路徑為：\n{}'.format(
            best_step_index, len(best_path), best_path))
        print('所花費的時間為：', Total_time)

        # path = path_relinking(dis_array, path[0])

        # draw_H(coords, best_path)
        # fig, axs = plt.subplots(1, 1, sharex=False, sharey=False)
        # axs.plot(iter, B_scores)
        E = datetime.datetime.now()
        print('本次執行時間：', (E-S))

        plt.ioff()
        plt.show()
