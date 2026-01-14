import pygame
import math
import sys
from pygame.locals import *
import time
from genome_decoder import GenomeDecoder
from genetic_algorithm import GeneticAlgorithm
from my_car import Car
from my_parking import Parking

# 初始化Pygame
pygame.init()
WIDTH, HEIGHT = 1200, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("小车停车入库")
clock = pygame.time.Clock()

# 颜色定义
BLUE_BG = (173, 216, 230)      # 背景浅蓝色
RED_TEXT = (255, 0, 0)         # 距离文字颜色

# 全局配置
PARK_WIDTH = 100                # 车位宽度
PARK_HEIGHT = 50             # 车位长度
PARK_SPACING = 30              # 车位间距
CAR_INIT_ANGLE = 0             # 初始朝向：0度 = 朝右（标准数学角度）✅
TARGET_PARKING_SPOT = ((230, 350), (330, 350), (230, 400), (330, 400)) # 目标车位坐标
RESET_INTERVAL = 9000  # 每次模拟限定时长


def create_regular_parking_lots():
    parking_rects = []
    start_x = 100
    start_y = 100
    for i in range(6):
        rect = Parking(start_x + i * (PARK_WIDTH + PARK_SPACING), start_y, PARK_WIDTH, PARK_HEIGHT, screen)
        parking_rects.append(rect)

    start_y = 350
    for i in range(6):
        rect = Parking(start_x + i * (PARK_WIDTH + PARK_SPACING), start_y, PARK_WIDTH, PARK_HEIGHT, screen)
        parking_rects.append(rect)

    return parking_rects


def create_scene():
    parking_rects = create_regular_parking_lots()
    obstacle_cars = []
    for idx, rect in enumerate(parking_rects):
        if idx != 7 :
            car_x, car_y = rect.get_rect_center()
            # 障碍车也使用 0° 朝右，保持一致
            obstacle_cars.append(Car(car_x, car_y, screen, angle=0))

    # 可控车初始位置（中间偏左），朝右
    controllable_car = Car(400, 300, screen, angle=CAR_INIT_ANGLE, is_controllable=True)
    return parking_rects, obstacle_cars, controllable_car


def draw_ui(reset_timer_seconds=None):
    font = pygame.font.SysFont(None, 24)
    tips = [

    ]

    # ✅ 新增：倒计时显示
    if reset_timer_seconds is not None:
        timer_text = f"Auto Reset: {reset_timer_seconds}s"
        timer_surf = font.render(timer_text, True, RED_TEXT)
        screen.blit(timer_surf, (20, 20 + len(tips) * 30))  # 放在提示下方



def get_actionSignal(player_car, decoder, gene):
    # 模拟 8 个传感器值（单位：米）
    sensor_values = player_car.sensor_values
    wheel, engine = decoder.decode(gene, sensor_values)
    # if engine == 1:
    #     player_car.speed = 2.0
    # elif engine == -1:
    #     player_car.speed = -2.0
    # else:
    #     player_car.speed = 0
    #
    # if wheel == 1:
    #     player_car.angle += player_car.turn_speed
    # elif wheel == -1:
    #     player_car.angle -= player_car.turn_speed

    # 优化1：计算最小传感器距离（离最近障碍的距离）
    min_sensor_dist = min([v for v in sensor_values if v > 0] + [4.0])  # 无障碍时取4米

    # 优化2：动态速度（离障碍越近，速度越小）
    speed_scale = min_sensor_dist / 4.0  # 0~1的缩放因子
    base_speed = 2.0
    if engine == 1:
        player_car.speed = base_speed * speed_scale  # 前进：近障碍则慢
    elif engine == -1:
        player_car.speed = -base_speed * speed_scale  # 后退：近障碍则慢
    else:
        player_car.speed = 0

    # 优化3：动态转向（离障碍越近，转向越小）
    turn_scale = min_sensor_dist / 4.0
    base_turn = 2.0
    if wheel == 1:
        player_car.angle += base_turn * turn_scale  # 左转：近障碍则慢转
    elif wheel == -1:
        player_car.angle -= base_turn * turn_scale  # 右转：近障碍则慢转



def reset_scene():
    '''重置场景'''
    return create_scene()  # 返回新的 (parking_rects, obstacle_cars, player_car)


counter = 1
has_collision = False
collision_count = 0
is_loop = False
is_timeout = False
def my_fitness_func(genome):
    global counter, has_collision, is_loop, is_timeout, collision_count
    has_collision = False
    is_loop = False
    is_timeout = False
    collision_count = 0
    play_car = simulate_car(genome)
    wheels = play_car.get_corners()
    loss = GeneticAlgorithm.car_loss(wheels, TARGET_PARKING_SPOT)
    # has_collision: bool = False, is_loop: bool = False,
    #                               is_timeout: bool = False
    # TODO
    fitness = GeneticAlgorithm.car_fitness_from_lossNew(loss, has_collision, is_loop, is_timeout)
    print(f'index:{counter}, loss:{loss}, fitness:{fitness}, collision:{collision_count} wheels:{wheels}, gene:{genome}')
    counter = counter +1
    return fitness


def start_autoparking():
    print("【自动泊车启动】—— 此处接入您的算法")
    ga = GeneticAlgorithm(
        genome_length=180,
        population_size=100,
        mutation_rate=0.05,
        elite_ratio=0.1,
        max_generations=50
    )

    ga.set_fitness_function(my_fitness_func)

    best_genome = ga.evolve()
    print("最优基因组:", best_genome)


def simulate(genome, decoder, parking_rects, obstacle_cars, player_car, is_auto=True):
    running = True

    # ✅ 新增：自动重置计时器（单位：毫秒）
    last_reset_time = pygame.time.get_ticks()
    # 记录最近位置用于检测循环
    position_history = []
    global has_collision, is_loop, is_timeout, collision_count

    while running:
        screen.fill(BLUE_BG)

        current_time = pygame.time.get_ticks()

        # 计算距离下次重置的剩余时间（毫秒）
        elapsed = current_time - last_reset_time
        remaining_ms = max(0, RESET_INTERVAL - elapsed)
        reset_timer_seconds = remaining_ms // 1000  # 转为整数秒

        # 自动重置逻辑
        if elapsed >= RESET_INTERVAL:
            print(f"【自动重置】{RESET_INTERVAL // 1000}秒到，模拟结束")
            return player_car

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        if is_auto:
            get_actionSignal(player_car, decoder, genome)

        player_car.update(obstacle_cars, parking_rects)
        for car in obstacle_cars:
            car.update([], parking_rects)

        # 绘制车位
        for rect in parking_rects:
            rect.draw()

        # 绘制车辆（先障碍物，再玩家）
        player_car.draw()
        player_car.draw_sensors()
        for car in obstacle_cars:
            if car.is_collision:
                print("发生碰撞，模拟结束")
                return player_car
                has_collision = True
                collision_count = collision_count + 1
            car.draw()

        # 检测循环：记录最近10个位置，如果重复超过3次则认为陷入循环
        current_pos = (round(player_car.x, 1), round(player_car.y, 1))
        position_history.append(current_pos)
        if len(position_history) > 10:
            position_history.pop(0)

        # 检查是否有重复位置
        if len(position_history) == 10:
            unique_positions = set(position_history)
            if len(unique_positions) < 6:  # 如果10个位置中少于6个不同位置
                # print("检测到循环，模拟结束")
                # return player_car
                # 步骤1：计算车位中心
                park_center_x = (TARGET_PARKING_SPOT[0][0] + TARGET_PARKING_SPOT[1][0]) / 2
                park_center_y = (TARGET_PARKING_SPOT[0][1] + TARGET_PARKING_SPOT[2][1]) / 2

                # 步骤2：计算小车到车位的方向角
                dx = park_center_x - player_car.x
                dy = park_center_y - player_car.y
                target_angle = math.degrees(math.atan2(-dy, dx))  # 匹配pygame的Y轴方向

                # 步骤3：调整朝向（向车位方向转10°）
                angle_diff = (target_angle - player_car.angle) % 360
                if angle_diff > 180:
                    angle_diff -= 360
                if angle_diff > 0:
                    player_car.angle += 10  # 右转朝向车位
                else:
                    player_car.angle -= 10  # 左转朝向车位

                # 步骤4：低速向车位移动
                player_car.speed = 1.0 if angle_diff < 90 else -1.0  # 朝向对则前进，否则后退
                player_car.update(obstacle_cars, parking_rects)
                position_history = []  # 清空历史
                print("检测到循环，智能调整朝向车位")
                is_loop = True

        draw_ui(reset_timer_seconds)
        pygame.display.flip()
        clock.tick(100)
    pygame.quit()
    sys.exit()

decoder = GenomeDecoder()

def simulate_car(genome):
    """
    基于输入基于，模拟自动泊车
    """
    # 这里应调用你的解码器 + 物理仿真
    global decoder
    parking_rects, obstacle_cars, player_car = reset_scene()
    player_car = simulate(genome, decoder, parking_rects, obstacle_cars, player_car)
    print(f'car angle:{player_car.angle} corners:{player_car.get_corners()} x:{player_car.x} y:{player_car.y}')
    print(f'car angle:{player_car.angle}  x:{player_car.x} y:{player_car.y}')
    return player_car

if __name__ == "__main__":
    ## 优秀个体
    gene=[1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    simulate_car(gene)