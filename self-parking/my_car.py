import math
import pygame
from genetic_algorithm import GeneticAlgorithm

RED = (255, 0, 0)
WHITE = (225, 225, 225)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 215, 0)
GRAY = (150, 150, 150)


TARGET_PARKING_SPOT = ((230, 350), (330, 350), (230, 400), (330, 400)) # 目标车位坐标
WIDTH, HEIGHT = 1200, 600
SENSOR_MAX_DIST = 400          # 传感器最大距离（像素）=4米
SENSOR_UPDATE_INTERVAL = 100   # 传感器更新间隔（毫秒）
CAR_SIZE = (90, 45)            # 车辆尺寸（长，宽）— 注意：现在长沿X轴
CAR_INIT_ANGLE = 0             # 初始朝向：0度 = 朝右（标准数学角度）✅

class Car:
    def __init__(self, x, y,screen, angle=0, is_controllable=False):
        self.x = x
        self.y = y
        self.angle = angle  # 0=右, 90=上, 180=左, 270=下
        self.length, self.width = CAR_SIZE  # length 沿朝向
        self.is_controllable = is_controllable
        self.is_collision = False
        self.screen = screen


        if is_controllable:
            self.speed = 0
            self.turn_speed = 2
            self.last_sensor_update = 0
            self.sensor_angles = [0, 45, 90, 135, 180, 225, 270, 315]
            self.sensor_values = [0.0] * 8
            self.loss = 0.0

    def update(self, obstacle_cars, parking_rects):
        if self.is_controllable:
            # 检查是否与任意障碍车发生碰撞
            for car in obstacle_cars:
                if self.is_colliding_aabb(car):
                    car.is_collision =True
                    #print('发生碰撞！！')
            rad = math.radians(self.angle)
            self.x += self.speed * math.cos(rad)
            self.y -= self.speed * math.sin(rad)  # Y轴向下，取反

            # 边界限制
            half_l = self.length / 2
            half_w = self.width / 2
            self.x = max(half_l, min(WIDTH - half_l, self.x))
            self.y = max(half_w, min(HEIGHT - half_w, self.y))

            # 更新传感器
            # 用于获取自 pygame.init() 被调用以来经过的毫秒数
            current_time = pygame.time.get_ticks()
            if current_time - self.last_sensor_update >= SENSOR_UPDATE_INTERVAL:
                self.update_sensors(obstacle_cars, parking_rects)
                self.update_loss()
                self.last_sensor_update = current_time

    def update_loss(self):
        wheels = self.get_corners()
        self.loss = GeneticAlgorithm.car_loss(wheels, TARGET_PARKING_SPOT)

    def get_corners(self):
        """
        返回车辆四角的世界坐标。
        :return: 四个角的坐标 (左上, 右上, 左下, 右下)。
        """
        half_length = self.length / 2
        half_width = self.width / 2

        # 定义本地坐标系下的四个角点（假设车头向右）
        local_points = [
            (-half_width, half_length),  # 左前
            (half_width, half_length),  # 右前
            (-half_width, -half_length),  # 左后
            (half_width, -half_length)  # 右后
        ]

        # 将本地坐标转换为世界坐标
        world_points = []
        angle_rad = math.radians(self.angle)
        for x_local, y_local in local_points:
            # 先绕原点旋转再平移
            rotated_x = x_local * math.cos(angle_rad) - y_local * math.sin(angle_rad)
            rotated_y = x_local * math.sin(angle_rad) + y_local * math.cos(angle_rad)
            world_x = self.x + rotated_x
            world_y = self.y - rotated_y  # 注意Y轴方向

            world_points.append((world_x, world_y))

        return world_points

    def update_sensors(self, obstacle_cars, parking_rects):
        """优化：仅检测障碍车辆，不检测车位；未命中返回0"""
        for i in range(8):
            sensor_angle = self.angle + self.sensor_angles[i]
            rad = math.radians(sensor_angle)
            start_x, start_y = self.x, self.y
            end_x = start_x + SENSOR_MAX_DIST * math.cos(rad)
            end_y = start_y - SENSOR_MAX_DIST * math.sin(rad)  # 注意Y轴方向

            min_dist = SENSOR_MAX_DIST  # 初始化为最大距离

            # 只检测其他车辆（障碍车），不检测车位！
            for car in obstacle_cars:
                if car == self:
                    continue
                dist = self._ray_to_car_distance(start_x, start_y, end_x, end_y, car)
                if dist < min_dist:
                    min_dist = dist

            # 转换为米，并处理未命中情况
            if min_dist >= SENSOR_MAX_DIST:
                self.sensor_values[i] = 0.0  # 未检测到障碍物 → 0
            else:
                dist_m = min_dist / 100.0  # 像素转米（100px = 1m）
                # 限制最小显示值为0.01m（避免显示0.00但实际有障碍）
                self.sensor_values[i] = max(0.01, round(dist_m, 2))

    def _ray_to_car_distance(self, rx1, ry1, rx2, ry2, car):
        """计算射线到车辆包围盒的最近交点距离"""
        # 车辆包围盒（以中心为原点）
        half_l = car.length / 2
        half_w = car.width / 2
        left = car.x - half_l
        right = car.x + half_l
        top = car.y - half_w
        bottom = car.y + half_w

        rect_edges = [
            ((left, top), (right, top)),  # 上边
            ((right, top), (right, bottom)),  # 右边
            ((right, bottom), (left, bottom)),  # 下边
            ((left, bottom), (left, top))  # 左边
        ]

        closest_dist = float('inf')
        for p1, p2 in rect_edges:
            intersect = self._line_intersection((rx1, ry1), (rx2, ry2), p1, p2)
            if intersect:
                dist = math.hypot(rx1 - intersect[0], ry1 - intersect[1])
                if dist < closest_dist:
                    closest_dist = dist

        return closest_dist if closest_dist != float('inf') else SENSOR_MAX_DIST

    # 在Car类中增加一个碰撞检测方法
    def detect_collision(self, other_car):
        # 计算两辆车的距离
        dist_x = self.x - other_car.x
        dist_y = self.y - other_car.y
        distance = math.hypot(dist_x, dist_y)

        # 如果距离小于两车半长之和，则认为发生碰撞
        if distance < (self.width / 2 + other_car.width / 2):
            return True
        return False

    def is_colliding_aabb(self, car2):
        """
        使用 AABB（轴对齐包围盒）判断两辆车是否碰撞。
        忽略 angle，将每辆车视为中心在 (x, y)、宽 CAR_SIZE[1]、长 CAR_SIZE[0] 的矩形（轴对齐）。
        """
        # 半长和半宽
        half_l1 = self.length / 2
        half_w1 = self.width / 2
        half_l2 = car2.length / 2
        half_w2 = car2.width / 2

        # 计算 AABB 的边界
        left1 = self.x - half_l1
        right1 = self.x + half_l1
        top1 = self.y + half_w1  # 假设 y 向上为正
        bottom1 = self.y - half_w1

        left2 = car2.x - half_l2
        right2 = car2.x + half_l2
        top2 = car2.y + half_w2
        bottom2 = car2.y - half_w2

        # AABB 碰撞条件：两个矩形在 x 和 y 轴上都重叠
        x_overlap = not (right1 < left2 or right2 < left1)
        y_overlap = not (top1 < bottom2 or top2 < bottom1)

        return x_overlap and y_overlap

    def _line_intersection(self, a1, a2, b1, b2):
        """计算两条线段的交点（标准算法）"""
        x1, y1 = a1
        x2, y2 = a2
        x3, y3 = b1
        x4, y4 = b2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        return None

    def draw(self):
        # 创建表面（以车辆中心为原点）
        surf_width = self.length * 2
        surf_height = self.width * 2
        car_surf = pygame.Surface((surf_width, surf_height), pygame.SRCALPHA)
        cx, cy = surf_width // 2, surf_height // 2

        front_color = RED if self.is_controllable else (189, 189, 189)

        pygame.draw.circle(car_surf, front_color, (cx + self.length // 4, cy ), self.width//2)

        body_color = YELLOW if self.is_controllable else GRAY
        body_color = RED if self.is_collision else body_color
        # 2. 车身（主矩形）
        pygame.draw.rect(car_surf, body_color, [
            cx - self.length // 2,
            cy - self.width // 2,
            self.length // 4*3,
            self.width
        ])


        # 6. 车头灯（右侧）
        light_r = 4
        pygame.draw.circle(car_surf, YELLOW,
                           (cx + self.length // 2 - 5, cy - self.width // 4), light_r)
        pygame.draw.circle(car_surf, YELLOW,
                           (cx + self.length // 2 - 5, cy + self.width // 4), light_r)

        # 旋转并绘制
        rotated = pygame.transform.rotate(car_surf, self.angle)
        rect = rotated.get_rect(center=(int(self.x), int(self.y)))
        self.screen.blit(rotated, rect.topleft)
        self.is_collision = False

    def draw_sensors(self):
        if not self.is_controllable:
            return

        font = pygame.font.SysFont(None, 18)
        for i in range(8):
            sensor_angle = self.angle + self.sensor_angles[i]
            rad = math.radians(sensor_angle)
            # dist_px = self.sensor_values[i] * 100
            end_x = self.x + 100 * math.cos(rad)
            end_y = self.y - 100 * math.sin(rad)

            color = GREEN if self.sensor_values[i] > 0 else GRAY
            pygame.draw.line(self.screen, color, (self.x, self.y), (end_x, end_y), 2)

            text = font.render(f"{self.sensor_values[i]:.2f}m", True, RED)
            text_x = end_x + 10 * math.cos(rad)
            text_y = end_y - 10 * math.sin(rad)
            self.screen.blit(text, (text_x, text_y))
        text_loss = font.render(f"{self.loss:.2f}m", True, WHITE)
        self.screen.blit(text_loss, (self.x, self.y))
