import pygame
WHITE = (225, 225, 225)
BLACK = (0, 0, 0)

class Parking:
    def __init__(self, start_x, start_y, width, height, screen, is_target=False):
        self.start_x = start_x
        self.start_y = start_y
        self.width = width
        self.height = height
        self.is_target = is_target
        self.screen = screen
    def get_corner(self):
        top_left = (self.start_x, self.start_y)
        top_right = (self.start_x + self.width, self.start_y)
        bottom_left = (self.start_x, self.start_y + self.height)
        bottom_right = (self.start_x + self.width, self.start_y + self.height)
        return (top_left, top_right, bottom_left, bottom_right)

    def get_rect_center(self):
        """根据左上角 (x, y) 和尺寸 (width, height) 返回矩形中心坐标"""
        return self.start_x + self.width / 2, self.start_y + self. height / 2

    def draw(self):
        rect = pygame.Rect(
            self.start_x,
            self.start_y,
            self.width,
            self.height
        )
        pygame.draw.rect(self.screen, BLACK, rect)
        pygame.draw.rect(self.screen, WHITE, rect, 3)