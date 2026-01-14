import math
import random
from typing import List, Callable, Tuple, Optional
import csv

# 类型别名
Bit = int  # 实际只取 0 或 1
Bits = List[Bit]
NumVec2 = Tuple[int, int]
RectanglePoints = Tuple[NumVec2, NumVec2, NumVec2, NumVec2]


class Genome:
    def __init__(self, genes):
        if not all(g in (0, 1) for g in genes):
            raise ValueError("All genes must be 0 or 1")
        self.genes = list(genes)

    def __repr__(self):
        #return f"Genome(len={len(self.genes)})"
        return f"Genome({self.genes})"

    def __len__(self):
        return len(self.genes)

    def __getitem__(self, index):
        return self.genes[index]

    def __setitem__(self, index, value):
        if value not in (0, 1):
            raise ValueError("Gene value must be 0 or 1")
        self.genes[index] = int(value)

    def copy(self) -> "Genome":
        return Genome(self.genes[:])


class GeneticAlgorithm:
    """
    遗传算法控制器：用于进化基因组以优化停车任务

    外部需提供：
      - 基因组长度（默认 180）
      - 适应度函数：genome -> float
      - 可选：轮子位置计算函数（若 fitness 依赖物理模拟）
    """

    def __init__(
        self,
            path,
        genome_length: int = 180,
        population_size: int = 100,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        max_generations: int = 40,
    ):
        self.genome_length = genome_length
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.max_generations = max_generations
        self.path = path

        # 内部状态
        self._population: List[Genome] = []
        self._fitness_cache: dict = {}

    def _create_random_genome(self) -> Genome:
        genes = [1 if random.random() < 0.5 else 0 for _ in range(self.genome_length)]
        return Genome(genes)

    def initialize_population(self):
        """初始化第一代种群"""
        self._population = [self._create_random_genome() for _ in range(self.population_size)]
        self._fitness_cache.clear()

    def set_fitness_function(self, func: Callable[[Genome], float]):
        """
        设置适应度函数
        函数签名: fitness_func(genome: Genome) -> float (越大越好)
        """
        self._fitness_func = func

    def _mutate(self, genome: Genome) -> Genome:
        new_genes = genome.genes[:]
        for i in range(len(new_genes)):
            if random.random() < self.mutation_rate:
                new_genes[i] = 1 - new_genes[i]
        return Genome(new_genes)

    def _crossover(self, parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
        """单点随机交叉（简化版：逐位按概率选择）"""
        child1_genes = []
        child2_genes = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child1_genes.append(parent1[i])
                child2_genes.append(parent2[i])
            else:
                child1_genes.append(parent2[i])
                child2_genes.append(parent1[i])
        return Genome(child1_genes), Genome(child2_genes)

    def _weighted_random_choice(self, items: List[Genome], weights: List[float]) -> Genome:
        total = sum(weights)
        if total == 0:
            return random.choice(items)
        r = random.uniform(0, total)
        upto = 0
        for item, w in zip(items, weights):
            if upto + w >= r:
                return item
            upto += w
        return items[-1]  # fallback

    def _evaluate_population(self) -> List[float]:
        fitnesses = []
        for genome in self._population:
            key = tuple(genome.genes)
            if key not in self._fitness_cache:
                self._fitness_cache[key] = self._fitness_func(genome)
            fitnesses.append(self._fitness_cache[key])
        return fitnesses

    def evolve(self) -> Genome:
        """
        执行完整遗传算法流程，返回最优个体
        """
        if not hasattr(self, '_fitness_func'):
            raise RuntimeError("必须先调用 set_fitness_function()")

        self.initialize_population()

        for generation in range(self.max_generations):
            # 评估适应度
            fitnesses = self._evaluate_population()

            # 按适应度排序（降序）
            sorted_pairs = sorted(zip(self._population, fitnesses), key=lambda x: x[1], reverse=True)
            sorted_population = [g for g, _ in sorted_pairs]

            # 精英保留
            elite_count = int(self.elite_ratio * self.population_size)
            new_population = [g.copy() for g in sorted_population[:elite_count]]

            # 轮盘赌选择 + 交叉 + 变异
            weights = [f for _, f in sorted_pairs]
            while len(new_population) < self.population_size:
                parent1 = self._weighted_random_choice(sorted_population, weights)
                parent2 = self._weighted_random_choice(sorted_population, weights)
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(child2))

            self._population = new_population[:self.population_size]
            self.writeLog(generation)
            print(f'{generation} done++++++++++++++++')

        # 返回最终最优个体
        final_fitnesses = self._evaluate_population()
        best_idx = final_fitnesses.index(max(final_fitnesses))
        return self._population[best_idx].copy()

    def writeLog(self,index):
        """记录日志"""
        final_fitnesses = self._evaluate_population()
        best_idx = final_fitnesses.index(max(final_fitnesses))
        best_genome = self._population[best_idx].copy()
        best_fitness = self._fitness_func(best_genome)

        total_fitness = sum(final_fitnesses)
        average_fitness = total_fitness / len(final_fitnesses)

        file_path = self.path
        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[index, average_fitness, best_fitness, best_genome]])

    @staticmethod
    def car_loss(wheels: RectanglePoints, target: RectanglePoints) -> float:
        """计算车辆与目标车位的平均欧氏距离"""
        return sum(
            math.hypot(w[0] - t[0], w[1] - t[1])
            for w, t in zip(wheels, target)
        ) / 4.0



    @staticmethod
    def car_fitness_from_loss(loss: float, has_collision: bool = False, is_loop: bool = False,
                              is_timeout: bool = False) -> float:
        """
        增强版适应度函数：
        - loss越小（离车位越近），适应度越高
        - 碰撞/循环/超时大幅惩罚适应度(待优化)
        """
        base_fitness = 1.0 / (loss + 1e-6)

        # 惩罚项：碰撞/循环/超时直接将适应度降至1%
        if has_collision:
            base_fitness *= 0.01

        # 额外奖励：离车位足够近（<50像素）时翻倍奖励
        if loss < 50:
            print()
            base_fitness *= 2.0

        return base_fitness



# ==================== 使用示例 ====================

TARGET_PARKING_SPOT: RectanglePoints = ((0, 0), (2, 0), (2, 2), (0, 2))

# 模拟函数（由你的物理引擎或神经网络提供）
def simulate_car(genome) -> RectanglePoints:
    # 这里应调用你的解码器 + 物理仿真
    # 返回：基于输入基因模拟后的汽车的四角坐标
    return ((10, 10), (12, 10), (12, 12), (10, 12))


def my_fitness_func(genome) -> float:
    wheels = simulate_car(genome)
    loss = GeneticAlgorithm.car_loss(wheels, TARGET_PARKING_SPOT)
    return GeneticAlgorithm.car_fitness_from_loss(loss)

if __name__ == "__main__":
    # 启动进化
    ga = GeneticAlgorithm(
        genome_length=180,
        population_size=100,
        mutation_rate=0.05,
        elite_ratio=0.2,
        max_generations=50
    )
    ga.set_fitness_function(my_fitness_func)

    best_genome = ga.evolve()
    print("最优基因组:", best_genome)